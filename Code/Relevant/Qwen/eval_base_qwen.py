#!/usr/bin/env python3
# Evaluate BASE Qwen2.5-VL-7B-Instruct (no adapters) on a JSONL of images.
# No LoRA/PEFT imports. No links to your finetuning code.

import os, sys, json, math, re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# ---- Model/processor (base only) ----
from transformers import AutoProcessor as _AutoProcessor
try:
    from transformers import Qwen2VLProcessor as ProcessorCls  # preferred
except Exception:
    ProcessorCls = _AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelCls
except Exception:
    from transformers import AutoModelForVision2Seq as AutoModelCls  # fallback


# ------------- CONFIG -------------
PII_TYPES = [
    "CREDIT_CARD_NUMBER","SSN","DRIVER_LICENSE","PERSONAL_ID",
    "PIN_CODE","MEDICAL_LETTER","PHONE_BILL","NAME","ADDRESS",
    "EMAIL","PHONE","OTHER_PII","BANK_ACCOUNT_NUMBER",
]
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
LONG_SIDE_MAX = 896
SEED = 42
# -----------------------------------

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def locate_images_root(search_from: Path) -> Optional[Path]:
    # Heuristic: look for a folder that contains 'sensitive' and 'non_sensitive'
    for p in search_from.rglob("*"):
        if p.is_dir() and (p/"sensitive").is_dir() and (p/"non_sensitive").is_dir():
            return p
    return None

def build_basename_index(root: Path) -> Dict[str, str]:
    idx = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            idx.setdefault(p.name, str(p.resolve()))
    return idx

def repair_path(original: str, root: Path, basename_index: Optional[Dict[str,str]]) -> Optional[str]:
    # 1) if already exists
    if os.path.exists(original):
        return original
    o = original.replace("\\", "/")
    low = o.lower()
    for anchor in ("sensitive/", "non_sensitive/"):
        k = low.find(anchor)
        if k != -1:
            rel = o[k:]
            cand = root / rel
            if cand.exists():
                return str(cand)
    if basename_index:
        base = os.path.basename(original)
        cand = basename_index.get(base)
        if cand and os.path.exists(cand):
            return cand
    return None

def cap_long_side(im: Image.Image, limit: int = LONG_SIDE_MAX) -> Image.Image:
    w, h = im.size
    m = max(w, h)
    if m <= limit:
        return im
    s = limit / float(m)
    return im.resize((int(round(w*s)), int(round(h*s))), Image.BICUBIC)

def normalize_labels(rec: Dict[str, Any]):
    is_sensitive = bool(rec.get("is_sensitive", False))
    types = rec.get("types", []) or []
    types = [t for t in types if t in PII_TYPES]
    labels = {t: False for t in PII_TYPES}
    for t in types:
        labels[t] = True
    return is_sensitive, labels

def build_prompt(class_list: List[str], width: int, height: int) -> str:
    return (
        "You are a PII auditor. Read the image and decide which PII types are present. "
        "Consider these classes ONLY:\n"
        + ", ".join(class_list) + ".\n"
        "Return ONLY JSON with fields:\n"
        "{\n"
        '  "labels": {<CLASS>: true|false, ...},\n'
        '  "evidence_text": ["short snippets that justify positives"]\n'
        "}\n"
        f"Image size (pixels): width={width}, height={height}."
    )

def parse_json_pred(s: str) -> Dict[str, bool]:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[^\n]*\n", "", s)  # drop ```json\n
        s = re.sub(r"\n```$", "", s)
    try:
        l = s.find("{"); r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            obj = json.loads(s[l:r+1])
            lbls = obj.get("labels", {}) or {}
            return {k: bool(lbls.get(k, False)) for k in PII_TYPES}
    except Exception:
        pass
    return {k: False for k in PII_TYPES}

def metrics_from_preds(y_true: List[Dict[str,bool]], y_pred: List[Dict[str,bool]]):
    per = {}
    for cls in PII_TYPES:
        t = [int(d.get(cls, False)) for d in y_true]
        p = [int(d.get(cls, False)) for d in y_pred]
        if sum(t) == 0 and sum(p) == 0:
            continue
        per[cls] = {
            "f1": f1_score(t, p, zero_division=0),
            "precision": precision_score(t, p, zero_division=0),
            "recall": recall_score(t, p, zero_division=0),
            "support": int(sum(t)),
        }
    macro_f1 = (sum(v["f1"] for v in per.values()) / max(len(per),1)) if per else 0.0
    t_bin = [int(any(d.values())) for d in y_true]
    p_bin = [int(any(d.values())) for d in y_pred]
    bin_metrics = {
        "f1": f1_score(t_bin, p_bin, zero_division=0),
        "precision": precision_score(t_bin, p_bin, zero_division=0),
        "recall": recall_score(t_bin, p_bin, zero_division=0),
        "accuracy": accuracy_score(t_bin, p_bin),
    }
    return {"per_class": per, "macro_f1": macro_f1, "binary": bin_metrics}

class PIIDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.recs = records
    def __len__(self): return len(self.recs)
    def __getitem__(self, idx):
        r = self.recs[idx]
        im = Image.open(r["image"]).convert("RGB")
        im = cap_long_side(im, LONG_SIDE_MAX)
        W, H = im.size
        _, labels = normalize_labels(r)
        prompt = build_prompt(PII_TYPES, W, H)
        return {
            "image": im,
            "prompt": prompt,
            "labels_dict": labels,
            "is_sensitive": bool(r.get("is_sensitive", False)),
            "image_path": r["image"],
        }

def main():
    import argparse
    ap = argparse.ArgumentParser("Evaluate BASE Qwen2.5-VL (no adapters) on a JSONL")
    ap.add_argument("--jsonl", required=True, type=Path, help="Path to your JSONL file.")
    ap.add_argument("--images-root", type=Path, default=None,
                    help="Root folder that contains the images (helps fix broken paths). If omitted, we try to locate.")
    ap.add_argument("--progress-every", type=int, default=25, help="Print progress every N images (0=off).")
    ap.add_argument("--debug-first", type=int, default=0, help="Print PRED_POS/GT_POS for first N items (0=none).")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    if not args.jsonl.exists():
        print(f"ERROR: {args.jsonl} not found", file=sys.stderr)
        sys.exit(2)

    # Device / dtype
    set_seed(SEED)
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    dtype = torch.bfloat16 if has_cuda else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Resolve images root
    images_root = args.images_root
    if images_root is None:
        # try: folder near JSONL, else search from /notebooks, else current dir
        candidate = locate_images_root(args.jsonl.parent)
        if candidate is None:
            base = Path("/notebooks")
            candidate = locate_images_root(base) or locate_images_root(Path(".")) or args.jsonl.parent
        images_root = candidate
    print(f"[ImagesRoot] {images_root}")

    # Read JSONL and repair paths if needed
    basename_index = build_basename_index(images_root) if images_root else None
    records, missing = [], 0
    for rec in read_jsonl(args.jsonl):
        p = rec.get("image", "")
        if not os.path.exists(p):
            fixed = repair_path(p, images_root, basename_index) if images_root else None
            if fixed is None:
                missing += 1
                continue
            rec["image"] = fixed
        records.append(rec)
    if missing:
        print(f"[PathFix] Skipped {missing} record(s) with unresolved image paths.")
    if not records:
        print("ERROR: No usable records after path fixing.", file=sys.stderr)
        sys.exit(3)

    print(f"[Dataset] N={len(records)} record(s)")

    # Load base model + processor (NO adapters)
    print("[Model] Loading base Qwen2.5-VL-7B-Instruct (no adapters)…")
    processor = ProcessorCls.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelCls.from_pretrained(MODEL_ID, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()
    model.config.use_cache = True

    pad_id = getattr(processor.tokenizer, "pad_token_id", None) or processor.tokenizer.eos_token_id
    eos_id = getattr(getattr(model, "generation_config", None), "eos_token_id", processor.tokenizer.eos_token_id)

    ds = PIIDataset(records)
    y_true, y_pred = [], []

    total = len(ds)
    for i, rec in enumerate(ds):
        im = rec["image"]
        prompt = rec["prompt"]

        msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text": prompt}]}]
        chat_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        inputs = processor(text=[chat_text], images=[im], return_tensors="pt", padding=True)
        inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

        # decode only continuation
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[:, prompt_len:]
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        pred = parse_json_pred(text)
        _, truth = normalize_labels({"is_sensitive": rec["is_sensitive"], "types": [k for k,v in rec["labels_dict"].items() if v]})
        y_true.append(truth)
        y_pred.append(pred)

        if args.debug_first and i < args.debug_first:
            pred_pos = [k for k,v in pred.items() if v]
            gt_pos   = [k for k,v in truth.items() if v]
            shown = text if len(text) <= 300 else (text[:300] + "…")
            print(f"[{i}] TEXT={json.dumps(shown)}")
            print(f"     PRED_POS={pred_pos}")
            print(f"     GT_POS={gt_pos}")

        if args.progress_every and ((i+1) % args.progress_every == 0 or (i+1) == total):
            pct = (i+1) * 100.0 / total
            print(f"[Progress] {i+1}/{total} ({pct:.1f}%)")

    # Metrics
    m = metrics_from_preds(y_true, y_pred)
    print("\n==== TEST METRICS ====")
    print(f"Macro-F1 (per-class): {m['macro_f1']:.4f}")
    b = m["binary"]
    print(f"Binary (any-PII)  ->  F1={b['f1']:.3f}  P={b['precision']:.3f}  R={b['recall']:.3f}  Acc={b['accuracy']:.3f}")
    for cls, mm in m["per_class"].items():
        print(f"{cls:>18s}  F1={mm['f1']:.3f}  P={mm['precision']:.3f}  R={mm['recall']:.3f}  (support={mm['support']})")
    print("\nDone.")

if __name__ == "__main__":
    # make sure TF noise doesn't break anything if present
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    main()
