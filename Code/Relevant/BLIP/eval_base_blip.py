#!/usr/bin/env python3
# Base BLIP2 (Salesforce/blip2-flan-t5-xl) evaluation that MATCHES the LoRA script:
# - Strict JSON-skeleton prompt, robust parsing, clamp, resume, batching, dedup metrics
# - PLUS fallback: if no 'true/false' appears in the raw output, run class-by-class QA
#   to fill a JSON labels dict so results aren't all-zeros.

import os, sys, json, re, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Keep TF out of the way
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# ------------------ CONFIG (defaults) ------------------
MODEL_ID           = "Salesforce/blip2-flan-t5-xl"
DEFAULT_JSONL      = Path("/notebooks/pii_eval_data_2k_2.jsonl")
DEFAULT_IMAGESROOT = Path("/notebooks/Eval_data/Eval_data")
DEFAULT_LOG        = Path("runs/blip2_base/eval_on_pii_eval_data_2k_base_like_lora.log")
DEFAULT_PREDS      = Path("runs/blip2_base/eval_preds_base_like_lora.jsonl")

PII_TYPES = [
    "CREDIT_CARD_NUMBER","SSN","DRIVER_LICENSE","PERSONAL_ID","PIN_CODE",
    "MEDICAL_LETTER","PHONE_BILL","NAME","ADDRESS","EMAIL","PHONE",
    "OTHER_PII","BANK_ACCOUNT_NUMBER",
]
IMG_EXTS      = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
LONG_SIDE_MAX = 896
SEED          = 42
# ------------------------------------------------------

# BLIP-2 (avoid Auto* that can drag in TF)
from transformers import Blip2Processor as ProcessorCls
from transformers import Blip2ForConditionalGeneration as AutoModelCls

# ---------- utils ----------
def set_seed(seed=SEED):
    import random
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def build_basename_index(root: Path) -> Dict[str, str]:
    idx = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            idx.setdefault(p.name, str(p.resolve()))
    return idx

def repair_path(original: str, root: Path, basename_index: Optional[Dict[str,str]]) -> Optional[str]:
    if os.path.exists(original): return original
    o = original.replace("\\", "/"); low = o.lower()
    for anchor in ("sensitive/", "non_sensitive/"):
        k = low.find(anchor)
        if k != -1:
            rel = o[k:]; cand = root / rel
            if cand.exists(): return str(cand)
    if basename_index:
        base = os.path.basename(original)
        cand = basename_index.get(base)
        if cand and os.path.exists(cand): return cand
    return None

def cap_long_side(im: Image.Image, limit: int = LONG_SIDE_MAX) -> Image.Image:
    w, h = im.size; m = max(w, h)
    if m <= limit: return im
    s = limit / float(m)
    return im.resize((int(round(w*s)), int(round(h*s))), Image.BICUBIC)

def normalize_labels(rec: Dict[str, Any]):
    is_sensitive = bool(rec.get("is_sensitive", False))
    types = rec.get("types", []) or []
    types = [t for t in types if t in PII_TYPES]
    labels = {t: False for t in PII_TYPES}
    for t in types: labels[t] = True
    return is_sensitive, labels

# ---------- strict prompt with JSON skeleton ----------
def build_prompt(class_list: List[str], width: int, height: int) -> str:
    keys = ", ".join(class_list)
    json_skeleton = "{ " + ", ".join([f'\"{k}\": false' for k in class_list]) + ', "evidence_text": [] }'
    return (
        "You are a STRICT PII auditor. Decide which PII types are PRESENT in the image.\n"
        f"Return EXACTLY ONE JSON object with ALL of these keys: {keys} and 'evidence_text'.\n"
        "Rules:\n"
        "- Start your answer with '{' and end with '}'.\n"
        "- Every key must appear exactly once.\n"
        "- Use ONLY true/false booleans (no quotes) for each label.\n"
        "- If unsure, use false. Do NOT infer; mark true only if the visual content makes it unambiguous.\n"
        "- 'evidence_text' is a SHORT list (0–2 items) justifying any 'true' labels.\n\n"
        f"Image size: width={width}, height={height}.\n"
        "Return JSON only. Example template to follow:\n"
        + json_skeleton
    )

# ---------- clamp ----------
def clamp_pred(pred: Dict[str, bool], cap_true: int = 0, never: Optional[List[str]] = None) -> Dict[str, bool]:
    if never:
        for k in never:
            if k in pred:
                pred[k] = False
    if cap_true and cap_true > 0:
        priority = ["CREDIT_CARD_NUMBER", "SSN", "PERSONAL_ID", "PIN_CODE",
                    "MEDICAL_LETTER", "PHONE_BILL", "NAME", "ADDRESS",
                    "EMAIL", "PHONE", "OTHER_PII", "BANK_ACCOUNT_NUMBER"]
        pos = [k for k in priority if pred.get(k, False)]
        keep = set(pos[:cap_true])
        for k in pred.keys():
            if k not in keep:
                pred[k] = False
    return pred

# ---------- tolerant JSON/regex parser ----------
def parse_json_pred(s: str) -> Dict[str, bool]:
    out = {k: False for k in PII_TYPES}
    if not s: return out
    raw = s.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[^\n]*\n", "", raw)
        raw = re.sub(r"\n```$", "", raw)
    l = raw.find("{"); r = raw.rfind("}")
    if l != -1 and r != -1 and r > l:
        j = raw[l:r+1]
        j = j.replace("'", '"')
        j = re.sub(r'\bTrue\b', 'true', j)
        j = re.sub(r'\bFalse\b', 'false', j)
        j = re.sub(r",\s*([}\]])", r"\1", j)
        try:
            obj = json.loads(j)
            lbls = obj.get("labels", obj)
            if isinstance(lbls, dict):
                for k in PII_TYPES:
                    if k in lbls:
                        out[k] = bool(lbls[k])
                return out
        except Exception:
            pass
    low = raw.lower()
    for cls in PII_TYPES:
        cname = cls.lower()
        pat = re.compile(rf'["\']?{re.escape(cname)}["\']?\s*:\s*(true|false|1|0)', re.IGNORECASE)
        last = None
        for m in pat.finditer(low):
            last = m.group(1)
        if last is not None:
            out[cls] = True if last in ("true", "1") else False
    return out

def has_bool_tokens(s: str) -> bool:
    return bool(re.search(r"\btrue\b|\bfalse\b|: *1|: *0", s, re.I))

# ---------- QA fallback ----------
def build_qa_prompt(label: str, width: int, height: int) -> str:
    return (
        f"Question: Does this image clearly contain '{label}'?\n"
        "Answer strictly with 'yes' or 'no'.\n"
        f"(image size: {width}x{height})"
    )

def parse_yes_no(s: str) -> bool:
    t = s.strip().lower()
    if "yes" in t and "no" not in t: return True
    if "true" in t and "false" not in t: return True
    if re.search(r"\bno\b", t) or "false" in t: return False
    return False

def qa_fallback_batch(model, processor, samples, pad_id, eos_id, qa_max_new_tokens: int = 4) -> List[Dict[str,bool]]:
    """
    Run per-class yes/no across the whole batch for speed.
    Returns one dict per sample with booleans for all PII_TYPES.
    """
    device = next(model.parameters()).device
    out_by_sample = [{k: False for k in PII_TYPES} for _ in samples]
    images = [s.image for s in samples]

    for label in PII_TYPES:
        prompts = [build_qa_prompt(label, *s.size) for s in samples]
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                max_new_tokens=qa_max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        texts = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        for i, t in enumerate(texts):
            out_by_sample[i][label] = parse_yes_no(t)

    return out_by_sample

def metrics_from_preds(y_true: List[Dict[str,bool]], y_pred: List[Dict[str,bool]]):
    per = {}
    for cls in PII_TYPES:
        t = [int(x.get(cls, False)) for x in y_true]
        p = [int(x.get(cls, False)) for x in y_pred]
        if sum(t) == 0 and sum(p) == 0: continue
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

def record_key_from_fields(image_path: str, is_sensitive: bool, labels_dict: Dict[str, bool]) -> str:
    parts = [image_path, str(int(is_sensitive))]
    parts += [f"{k}:{int(labels_dict.get(k, False))}" for k in sorted(PII_TYPES)]
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

# ---------- dataset ----------
@dataclass
class Sample:
    image: Image.Image
    prompt: str
    labels_dict: Dict[str, bool]
    is_sensitive: bool
    image_path: str
    rec_key: str
    size: tuple

class PIIDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.recs = records
    def __len__(self): return len(self.recs)
    def __getitem__(self, idx):
        r = self.recs[idx]
        im = Image.open(r["image"]).convert("RGB")
        im = cap_long_side(im, LONG_SIDE_MAX)
        W, H = im.size
        is_sensitive, labels = normalize_labels(r)
        prompt = build_prompt(PII_TYPES, W, H)
        key = record_key_from_fields(r["image"], is_sensitive, labels)
        return Sample(
            image=im, prompt=prompt, labels_dict=labels,
            is_sensitive=is_sensitive, image_path=r["image"], rec_key=key,
            size=(W, H),
        )

# ================= main =================
def main():
    import argparse
    ap = argparse.ArgumentParser("BASE BLIP2 evaluator (LoRA-matching JSON prompt) with QA fallback.")
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--images-root", type=Path, default=DEFAULT_IMAGESROOT)
    ap.add_argument("--preds", type=Path, default=DEFAULT_PREDS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-by", choices=["record","image"], default="record",
                    help="record: skip exact records; image: skip any seen image path.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--debug-first", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256, help="for JSON-skeleton pass")
    ap.add_argument("--qa-max-new-tokens", type=int, default=4, help="for fallback yes/no")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--cap-true", type=int, default=0)
    ap.add_argument("--never-labels", type=str, default="")
    args = ap.parse_args()

    never_list = [x.strip() for x in args.never_labels.split(",") if x.strip()] if args.never_labels else []

    log_f = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(args.log, "w", encoding="utf-8")
    def say(msg: str):
        print(msg)
        if log_f: log_f.write(msg + "\n"); log_f.flush()

    if not args.jsonl.exists():      say(f"ERROR: JSONL not found: {args.jsonl}"); sys.exit(2)
    if not args.images_root.exists():say(f"ERROR: images-root not found: {args.images_root}"); sys.exit(2)

    set_seed(SEED)
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    dtype = torch.bfloat16 if has_cuda else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    say(f"[ImagesRoot] {args.images_root}")
    basename_index = build_basename_index(args.images_root)

    # Read & path-fix records
    records, missing = [], 0
    for rec in read_jsonl(args.jsonl):
        p = rec.get("image", "")
        if not os.path.exists(p):
            fixed = repair_path(p, args.images_root, basename_index)
            if fixed is None:
                missing += 1
                continue
            rec["image"] = fixed
        records.append(rec)
    if missing: say(f"[PathFix] Skipped {missing} record(s) with unresolved image paths).")
    if not records: say("ERROR: No usable records after path fixing."); sys.exit(3)
    say(f"[Dataset] N={len(records)} record(s)")

    # Resume setup
    done_keys, done_images = set(), set()
    if args.resume and args.preds.exists():
        count_rows = 0
        for row in read_jsonl(args.preds):
            count_rows += 1
            if "key" in row:
                done_keys.add(row["key"])
            if "image" in row:
                done_images.add(row["image"])
        say(f"[Resume] Found {len(done_keys) or len(done_images)} completed example(s) in {args.preds} "
            f"(rows={count_rows}, resume-by={args.resume_by}).")

    # Load BASE model + processor
    say("[Model] Loading BASE BLIP-2 model and processor…")
    processor = ProcessorCls.from_pretrained(MODEL_ID)
    model = AutoModelCls.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model.to(device); model.eval()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    pad_id = getattr(processor.tokenizer, "pad_token_id", None) or processor.tokenizer.eos_token_id
    eos_id = getattr(getattr(model, "generation_config", None), "eos_token_id", processor.tokenizer.eos_token_id)

    # Output preds
    args.preds.parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.preds, "a", encoding="utf-8")
    def write_pred(row: Dict[str, Any]):
        fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()

    # Eval list respecting resume mode
    ds = PIIDataset(records)
    def is_done(sample) -> bool:
        return (sample.rec_key in done_keys) if (args.resume_by=="record" and done_keys) \
               else (sample.image_path in done_images) if (args.resume_by=="image" and done_images) \
               else False

    to_eval_indices = [i for i in range(len(ds)) if not is_done(ds[i])]
    already = len(ds) - len(to_eval_indices)
    say(f"[Resume] Skipping {already} already-done example(s). Remaining: {len(to_eval_indices)}")

    # Batched loop
    bsz = max(1, int(args.batch_size))
    processed = already
    for start in range(0, len(to_eval_indices), bsz):
        idxs = to_eval_indices[start:start+bsz]
        batch = [ds[i] for i in idxs]
        images  = [s.image for s in batch]
        prompts = [s.prompt for s in batch]

        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        texts = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        # First pass: JSON parser
        preds = [parse_json_pred(t) for t in texts]

        # Fallback needed?
        need_qa = [i for i, t in enumerate(texts) if not has_bool_tokens(t)]
        if need_qa:
            fb_preds = qa_fallback_batch(
                model, processor, [batch[i] for i in need_qa],
                pad_id, eos_id, qa_max_new_tokens=args.qa_max_new_tokens
            )
            for j, i in enumerate(need_qa):
                preds[i] = fb_preds[j]

        for s, text, pred in zip(batch, texts, preds):
            # optional clamp
            if args.cap_true or never_list:
                pred = clamp_pred(pred, cap_true=args.cap_true, never=never_list)

            _, truth = normalize_labels({
                "is_sensitive": s.is_sensitive,
                "types": [k for k,v in s.labels_dict.items() if v]
            })

            if args.debug_first > 0:
                pred_pos = [k for k,v in pred.items() if v]
                gt_pos   = [k for k,v in truth.items() if v]
                print(f"[DEBUG] PRED_POS={pred_pos}  GT_POS={gt_pos}  img={s.image_path}")
                args.debug_first -= 1

            write_pred({
                "key": s.rec_key,
                "image": s.image_path,
                "pred_labels": pred,
                "gt_labels": truth,
                "raw": text.strip(),
            })
            processed += 1

        if args.progress_every and (processed % args.progress_every == 0 or processed == len(ds)):
            pct = processed * 100.0 / max(len(ds), 1)
            say(f"[Progress] {processed}/{len(ds)} ({pct:.1f}%)")

    fout.close()

    # Metrics from preds (full or partial) — DEDUP by key (or image if no key)
    preds_in_file = list(read_jsonl(args.preds))
    y_true, y_pred = [], []
    seen = set()
    for row in preds_in_file:
        k = row.get("key") or row.get("image")
        if k in seen:
            continue
        seen.add(k)
        if "gt_labels" in row and "pred_labels" in row:
            y_true.append(row["gt_labels"]); y_pred.append(row["pred_labels"])

    print("\n==== METRICS ====" + (" (full dataset)" if len(y_true)==len(ds) else f" (partial {len(y_true)}/{len(ds)})"))
    m = metrics_from_preds(y_true, y_pred)
    print(f"Macro-F1 (per-class): {m['macro_f1']:.4f}")
    b = m["binary"]
    print(f"Binary (any-PII)  ->  F1={b['f1']:.3f}  P={b['precision']:.3f}  R={b['recall']:.3f}  Acc={b['accuracy']:.3f}")
    for cls, mm in m["per_class"].items():
        print(f"{cls:>18s}  F1={mm['f1']:.3f}  P={mm['precision']:.3f}  R={mm['recall']:.3f}  (support={mm['support']})")

    print("\nDone.")
    if log_f: print(f"Log saved to: {args.log}"); log_f.close()

if __name__ == "__main__":
    main()
