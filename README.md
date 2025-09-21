# Sensitive Data Leakage Detection with Vision-Language Models (VLMs)

> Detect PII in document-like images using BLIP-2 and Qwen2.5-VL, fine-tuned with LoRA adapters. Includes a clean JSONL schema, reproducible training/eval scripts, and per-class + binary metrics. &#x20;

---

## Overview

This project explores multimodal (image+text) **sensitive-data/PII detection** with Vision-Language Models. We build a document-image dataset covering PII classes, fine-tune **Qwen2.5-VL-7B-Instruct** and **BLIP-2 (flan-t5-xl)** with **LoRA**, and evaluate both per-class (macro-F1) and binary “any-PII” detection. Results show clear benefits for Qwen2.5-VL + LoRA, and trade-offs for BLIP-2. &#x20;

---

## Key Contributions

* **Dataset & JSONL schema** with PII labels, balanced splits and an unseen 2k eval set.&#x20;
* **LoRA fine-tuning pipelines** for BLIP-2 and Qwen2.5-VL, base weights frozen, portable **checkpoint-4000** adapters.&#x20;
* **Structured generation**: models return strict JSON for robust parsing.&#x20;
* **Reproducible training/eval scripts** with resume-from-checkpoint, progress logging, and metrics.&#x20;

---

## Dataset

* **Total**: 42k images (≈50/50 sensitive vs. non-sensitive). Used **12k** for training due to hardware limits, held-out **2k** evaluation set.
* **Sensitive**: credit cards, IDs, bank letters/PIN, medical letters, phone bills, etc.
* **Non-sensitive**: ads, reports, emails, forms, memos, resumes, handwritten notes.

**PII classes**: `CREDIT_CARD_NUMBER, SSN, DRIVER_LICENSE, PERSONAL_ID, PIN_CODE, MEDICAL_LETTER, PHONE_BILL, NAME, ADDRESS, EMAIL, PHONE, OTHER_PII, BANK_ACCOUNT_NUMBER`.&#x20;

---

## Repo structure (suggested)

```
.
├─ Code/
│   └─ Relevant/
│      ├─ BLIP/
|      ├─ Qwen/
|      ├─ Utils/
│      └─ json_files/
├─ DATASETS/
│  ├─ Custom/
   └─ From the internet/
├─ Results/
├─ Final Presentation.pptx
├─ Final Report.pdf
├─ README.md
└─ requirements.txt
└─ setup.sh
```

---

## Setup

```bash
# Python 3.10+ recommended
pip install --no-cache-dir -r requirements.txt

chmod +x setup.sh
./setup.sh
```

On "Paperspace" servers:
> GPU: A100-80G recommended for the training process.
> free-A6000 can work on evaluation process. 

---

## Training (BLIP-2 + LoRA)

The **BLIP-2** script fine-tunes LoRA adapters and evaluates on the held-out split. It also **tees** console logs to `runs/.../logs/*.log` and supports **resume**.

```bash
python train_blip2.py \
  --log runs/blip_pii_12k_lora/logs/blip2_$(date +%Y%m%d_%H%M%S).log
```

**Environment knobs** (all optional):

* `EVAL_ONLY=0|1` (default 0)
* `ADAPTER_PATH=/path/to/runs/blip_pii_12k_lora/checkpoint-XXXX` (force using a specific checkpoint)
* `PROGRESS_EVERY=25` (progress prints during eval)
* `MAX_NEW_TOKENS=256`

**Resume after interruption**

Just re-run the same command; the script auto-detects the latest `checkpoint-XXXX` in `runs/blip_pii_12k_lora/` and resumes training (or evaluates if `EVAL_ONLY=1`).&#x20;

---

## Evaluation (Qwen2.5-VL + LoRA)

Use the provided evaluator to attach LoRA adapters to **Qwen2.5-VL-7B-Instruct** and run on a JSONL list of images. It “tees” logs and prints periodic progress:

```bash
python eval_qwen_lora.py \
  --jsonl /path/to/eval.jsonl \
  --images-root /path/to/images_root \
  --adapter-path runs/qwen_pii_12k_lora/checkpoint-4000 \
  --progress-every 25 \
  --debug-first 0 \
  --max-new-tokens 256 \
  --log runs/qwen/logs/qwen_$(date +%Y%m%d_%H%M%S).log
```

---

## Results (held-out 2k images)

### BLIP-2 (flan-t5-xl)

| Model         | Macro-F1 (per-class) | Any-PII F1 | Precision | Recall | Accuracy |
| ------------- | -------------------- | ---------- | --------- | ------ | -------- |
| Base BLIP-2   | 0.2227               | 0.787      | 0.780     | 0.794  | 0.785    |
| BLIP-2 + LoRA | 0.1089               | 0.626      | 0.514     | 0.800  | 0.522    |

*Base BLIP-2 is a stronger **general** detector (fewer false positives). LoRA helps **Credit Card Number** and sometimes **Name**, but increases false positives overall.*&#x20;

### Qwen2.5-VL-7B-Instruct

| Model          | Macro-F1 | Macro-F1\* | Any-PII F1 | Precision | Recall | Accuracy |
| -------------- | -------- | ---------- | ---------- | --------- | ------ | -------- |
| Base Qwen2.5   | 0.5481   | 0.7306     | 0.676      | 0.511     | 1.000  | 0.521    |
| Qwen2.5 + LoRA | 0.6763   | 0.8266     | 0.774      | 0.631     | 1.000  | 0.707    |

\*Excluding 0-support classes on this eval (e.g., EMAIL, MEDICAL\_LETTER). **Recall stays 1.0**, while **precision/accuracy improve substantially** → far fewer false alarms. &#x20;

---

## Why this works

* **Structured generation** (strict JSON) simplifies parsing and metric computation.
* **LoRA** adapts VLMs efficiently without touching base weights → cheap, portable adapters.
* **Balanced splits + unseen eval** provide honest generalization checks.&#x20;

---

## Reproduce

1. **Place data** under `datasets/pii/{sensitive,non_sensitive}` and the master `pii_42k.jsonl` in the repo root.
2. **Train** BLIP-2 adapters: `python train_blip2.py`.
3. **Evaluate** Qwen adapters (or BLIP-2): run the eval script with your adapter path and `--progress-every 25`.
4. **Check logs** under `runs/.../logs/*.log` and metrics printed to console.

---

## Hardware notes

* Trained on **A100-80G** (fastest). Works on **A5000/A6000** with smaller `PER_DEVICE_BATCH` and higher `ACCUM`.
* Uses **bf16**, **gradient checkpointing**, and TF32 where available to reduce memory.&#x20;

---

## Limitations & Future Work

* Some classes had **low/zero support** in the 2k eval, depressing macro-F1.
* Document-centric data may limit generalization to natural scenes or multilingual content.
* Future: expand class coverage, add real-world photos, tune prompts to cut false positives, and scale training on larger hardware.&#x20;

---

## Citation

If you use this code, dataset schema, or results, please cite the **Final Report** and **Presentation**:

* Final Report: *Sensitive Data Leakage Detection Based on Vision-Language Models*.&#x20;
* Final Presentation: *Sensitive Data Leakage Detection Based on Vision-Language Models*.&#x20;

---

## License

Add your preferred license (MIT/Apache-2.0/BSD-3-Clause). If using external datasets, respect their licenses/terms.

---

## Acknowledgements

Thanks to the open-source communities behind **Hugging Face Transformers**, **peft/LoRA**, **PyTorch**, and the researchers advancing privacy-aware VLMs. See report references for related work.&#x20;

---
