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

On "Paperspace" machines:
> GPU: A100-80G recommended for the training process.
> free-A6000 can work on evaluation process. 

---

## Training (BLIP-2 + LoRA)

The **BLIP-2** script fine-tunes LoRA adapters and evaluates on the held-out split. It also **tees** console logs to `runs/.../logs/*.log` and supports **resume**.

```bash
python train_blip.py \
  --jsonl /notebooks/pii_42k.jsonl \
  --images-root /notebooks/datasets/datasets/pii \
  --output-dir runs/blip_pii_lora 
```

---

## Training (Qwen2.5-VL + LoRA)

The **Qwen2.5-VL** script fine-tunes LoRA adapters and evaluates on the held-out split. It also **tees** console logs to `runs/.../logs/*.log` and supports **resume**.

```bash
PROGRESS_EVERY=25 python train_qwen.py
```

---

## Results (held-out 2k images)

### BLIP-2 (flan-t5-xl)

| Model         | Macro-F1 (per-class) | Any-PII F1 | Precision | Recall | Accuracy |
| ------------- | -------------------- | ---------- | --------- | ------ | -------- |
| Base BLIP-2   | 0.2227               | 0.787      | 0.780     | 0.794  | 0.785    |
| BLIP-2 + LoRA | 0.1089               | 0.626      | 0.514     | 0.800  | 0.522    |

### Qwen2.5-VL-7B-Instruct

| Model          | Macro-F1 | Macro-F1(Excluding 0-support classes) | Any-PII F1 | Precision | Recall | Accuracy |
| -------------- | -------- | ------------------------------------- | ---------- | --------- | ------ | -------- |
| Base Qwen2.5   | 0.5481   |                0.7306                 | 0.676      | 0.511     | 1.000  | 0.521    |
| Qwen2.5 + LoRA | 0.6763   |                0.8266                 | 0.774      | 0.631     | 1.000  | 0.707    |

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
