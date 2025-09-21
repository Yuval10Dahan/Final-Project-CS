#!/usr/bin/env bash
set -euo pipefail

# 0) Make sure we don't drag in system CUDA
unset LD_LIBRARY_PATH || true

# 1) Clean any older torch/transformers stack
pip uninstall -y torch torchvision torchaudio transformers accelerate peft timm tokenizers \
  qwen-vl-utils huggingface-hub safetensors sentencepiece protobuf >/dev/null 2>&1 || true
pip cache purge || true

# 2) Install a known-good stack (CUDA 12.1 wheels + recent Transformers)
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

pip install --no-cache-dir \
  "transformers==4.55.4" \
  "accelerate==1.10.0" \
  "peft==0.17.1" \
  "timm==1.0.19" \
  "qwen-vl-utils>=0.0.11" \
  "tokenizers==0.21.4" \
  "safetensors>=0.6.2" \
  "huggingface-hub>=0.34.0" \
  "sentencepiece>=0.2.1" \
  "protobuf<5"

pip install -U hf_transfer || true
pip uninstall -y deepspeed >/dev/null 2>&1 || true

# clean out conflicting installs
pip uninstall -y bitsandbytes torch torchvision torchaudio xformers >/dev/null 2>&1 || true

# install PyTorch built for CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1

# install bnb for cu121
pip install bitsandbytes==0.43.2

# (optional) xformers that matches this torch
pip install --index-url https://download.pytorch.org/whl/cu121 xformers || true

# ensure torchvision pinned (idempotent)
pip install -U --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torchvision==0.20.1"

