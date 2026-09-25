#!/usr/bin/env bash
set -euo pipefail

: "${QWEN_MODEL:?Set QWEN_MODEL to a local Qwen3-4B checkpoint}"
: "${OPENJEV_ROOT:?Set OPENJEV_ROOT to the downloaded OpenJev repository}"
: "${OPENJEV_MODEL:?Set OPENJEV_MODEL to the 4B v5 checkpoint folder}"

mkdir -p results

python benchmarks/flashinfer_smoke.py
python benchmarks/qwen3_attention.py \
  --model-path "$QWEN_MODEL" --prefix-len 2048 --branches 32 \
  --suffix-len 8 --repeats 200 --flashinfer \
  --output results/flashinfer_attention_p2048_b32.json
python benchmarks/qwen3_decisions.py \
  --model-path "$QWEN_MODEL" --prefix-len 2048 --branches 32 \
  --suffix-len 8 --repeats 5 --flashinfer \
  --output results/flashinfer_decisions_p2048_b32.json
python benchmarks/compare_openjev.py \
  --qwen-model "$QWEN_MODEL" --openjev-root "$OPENJEV_ROOT" \
  --openjev-model "$OPENJEV_MODEL" --qwen-backend flashinfer \
  --prefix-len 256 --branches 8 --repeats 5 \
  --output results/flashinfer_compare_p256_b8.json
python benchmarks/compare_openjev.py \
  --qwen-model "$QWEN_MODEL" --openjev-root "$OPENJEV_ROOT" \
  --openjev-model "$OPENJEV_MODEL" --qwen-backend flashinfer \
  --prefix-len 2048 --branches 16 --repeats 5 \
  --output results/flashinfer_compare_p2048_b16.json
