#!/usr/bin/env bash
set -euo pipefail

model_path=${1:?Pass a local Qwen3 model path}
mkdir -p results
for spec in "256 1" "256 16" "2048 32" "8192 64"; do
  read -r prefix branches <<< "$spec"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /opt/conda/bin/python \
    benchmarks/qwen3_attention.py \
    --model-path "$model_path" \
    --prefix-len "$prefix" --branches "$branches" \
    --suffix-len 8 --repeats 500 \
    --output "results/qwen3_p${prefix}_b${branches}.json"
done
