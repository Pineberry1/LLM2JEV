#!/usr/bin/env bash
set -euo pipefail

qwen_model=${1:?Pass local Qwen3-4B path}
openjev_root=${2:?Pass downloaded OpenJev repository path}
openjev_model="$openjev_root/qwen3.5-4b-nli-v5"
mkdir -p results

for spec in "256 8" "2048 16"; do
  read -r prefix branches <<< "$spec"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="src:$openjev_root" /opt/conda/bin/python \
    benchmarks/compare_openjev.py \
    --qwen-model "$qwen_model" \
    --openjev-root "$openjev_root" --openjev-model "$openjev_model" \
    --prefix-len "$prefix" --branches "$branches" --repeats 5 \
    --output "results/compare_p${prefix}_b${branches}.json"
done
