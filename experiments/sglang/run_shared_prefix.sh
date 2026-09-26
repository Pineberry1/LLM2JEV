#!/usr/bin/env bash
set -euo pipefail

model_path=${1:?Pass the same model path used to launch the SGLang server}
server_url=${2:-http://127.0.0.1:30000}
output_parent=${3:-experiments/sglang/results}
run_dir="${output_parent}/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$run_dir"

for prefix_len in 256 2048; do
  common=(
    --backend sglang
    --base-url "$server_url"
    --model "$model_path"
    --dataset-name generated-shared-prefix
    --gsp-num-groups 1
    --gsp-prompts-per-group 16
    --gsp-system-prompt-len "$prefix_len"
    --gsp-question-len 8
    --gsp-output-len 1
    --num-prompts 16
    --max-concurrency 16
    --warmup-requests 0
    --output-details
  )
  python -m sglang.bench_serving "${common[@]}" \
    --flush-cache --output-file "$run_dir/p${prefix_len}_cold.jsonl"
  python -m sglang.bench_serving "${common[@]}" \
    --output-file "$run_dir/p${prefix_len}_warm.jsonl"
done

printf 'SGLang results: %s\n' "$run_dir"
