# LLM2JEV

Benchmark many short decisions that share one long input state. The current
inference path uses **SGLang**, rather than a custom model runner.

SGLang's **RadixAttention / Radix Cache** reuses the shared prefix KV cache
across requests. The branches then attend to that shared prefix through
SGLang's attention backend; the example below selects **FlashInfer**. We
measure the whole request, including server scheduling and generation, instead
of reporting only an attention kernel time.

## Run

Install [SGLang](https://docs.sglang.ai/get_started/install.html) in a CUDA
environment and start one server with a local Qwen3-4B checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /path/to/Qwen3-4B \
  --attention-backend flashinfer \
  --host 127.0.0.1 --port 30000
```

In another shell, run the shared-prefix workload. It uses 16 short branches
with either a 256- or 2,048-token shared prefix and records cold-cache and
warm-cache latency separately:

```bash
bash experiments/sglang/run_shared_prefix.sh /path/to/Qwen3-4B
```

The script calls SGLang's own `bench_serving` with its
`generated-shared-prefix` dataset. Results are written under
`experiments/sglang/results/`. See the [experiment notes](experiments/sglang/README.md)
for timing scope and interpretation.

## Current findings

The earlier custom-kernel experiment on one A40 showed **12.5× faster
attention-kernel time** for a 2,048-token prefix and 32 branches (40.8 µs
versus 508.6 µs). This is a kernel-only result, not a whole-request speedup.
With a fixed 16-branch workload, including prefix prefill raised total time
from **237 ms at 256 input tokens to 425 ms at 2,048 tokens**. These numbers
are preserved with raw data in [`experiments/legacy/`](experiments/legacy/).

**SGLang has not yet been measured on the remote GPU.** We will report its
cold- and warm-cache results here after the server is available; the legacy
numbers above must not be treated as SGLang results.

This project is independent of TypeSafe and OpenJev. The Qwen3-4B checkpoint
is a different model from OpenJev 4B, so latency alone does not compare their
decision quality.
