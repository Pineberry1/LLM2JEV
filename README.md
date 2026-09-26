# LLM2JEV

A general-purpose LLM can perform **Jev-style parallel verification**: give it
one shared state, then ask several independent questions about that state at
the same time. Each branch produces a short decision, such as Yes or No.

The key is to avoid recomputing the long state for every branch:

- **Radix Cache** keeps the shared state's KV cache so requests with the same
  token prefix can reuse it.
- **Shared-prefix attention** lets each branch attend to that cached state
  while keeping its own question and answer tokens separate.
- **Parallel serving** schedules the branches together, so one state can be
  checked against many candidates concurrently.

This repository uses **SGLang's RadixAttention** and its serving scheduler for
that pattern. The example below selects its **FlashInfer** attention backend.
We do not maintain a separate inference engine for the current path.

## Run

Install [SGLang](https://docs.sglang.ai/get_started/install.html) in a CUDA
environment and start one server with a local Qwen3-4B checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /path/to/Qwen3-4B \
  --attention-backend flashinfer \
  --host 127.0.0.1 --port 30000
```

In another shell, run 16 concurrent branches with either a 256- or
2,048-token shared prefix. The script records cold-cache and warm-cache
latency separately:

```bash
bash experiments/sglang/run_shared_prefix.sh /path/to/Qwen3-4B
```

The script calls SGLang's own `bench_serving` with its synthetic
`generated-shared-prefix` dataset. Results are written under
`experiments/sglang/results/`. See the [experiment notes](experiments/sglang/README.md)
for timing scope and interpretation. The synthetic benchmark checks the
parallel shared-prefix mechanism; it does not establish Yes/No accuracy.

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
