# SGLang shared-prefix experiment

This is a serving benchmark using SGLang's built-in generated-shared-prefix
workload. One group contains 16 requests with the same 256- or 2,048-token
prefix, an 8-token branch question, and one generated token per request.
The requests are sent concurrently. Each shape is run once after `/flush_cache`
and then again without flushing. The first run measures a cold cache; the
second measures reuse of the prefix cached by the first run. Keep the server
running and use the same model and tokenizer for both passes.

The JSONL files contain SGLang's end-to-end latency and TTFT summaries plus
per-request details. A cold-cache run includes prefill. A warm-cache run
depends on the shared prefix remaining in the server's KV cache. This is a
synthetic latency workload, not a Jev accuracy test or an OpenJev comparison.

Run `bash experiments/sglang/run_shared_prefix.sh /path/to/Qwen3-4B` from the
repository root. Optionally pass a server URL and output parent directory as
the second and third arguments. See the root README for the server command.

The SGLang server provides RadixAttention and the FlashInfer attention backend;
this directory contains no custom attention implementation. Compare the
reported server version and attention backend when reproducing results.
