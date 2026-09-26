# Evaluation plan: can an LLM serve Jev-style decisions?

Status: **planned, not measured with SGLang**. The current
[`sglang/`](sglang/) workload measures synthetic shared-prefix serving only.
It does not answer JevBench questions or evaluate probability quality. The
earlier custom-kernel data in [`legacy/`](legacy/) is a systems prototype and
must be reported separately.

## Reference and scope

The [LLM2Jev reference repository](https://github.com/Yinsongxu/LLM2Jev/tree/0b3c1563c15fdd232fefb0b415e1c0d9bbf39d43)
uses prefill-only yes/no token scores, assembles Choice, Score and Noul
responses, and compares staged versus all-at-once submissions. Its
[JevBench report](https://github.com/Yinsongxu/LLM2Jev/blob/0b3c1563c15fdd232fefb0b415e1c0d9bbf39d43/docs/jevbench.md)
evaluates 231 public decisions. Its
[shared-prefix benchmark](https://github.com/Yinsongxu/LLM2Jev/blob/0b3c1563c15fdd232fefb0b415e1c0d9bbf39d43/docs/shared-prefix-benchmarks.md)
separates cold and warm cache behavior. These are reference methods, not
results for this repository.

The [JevBench public files](https://github.com/fstandhartinger/jevbench/tree/1bcc55eb6c8cffde2306b3db03ede39b61c6152a/datasets/public)
contain 72 original, 48 easy and 111 hard items. The current official
[v1.4 method](https://github.com/fstandhartinger/jevbench/blob/1bcc55eb6c8cffde2306b3db03ede39b61c6152a/docs/METHOD-v1.4.md)
also requires 308 sealed items unavailable for local evaluation. A run here
can claim a **231-item public-subset result**, not an official JevBench score
or ranking. Freeze the two upstream commits above before running.

## First prerequisite: a real decision endpoint

The current `sglang.bench_serving` script generates one token from synthetic
prompts. Before claiming Jev functionality, implement an adapter around
SGLang that accepts `POST /v1/systemone`, converts each Choice option, Score
level or Noul interpretation into an independent candidate prompt, reads
next-token yes/no scores during prefill, normalizes them and returns the
required typed fields and probability distribution. Check how the selected
tokenizer represents the two labels. Pin the prompt, token IDs and response
mapping before looking at benchmark results. Save raw responses, including
failures. This adapter is application glue; model execution and KV caching
remain inside SGLang.

## Experiments to run

| Priority | Question and controlled comparison | Workload | Report |
| --- | --- | --- | --- |
| 1 | Does the LLM produce valid and correct Jev-style answers? | All 231 public JevBench items, frozen prompt and model | Overall and per-tier accuracy; Choice/Score/Noul breakdown; invalid responses; probability calibration (ECE/Brier where defined); JevBench summary |
| 2 | Does prefix reuse improve a *cold* multi-candidate request? | Same model, prompts and candidate order; `all` versus staged submission; state lengths 256, 2,048, 8,192 and 2/8/16/32 candidates | End-to-end p50/p95, prefill or cached-token counts, tokens processed, speedup and crossover point |
| 3 | How much comes from Radix Cache and attention backend? | Compare all-at-once with cache enabled/disabled; compare staged versus all-at-once only with cache enabled; compare FlashInfer and another supported SGLang attention backend separately | End-to-end latency, throughput, GPU memory; document which component each toggle changes |
| 4 | Does parallel verification scale? | 1/8/16/32 candidate requests and 1/8/32 concurrent Jev requests; include both cold and warm prefixes | Decisions/s, p50/p95/p99 latency, GPU memory and failure rate |
| 5 | Is the answer stable when presentation changes? | Public paraphrase pairs, option-order permutations and repeated runs | Agreement, accuracy by variant, calibration shift and rank flips |
| 6 | Is prefill-only scoring preferable to generated answers? | Same base model, data and hardware; prefill-only logits versus constrained Yes/No or JSON generation | Accuracy, validity, output tokens and end-to-end latency |

Run priorities 1 and 2 before publishing a speed or Jev-equivalence claim.
Use JevBench's own scoring command for its metrics; do not substitute a few
hand-picked prompts. Where the public set has no labels for a proposed metric,
mark that metric unavailable instead of inventing one.

After the decision endpoint exists, run JevBench from the pinned JevBench
checkout (the SGLang model server alone is not a Jev endpoint):

```bash
python -m jevbench.cli run \
  --tasks datasets/public/original.jsonl,datasets/public/easy.jsonl,datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:31000 \
  --key-env '' --model qwen3-4b \
  --cost-basis no_billable_account_public_endpoint --reserve-usd 0 \
  --results RUN/results.jsonl --raw-dir RUN/raw --ledger RUN/ledger.jsonl

python -m jevbench.cli summarize \
  --tasks datasets/public/original.jsonl,datasets/public/easy.jsonl,datasets/public/hard.jsonl \
  --results RUN/results.jsonl --public-export RUN/summary.json
```

The endpoint address is an example for the future adapter; no service is
currently started on that port. For a claim about generalization beyond public
items, obtain an independent locked evaluation set or an official sealed run.

## Measurement protocol

- Use one pinned model checkpoint, tokenizer, SGLang version, prompt and
  hardware configuration within each controlled comparison. Record GPU model,
  CUDA/driver versions, dtype, attention backend and cache settings.
- Define *cold KV* as a loaded, kernel-warmed server after clearing the
  relevant prefix cache. Define *warm KV* as a measured request whose shared
  prefix was populated earlier. Keep model loading and kernel compilation out
  of both, and never compare a cold row with a warm row without labeling it.
- For cold multi-candidate tests, compare staged submission with all-at-once
  submission. A simultaneous cold batch may finish before any branch can
  populate Radix Cache, so a cache-enabled flag alone is not proof of reuse.
- Record individual request times and errors, not only a mean. Use at least
  five warm-up iterations and 30 measured repetitions per latency shape when
  the server is available; report medians and p95 with the raw sample count.
  Keep the number of branches fixed when comparing input lengths.
- Report wall-clock request latency and throughput separately from isolated
  kernel timings. For JevBench, preserve raw JSONL, the benchmark ledger,
  summary, model identity and run manifest under a dated experiment folder.
- Official Jev, local OpenJev and this Qwen model differ in weights, prompts,
  hosting and network path. Public benchmark accuracy can be compared on the
  same items; latency is descriptive unless measured under a matched setup.

## Result publication gate

The root README should show only a compact table from completed runs: public
JevBench accuracy and validity, cold/warm end-to-end p50/p95, and one matched
cache-ablation speedup. Link each row to its raw run folder. Until then, label
the SGLang path as unmeasured and keep legacy kernel numbers separate.
