# LLM2JEV

An independent experiment on one long shared state and many small decisions. It
implements a Triton attention kernel that lets a group of branch queries reuse
each prefix K/V tile, plus a Qwen3 scoring path that reads only candidate token
logits. An optional FlashInfer backend uses its two-level
[`MultiLevelCascadeAttentionWrapper`](https://docs.flashinfer.ai/api/cascade.html)
for the same shared-prefix branch workload. It also includes a same-GPU comparison with a separate OpenJev 4B
checkpoint. This project is not affiliated with TypeSafe or the authors of
OpenJev.

## What is measured

- **Attention kernel:** each branch's final query attends to one shared prefix
  and its own suffix. Grouping 16 queries into one Triton program shares prefix
  K/V loads. The independent baseline computes the same attention one branch
  at a time. A float32 reference checks the output of both kernels.
- **Qwen3 decision path:** a common prefix is prefetched once. Branch suffixes
  are processed through all Qwen3-4B layers, and `Yes`/`No` probabilities come
  from two selected language-head rows. The benchmark separates prefix prefill
  from branch work.
- **OpenJev comparison:** the same state and binary questions are sent to
  OpenJev 4B v5 through its author's `predict_hypotheses` method, which shares a
  prefix across the options. Both arms run on the same A40 in BF16, one after
  the other. The comparison includes tokenization and model execution.
- **FlashInfer comparison:** the shared prefix and independent branch suffixes
  are placed in two levels of paged KV cache. Plans and packed prefix pages are
  prepared once per state; suffix page writes are included in complete Qwen3
  decision timings. FlashInfer and Triton receive the same Q/K/V and model.

Qwen3-4B and OpenJev 4B v5 are different trained models with different prompt
templates and readouts. This is a systems comparison for the stated workload,
not a controlled architecture or accuracy comparison. These numbers do not
measure TypeSafe's hosted Jev service.

## Reproduce

Requires an NVIDIA GPU, Python 3.10+, PyTorch with CUDA, Triton, Transformers,
and the model weights. The model weights are not stored in this repository.

```bash
pip install -e '.[model]'
python -m unittest discover -s tests -v
python -m compileall -q src tests benchmarks

python benchmarks/qwen3_attention.py \
  --model-path /path/to/Qwen3-4B --prefix-len 2048 --branches 32 \
  --output results/qwen3_attention.json

python benchmarks/qwen3_decisions.py \
  --model-path /path/to/Qwen3-4B --prefix-len 2048 --branches 32 \
  --output results/qwen3_decisions.json
```

For the OpenJev comparison, download the author's checkpoint and inference
helper at the pinned revision:

```bash
hf download AlexWortega/openjev \
  --revision 058a6c24911b46d908fbe23541390f8af3df3e4d \
  --include 'qwen3.5-4b-nli-v5/*' --include modeling_openjev.py \
  --local-dir models/openjev-4b-v5

python benchmarks/compare_openjev.py \
  --qwen-model /path/to/Qwen3-4B \
  --openjev-root models/openjev-4b-v5 \
  --openjev-model models/openjev-4b-v5/qwen3.5-4b-nli-v5 \
  --prefix-len 256 --branches 8 --repeats 3 \
  --output results/compare.json
```

The OpenJev checkpoint is [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev),
`qwen3.5-4b-nli-v5` (MIT on its model card). It is a Qwen3.5 sequence
classifier, not TypeSafe Jev's private weights. The separate
[`openjev/openjev`](https://huggingface.co/openjev/openjev) repository is a 27B
model and is not the 4B checkpoint used here.

To run the optional FlashInfer comparison, install a FlashInfer build that
matches your PyTorch and CUDA environment, then run:

```bash
python benchmarks/flashinfer_smoke.py
python benchmarks/qwen3_attention.py \
  --model-path /path/to/Qwen3-4B --prefix-len 2048 --branches 32 --flashinfer
python benchmarks/qwen3_decisions.py \
  --model-path /path/to/Qwen3-4B --prefix-len 2048 --branches 32 --flashinfer
```

The reported FlashInfer runs used `flashinfer-python==0.6.13`, PyTorch
`2.11.0+cu128`, and CUDA 12.8 to compile the required kernels on the A40.

## Results on one NVIDIA A40

The raw JSON files in [`reports/`](reports/) include shapes, timing scope, and
probability checks. Measurements were made on `linke-iot-node1`, GPU 0.

### Qwen3-4B attention kernel only

| Shared prefix | Branches | Independent | Grouped | Grouped speedup |
| ---: | ---: | ---: | ---: | ---: |
| 256 tokens | 1 | 10.8 µs | 8.2 µs | 1.3× |
| 256 tokens | 16 | 42.4 µs | 9.0 µs | 4.7× |
| 2,048 tokens | 32 | 508.6 µs | 40.8 µs | 12.5× |
| 8,192 tokens | 64 | 3,967.3 µs | 318.7 µs | 12.4× |

These are CUDA graph kernel-path timings on model-derived first-layer Q/K/V,
not full inference latency. The largest absolute attention-output error against
the float32 reference was below `0.001` in the reported runs.

### Complete Qwen3-4B decision path, excluding first prefix prefill

| Shared prefix | Branches | Independent | Grouped | Grouped speedup |
| ---: | ---: | ---: | ---: | ---: |
| 256 tokens | 16 | 243.5 ms | 245.6 ms | 0.99× |
| 2,048 tokens | 32 | 318.2 ms | 250.3 ms | 1.27× |

The shorter prefix is dominated by model projections, MLPs, and token-by-token
branch execution. A faster attention kernel does not guarantee a faster
complete request. With 2,048 prefix tokens, the grouped path changes the first
branch's `Yes`/`No` probability by at most `0.0039` versus Transformers in this
test; the maximum difference between grouped and independent modes over all
branches was `0.0078`.

### OpenJev 4B v5 versus this Qwen3-4B path

| State / binary questions | Qwen first request | Qwen cached state | OpenJev shared prefix |
| --- | ---: | ---: | ---: |
| ~250 tokens / 8 | 405.1 ms | 379.3 ms | 339.0 ms |
| ~2,000 tokens / 16 | 618.4 ms | 407.2 ms | 933.3 ms |

Each entry is the median of five runs after warmup; individual times are in
the JSON reports. The Qwen first request includes prefill; its cached-state
column reuses that prefix. OpenJev's method prefills the state in each call,
then branches across two hypotheses per question. Token counts differ slightly
because each model uses its own tokenizer and prompt template. OpenJev is
faster for the short state; this Qwen path is faster for the long state. The
models agreed on the binary direction of 5/8 and 12/16 questions respectively;
no ground-truth accuracy claim is made from those counts.

### FlashInfer shared-prefix cascade versus the Triton kernel

These pairs were measured in the same benchmark process on the same A40.
Attention timings use CUDA graphs with plans and KV page writes excluded.
Complete Qwen3 timings include suffix page writes and every model layer, but
exclude the first prefix prefill and one-time FlashInfer planning and prefix
page packing.

| Shared prefix / branches | Triton attention | FlashInfer attention | Triton full decision | FlashInfer full decision |
| --- | ---: | ---: | ---: | ---: |
| 256 / 16 | 9.06 µs | 22.07 µs | 213.10 ms | 234.59 ms |
| 2,048 / 32 | 40.87 µs | 51.65 µs | 211.99 ms | 233.18 ms |

The FlashInfer smoke check covered a 251-token prefix and suffix lengths of
1, 6, 17, and 20, with maximum absolute attention-output error `0.00196`
against the float32 reference. The largest probability difference between
FlashInfer and Triton across the complete-decision runs was `0.01161`.

Using the FlashInfer backend in the earlier OpenJev workload gave:

| State / binary questions | Qwen + FlashInfer first request | Qwen + FlashInfer cached state | OpenJev shared prefix |
| --- | ---: | ---: | ---: |
| ~250 tokens / 8 | 439.0 ms | 401.7 ms | 337.3 ms |
| ~2,000 tokens / 16 | 680.3 ms | 442.4 ms | 932.6 ms |

Each entry is the median of five runs after warmup. These are separate runs
from the original Triton versus OpenJev table above, so the direct backend
comparison is the same-process table. Raw samples and timings are in
[`reports/`](reports/).

## Limits

The Triton kernel handles the final query of each branch with a common prefix
and a bounded individual suffix; it is not a general attention backend. The
Qwen3 integration currently processes suffix tokens sequentially and does not
serve an HTTP API. The OpenJev and Qwen paths use their own intended readouts,
so matching latency does not imply matching decision quality or calibration.
The Qwen3-4B checkpoint was already present on the server; its upstream
revision was not recorded. Reproducing the exact Qwen numbers requires the
same weights and software environment.
The FlashInfer backend uses an optional external dependency and caches its
plan for each suffix position and prefix pages for each state. Its measured
performance is specific to FlashInfer 0.6.13, this A40, and these shapes.
