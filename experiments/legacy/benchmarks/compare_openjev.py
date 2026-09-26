"""Matched-state latency comparison with OpenJev 4B v5 on one GPU.

Both arms answer the same binary questions about one state. OpenJev uses the
author's shared-prefix predict_hypotheses method, with two hypotheses per
question. Qwen3 uses this project's shared-prefix branch scorer. Their prompt
templates and readout heads differ, so latency is a systems comparison, not a
controlled model-architecture ablation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm2jev.qwen3 import Qwen3BranchEngine


QUESTIONS = [
    "Should this request use web search?",
    "Should this request use a coding tool?",
    "Does this request need a human reviewer?",
    "Can this request wait in a queue?",
    "Is the GPU load currently high?",
    "Is the request about recent information?",
    "Should the answer include a source link?",
    "Is the task asking for a calculation?",
    "Should the system retry after a timeout?",
    "Does the request need image understanding?",
    "Is the user's question about software?",
    "Should a long response be avoided?",
    "Is the deadline stated as urgent?",
    "Can the answer be a yes or no?",
    "Does the request mention a document?",
    "Is the task complete already?",
]


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000, result


def median(values):
    values = sorted(values)
    return round(values[len(values) // 2], 3)


def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, local_files_only=True)
    sentence = (
        "The service is processing user requests. Its GPU queue has pending work, "
        "some requests need current information, and others need code or human review. "
    )
    seed = tokenizer.encode(sentence, add_special_tokens=False)
    prefix_ids = (seed * ((args.prefix_len // len(seed)) + 1))[:args.prefix_len]
    state = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    questions = [QUESTIONS[i % len(QUESTIONS)] for i in range(args.branches)]
    suffix_texts = ["\n" + q + " Answer Yes or No:" for q in questions]

    def qwen_inputs():
        # Match OpenJev's public helper, which tokenizes on every call.
        encoded_prefix = tokenizer.encode(state, add_special_tokens=False)
        encoded_suffixes = [tokenizer.encode(s, add_special_tokens=False) for s in suffix_texts]
        lengths = torch.tensor([len(row) for row in encoded_suffixes], dtype=torch.int32, device="cuda")
        width = max(map(len, encoded_suffixes))
        ids = torch.full((args.branches, width), tokenizer.pad_token_id or 0,
                         dtype=torch.long, device="cuda")
        for row, tokens in enumerate(encoded_suffixes):
            ids[row, :len(tokens)] = torch.tensor(tokens, device="cuda")
        return torch.tensor(encoded_prefix, device="cuda")[None], ids, lengths

    prefix_gpu, suffix_ids, suffix_lengths = qwen_inputs()
    suffix_sizes = suffix_lengths.cpu().tolist()
    candidates = [tokenizer.encode(s, add_special_tokens=False) for s in ("Yes", "No")]
    if any(len(row) != 1 for row in candidates):
        raise ValueError("Qwen tokenizer must encode Yes and No as single tokens")
    candidate_ids = torch.tensor([row[0] for row in candidates], device="cuda")

    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    engine = Qwen3BranchEngine(model)

    def qwen_cold():
        prefix_gpu, suffix_ids, suffix_lengths = qwen_inputs()
        engine.prepare_prefix(prefix_gpu)
        return engine.score_suffixes(suffix_ids, candidate_ids, suffix_lengths,
                                     branches_per_program=16, backend=args.qwen_backend)

    def qwen_warm():
        _, suffix_ids, suffix_lengths = qwen_inputs()
        return engine.score_suffixes(suffix_ids, candidate_ids, suffix_lengths,
                                     branches_per_program=16, backend=args.qwen_backend)

    qwen_cold()  # compile and warm up
    qwen_cold_times = []
    for _ in range(args.repeats):
        ms, qwen_probs = timed(qwen_cold)
        qwen_cold_times.append(ms)
    qwen_warm_times = []
    for _ in range(args.repeats):
        ms, qwen_probs = timed(qwen_warm)
        qwen_warm_times.append(ms)
    qwen_yes = qwen_probs[:, 0].float().cpu().tolist()
    del engine, model
    torch.cuda.empty_cache()

    # Load only the downloaded official code and checkpoint; no model code is
    # copied into this project's source tree.
    sys.path.insert(0, str(args.openjev_root))
    from modeling_openjev import ENT, OpenJevCrossEncoder

    ce = OpenJevCrossEncoder(str(args.openjev_model), device="cuda", dtype=torch.bfloat16,
                            bs=max(32, args.branches * 2), max_len=4096)
    hypotheses = []
    for question in questions:
        for label in ("yes", "no"):
            hypotheses.append(f'The answer to "{question}" is {label}: {label}')
    openjev_tokens = len(ce.tok(ce.template.format(premise=state, hypothesis=hypotheses[0]))["input_ids"])

    def openjev_call():
        entailment = ce.predict_hypotheses(state, hypotheses)[:, ENT].reshape(args.branches, 2)
        return entailment[:, 0] / entailment.sum(axis=1)

    openjev_call()  # compile and warm up
    openjev_times = []
    for _ in range(args.repeats):
        ms, openjev_yes = timed(openjev_call)
        openjev_times.append(ms)
    agreement = sum((a >= .5) == (float(b) >= .5) for a, b in zip(qwen_yes, openjev_yes))
    result = {
        "gpu": torch.cuda.get_device_name(),
        "qwen_model": "Qwen3-4B (existing local checkpoint; upstream revision unverified)",
        "qwen_backend": args.qwen_backend,
        "openjev_model": "AlexWortega/openjev/qwen3.5-4b-nli-v5",
        "openjev_revision": "058a6c24911b46d908fbe23541390f8af3df3e4d",
        "questions": args.branches,
        "state_tokens_qwen": prefix_gpu.shape[1],
        "input_tokens_openjev_first_hypothesis": openjev_tokens,
        "qwen_suffix_tokens_min_max": [min(suffix_sizes), max(suffix_sizes)],
        "qwen_cold_ms": median(qwen_cold_times),
        "qwen_warm_ms": median(qwen_warm_times),
        "openjev_shared_prefix_ms": median(openjev_times),
        "latency_samples_ms": {
            "qwen_cold": [round(x, 3) for x in qwen_cold_times],
            "qwen_warm": [round(x, 3) for x in qwen_warm_times],
            "openjev_shared_prefix": [round(x, 3) for x in openjev_times],
        },
        "openjev_to_qwen_cold_ratio": round(median(openjev_times) / median(qwen_cold_times), 3),
        "openjev_to_qwen_warm_ratio": round(median(openjev_times) / median(qwen_warm_times), 3),
        "answer_agreement": f"{agreement}/{args.branches}",
        "qwen_yes_probabilities": qwen_yes,
        "openjev_yes_probabilities": [float(x) for x in openjev_yes],
        "scope": "One state and all binary questions; both arms share the state prefix within each request. "
                 "Qwen cold includes prefill, Qwen warm reuses a prior prefill. OpenJev call includes prefill.",
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--openjev-root", type=Path, required=True)
    parser.add_argument("--openjev-model", type=Path, required=True)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--qwen-backend", choices=("triton", "flashinfer"),
                        default="triton")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
