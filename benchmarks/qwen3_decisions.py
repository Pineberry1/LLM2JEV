"""Measure complete Qwen3 branch scoring after a single prefix prefill."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm2jev.qwen3 import Qwen3BranchEngine


def measure(fn, repeats):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    return sorted(elapsed)[len(elapsed) // 2], result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--branches", type=int, default=16)
    parser.add_argument("--suffix-len", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    state = "The request involves a technical question and the GPU queue is not full. "
    state_ids = tokenizer.encode(state, add_special_tokens=False)
    prefix = torch.tensor((state_ids * ((args.prefix_len // len(state_ids)) + 1))[:args.prefix_len],
                          device="cuda")[None]
    questions = ["Should this request use coding?", "Should this request use search?",
                 "Should the request wait?", "Is human review needed?"]
    suffix_rows = []
    for i in range(args.branches):
        ids = tokenizer.encode(questions[i % len(questions)], add_special_tokens=False)
        suffix_rows.append((ids * ((args.suffix_len // len(ids)) + 1))[:args.suffix_len])
    suffixes = torch.tensor(suffix_rows, device="cuda")
    labels = ("Yes", "No")
    label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(ids) != 1 for ids in label_ids):
        raise ValueError("Yes and No must each tokenize to one token for this experiment")
    candidate_ids = torch.tensor([ids[0] for ids in label_ids], device="cuda")

    engine = Qwen3BranchEngine(model)
    with torch.inference_mode():
        engine.prepare_prefix(prefix)
        timings = {}
        probabilities = {}
        for group in (1, 16):
            latency, probs = measure(
                lambda: engine.score_suffixes(suffixes, candidate_ids,
                                              branches_per_program=group), args.repeats,
            )
            timings[str(group)] = round(latency, 3)
            probabilities[str(group)] = probs
        full_ids = torch.cat((prefix, suffixes[0:1]), dim=1)
        reference_logits = model(input_ids=full_ids, use_cache=False).logits[0, -1]
        reference_probs = reference_logits.index_select(0, candidate_ids).float().softmax(-1)

    result = {
        "model": args.model_path, "gpu": torch.cuda.get_device_name(),
        "prefix_tokens": args.prefix_len, "branches": args.branches,
        "suffix_tokens": args.suffix_len, "candidate_tokens": list(labels),
        "prefix_prefill_excluded": True,
        "latency_ms": timings,
        "speedup": round(timings["1"] / timings["16"], 3),
        "max_probability_delta_between_modes": round(
            (probabilities["1"] - probabilities["16"]).abs().max().item(), 6,
        ),
        "reference_first_branch_probability": reference_probs.tolist(),
        "grouped_first_branch_probability": probabilities["16"][0].tolist(),
        "max_first_branch_delta_vs_transformers": round(
            (probabilities["16"][0] - reference_probs).abs().max().item(), 6,
        ),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
