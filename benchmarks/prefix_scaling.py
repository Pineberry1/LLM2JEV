"""Compare prefix lengths with identical branch work in one model process."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm2jev.qwen3 import Qwen3BranchEngine


def median_ms(values):
    return round(statistics.median(values), 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prefix-lengths", nargs="+", type=int, default=[256, 2048])
    parser.add_argument("--branches", type=int, default=16)
    parser.add_argument("--suffix-len", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--flashinfer", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    sentence = "The request involves a technical question and the GPU queue is not full. "
    state_ids = tokenizer.encode(sentence, add_special_tokens=False)
    prefixes = {
        length: torch.tensor(
            (state_ids * ((length // len(state_ids)) + 1))[:length], device="cuda"
        )[None] for length in args.prefix_lengths
    }
    questions = ["Should this request use coding?", "Should this request use search?",
                 "Should the request wait?", "Is human review needed?"]
    suffix_rows = []
    for branch in range(args.branches):
        ids = tokenizer.encode(questions[branch % len(questions)], add_special_tokens=False)
        suffix_rows.append((ids * ((args.suffix_len // len(ids)) + 1))[:args.suffix_len])
    suffixes = torch.tensor(suffix_rows, device="cuda")
    candidate_ids = torch.tensor(
        [tokenizer.encode(label, add_special_tokens=False)[0] for label in ("Yes", "No")],
        device="cuda",
    )
    engine = Qwen3BranchEngine(model)
    backends = ["triton", "flashinfer"] if args.flashinfer else ["triton"]
    samples = {
        backend: {str(length): {"prefill": [], "branch": [], "total": []}
                  for length in args.prefix_lengths}
        for backend in backends
    }

    with torch.inference_mode():
        for repeat in range(args.repeats + 2):
            lengths = args.prefix_lengths if repeat % 2 == 0 else list(reversed(args.prefix_lengths))
            modes = backends if repeat % 2 == 0 else list(reversed(backends))
            for length in lengths:
                for backend in modes:
                    start = torch.cuda.Event(enable_timing=True)
                    after_prefill = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    engine.prepare_prefix(prefixes[length])
                    after_prefill.record()
                    engine.score_suffixes(
                        suffixes, candidate_ids, branches_per_program=16,
                        backend=backend,
                    )
                    end.record()
                    end.synchronize()
                    if repeat >= 2:
                        item = samples[backend][str(length)]
                        item["prefill"].append(round(start.elapsed_time(after_prefill), 3))
                        item["branch"].append(round(after_prefill.elapsed_time(end), 3))
                        item["total"].append(round(start.elapsed_time(end), 3))

    result = {
        "model": "Qwen3-4B (existing local checkpoint; upstream revision unverified)",
        "gpu": torch.cuda.get_device_name(),
        "prefix_lengths": args.prefix_lengths,
        "branches": args.branches,
        "suffix_tokens": args.suffix_len,
        "repeats": args.repeats,
        "timing": "CUDA events; same process; two warmups; length/backend order alternated",
        "samples_ms": samples,
        "median_ms": {
            backend: {
                length: {scope: median_ms(values) for scope, values in scopes.items()}
                for length, scopes in configurations.items()
            } for backend, configurations in samples.items()
        },
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
