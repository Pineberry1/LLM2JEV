"""Benchmark the first Qwen3 attention layer on model-derived Q/K/V tensors."""

import argparse
import json
from pathlib import Path

import torch

from llm2jev.attention import shared_prefix_attention
from llm2jev.reference import reference_attention


def model_tensors(model_path: str, prefix_len: int, branches: int, suffix_len: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    layer = model.model.layers[0]
    attn = layer.self_attn
    sentence = (
        "System state: the service has an incoming user request, a queue depth, "
        "a latency target, and a GPU utilization measurement. "
    )
    seed_ids = tokenizer.encode(sentence, add_special_tokens=False)
    prefix_ids = torch.tensor((seed_ids * ((prefix_len // len(seed_ids)) + 1))[:prefix_len], device="cuda")[None]
    choices = [
        "Should this request route to the search service?",
        "Should this request route to the coding service?",
        "Should this request wait for GPU capacity?",
        "Is the request likely to require human review?",
    ]
    suffix_rows = []
    for index in range(branches):
        ids = tokenizer.encode(choices[index % len(choices)], add_special_tokens=False)
        suffix_rows.append((ids * ((suffix_len // len(ids)) + 1))[:suffix_len])
    suffix_ids = torch.tensor(suffix_rows, device="cuda")
    hq, hk, dim = attn.config.num_attention_heads, attn.config.num_key_value_heads, attn.head_dim

    def projections(ids, start):
        hidden = layer.input_layernorm(model.model.embed_tokens(ids))
        batch, length = ids.shape
        q = attn.q_norm(attn.q_proj(hidden).view(batch, length, hq, dim)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden).view(batch, length, hk, dim)).transpose(1, 2)
        v = attn.v_proj(hidden).view(batch, length, hk, dim).transpose(1, 2)
        positions = torch.arange(start, start + length, device="cuda")[None].expand(batch, -1)
        cos, sin = model.model.rotary_emb(hidden, positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    with torch.inference_mode():
        _, pk, pv = projections(prefix_ids, 0)
        q, sk, sv = projections(suffix_ids, prefix_len)
        tensors = (q[:, -1].contiguous(), pk[0].contiguous(), pv[0].contiguous(),
                   sk.contiguous(), sv.contiguous())
    del model, layer, attn
    torch.cuda.empty_cache()
    return tensors


def measure(fn, repeats: int = 500) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, repeats // 10)):
        graph.replay()
    end.record()
    end.synchronize()
    return 1000 * start.elapsed_time(end) / (10 * max(1, repeats // 10))


def run(model_path: str, prefix_len: int, branches: int, suffix_len: int, repeats: int):
    q, pk, pv, sk, sv = model_tensors(model_path, prefix_len, branches, suffix_len)
    lengths = torch.full((branches,), suffix_len, dtype=torch.int32, device="cuda")
    expected = reference_attention(q, pk, pv, sk, sv, lengths)
    results = {}
    for group in (1, 16):
        fn = lambda: shared_prefix_attention(
            q, pk, pv, sk, sv, lengths, branches_per_program=group,
        )
        actual = fn()
        error = (actual.float() - expected.float()).abs().max().item()
        results[str(group)] = {"latency_us": round(measure(fn, repeats), 3), "max_abs_error": round(error, 6)}
    results["speedup"] = round(results["1"]["latency_us"] / results["16"]["latency_us"], 3)
    return {
        "model": model_path, "gpu": torch.cuda.get_device_name(),
        "prefix_tokens": prefix_len, "branches": branches, "suffix_tokens": suffix_len,
        "query_heads": q.shape[1], "kv_heads": pk.shape[1], "head_dim": q.shape[2],
        "dtype": str(q.dtype), "timing": "CUDA graph, kernel path, microseconds per call",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument("--branches", type=int, default=32)
    parser.add_argument("--suffix-len", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.model_path, args.prefix_len, args.branches, args.suffix_len, args.repeats)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
