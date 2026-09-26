"""GPU correctness check for FlashInfer cascade page mapping and masking."""

import json

import torch

from llm2jev.flashinfer_attention import FlashInferSharedPrefixAttention
from llm2jev.reference import reference_attention


def main():
    torch.manual_seed(7)
    batch, q_heads, kv_heads, head_dim = 4, 8, 2, 128
    prefix_len, suffix_len = 251, 20
    shape = (batch, q_heads, head_dim)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    pk = torch.randn((prefix_len, kv_heads, head_dim), dtype=q.dtype, device=q.device)
    pv = torch.randn_like(pk)
    sk = torch.randn((batch, suffix_len, kv_heads, head_dim), dtype=q.dtype, device=q.device)
    sv = torch.randn_like(sk)
    lengths = (1, 6, 17, 20)
    runner = FlashInferSharedPrefixAttention(
        batch, q_heads, kv_heads, head_dim, prefix_len, suffix_len, q.dtype, q.device,
    )
    cache = runner.prepare_layer_caches([pk], [pv], generation=1)[0]
    for position in range(suffix_len):
        runner.append(cache, position, sk[:, position], sv[:, position])
    actual = runner.forward(q, cache, lengths)
    expected = reference_attention(
        q, pk, pv, sk, sv,
        torch.tensor(lengths, dtype=torch.int32, device=q.device),
    )
    error = (actual.float() - expected.float()).abs().max().item()
    newer_pk = torch.randn_like(pk)
    newer_pv = torch.randn_like(pv)
    refreshed_cache = runner.prepare_layer_caches(
        [newer_pk], [newer_pv], generation=2,
    )[0]
    assert refreshed_cache is cache
    refreshed = runner.forward(q, refreshed_cache, lengths)
    refreshed_expected = reference_attention(
        q, newer_pk, newer_pv, sk, sv,
        torch.tensor(lengths, dtype=torch.int32, device=q.device),
    )
    refresh_error = (refreshed.float() - refreshed_expected.float()).abs().max().item()
    print(json.dumps({"max_abs_error": error, "refresh_max_abs_error": refresh_error,
                      "lengths": lengths}))
    assert error < 0.02, error
    assert refresh_error < 0.02, refresh_error


if __name__ == "__main__":
    main()
