"""A single-query-per-branch attention kernel that reuses shared prefix KV.

All branches have the same prefix, and each query attends to its own suffix.
The grouped kernel places several branch queries in one Triton program so a
prefix K/V tile is loaded once and used by all of them. This is an inference
prototype: it handles the final query of each branch, not general prefill.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    q_ptr, pk_ptr, pv_ptr, sk_ptr, sv_ptr, sl_ptr, out_ptr,
    q_stride_b: tl.constexpr, q_stride_h: tl.constexpr,
    p_stride_l: tl.constexpr, p_stride_h: tl.constexpr,
    s_stride_b: tl.constexpr, s_stride_l: tl.constexpr, s_stride_h: tl.constexpr,
    o_stride_b: tl.constexpr, o_stride_h: tl.constexpr,
    n_branch: tl.constexpr, n_q_head: tl.constexpr, n_kv_head: tl.constexpr,
    prefix_len: tl.constexpr, suffix_cap: tl.constexpr,
    head_dim: tl.constexpr, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    branches = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    head = tl.program_id(1)
    kv_head = head // (n_q_head // n_kv_head)
    dims = tl.arange(0, BLOCK_D)
    cols = tl.arange(0, BLOCK_N)

    q = tl.load(
        q_ptr + branches[:, None] * q_stride_b + head * q_stride_h + dims[None, :],
        mask=(branches[:, None] < n_branch) & (dims[None, :] < head_dim),
        other=0,
    )
    m = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    numerator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start in range(tl.cdiv(prefix_len, BLOCK_N)):
        positions = start * BLOCK_N + cols
        k = tl.load(
            pk_ptr + positions[:, None] * p_stride_l + kv_head * p_stride_h + dims[None, :],
            mask=(positions[:, None] < prefix_len) & (dims[None, :] < head_dim),
            other=0,
        )
        v = tl.load(
            pv_ptr + positions[:, None] * p_stride_l + kv_head * p_stride_h + dims[None, :],
            mask=(positions[:, None] < prefix_len) & (dims[None, :] < head_dim),
            other=0,
        )
        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(positions[None, :] < prefix_len, scores, float("-inf"))
        tile_max = tl.maximum(m, tl.max(scores, 1))
        old_weight = tl.exp(m - tile_max)
        weights = tl.exp(scores - tile_max[:, None])
        denominator = denominator * old_weight + tl.sum(weights, 1)
        numerator = numerator * old_weight[:, None] + tl.dot(weights.to(v.dtype), v)
        m = tile_max

    if suffix_cap > 0:
        suffix_pos = tl.arange(0, BLOCK_S)
        lengths = tl.load(sl_ptr + branches, mask=branches < n_branch, other=0)
        sk = tl.load(
            sk_ptr + branches[:, None, None] * s_stride_b
            + suffix_pos[None, :, None] * s_stride_l
            + kv_head * s_stride_h + dims[None, None, :],
            mask=(branches[:, None, None] < n_branch)
            & (suffix_pos[None, :, None] < suffix_cap)
            & (dims[None, None, :] < head_dim),
            other=0,
        )
        sv = tl.load(
            sv_ptr + branches[:, None, None] * s_stride_b
            + suffix_pos[None, :, None] * s_stride_l
            + kv_head * s_stride_h + dims[None, None, :],
            mask=(branches[:, None, None] < n_branch)
            & (suffix_pos[None, :, None] < suffix_cap)
            & (dims[None, None, :] < head_dim),
            other=0,
        )
        suffix_scores = tl.sum(q[:, None, :].to(tl.float32) * sk.to(tl.float32), 2) * scale
        suffix_scores = tl.where(suffix_pos[None, :] < lengths[:, None], suffix_scores, float("-inf"))
        suffix_max = tl.maximum(m, tl.max(suffix_scores, 1))
        old_weight = tl.exp(m - suffix_max)
        weights = tl.exp(suffix_scores - suffix_max[:, None])
        denominator = denominator * old_weight + tl.sum(weights, 1)
        numerator = numerator * old_weight[:, None] + tl.sum(weights[:, :, None] * sv.to(tl.float32), 1)

    output = numerator / denominator[:, None]
    tl.store(
        out_ptr + branches[:, None] * o_stride_b + head * o_stride_h + dims[None, :],
        output,
        mask=(branches[:, None] < n_branch) & (dims[None, :] < head_dim),
    )


@triton.jit
def _independent_attention_kernel(
    q_ptr, pk_ptr, pv_ptr, sk_ptr, sv_ptr, sl_ptr, out_ptr,
    q_stride_b: tl.constexpr, q_stride_h: tl.constexpr,
    p_stride_l: tl.constexpr, p_stride_h: tl.constexpr,
    s_stride_b: tl.constexpr, s_stride_l: tl.constexpr, s_stride_h: tl.constexpr,
    o_stride_b: tl.constexpr, o_stride_h: tl.constexpr,
    n_q_head: tl.constexpr, n_kv_head: tl.constexpr,
    prefix_len: tl.constexpr, suffix_cap: tl.constexpr,
    head_dim: tl.constexpr, scale: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
):
    branch = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // (n_q_head // n_kv_head)
    dims = tl.arange(0, BLOCK_D)
    cols = tl.arange(0, BLOCK_N)
    q = tl.load(q_ptr + branch * q_stride_b + head * q_stride_h + dims,
                mask=dims < head_dim, other=0).to(tl.float32)
    m = float("-inf")
    denominator = 0.0
    numerator = tl.zeros((BLOCK_D,), tl.float32)

    for start in range(tl.cdiv(prefix_len, BLOCK_N)):
        positions = start * BLOCK_N + cols
        k = tl.load(
            pk_ptr + positions[:, None] * p_stride_l + kv_head * p_stride_h + dims[None, :],
            mask=(positions[:, None] < prefix_len) & (dims[None, :] < head_dim), other=0,
        ).to(tl.float32)
        v = tl.load(
            pv_ptr + positions[:, None] * p_stride_l + kv_head * p_stride_h + dims[None, :],
            mask=(positions[:, None] < prefix_len) & (dims[None, :] < head_dim), other=0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], 1) * scale
        scores = tl.where(positions < prefix_len, scores, float("-inf"))
        next_m = tl.maximum(m, tl.max(scores, 0))
        old_weight = tl.exp(m - next_m)
        weights = tl.exp(scores - next_m)
        denominator = denominator * old_weight + tl.sum(weights, 0)
        numerator = numerator * old_weight + tl.sum(weights[:, None] * v, 0)
        m = next_m

    if suffix_cap > 0:
        suffix_pos = tl.arange(0, BLOCK_S)
        length = tl.load(sl_ptr + branch)
        sk = tl.load(
            sk_ptr + branch * s_stride_b + suffix_pos[:, None] * s_stride_l
            + kv_head * s_stride_h + dims[None, :],
            mask=(suffix_pos[:, None] < suffix_cap) & (dims[None, :] < head_dim), other=0,
        ).to(tl.float32)
        sv = tl.load(
            sv_ptr + branch * s_stride_b + suffix_pos[:, None] * s_stride_l
            + kv_head * s_stride_h + dims[None, :],
            mask=(suffix_pos[:, None] < suffix_cap) & (dims[None, :] < head_dim), other=0,
        ).to(tl.float32)
        scores = tl.sum(sk * q[None, :], 1) * scale
        scores = tl.where(suffix_pos < length, scores, float("-inf"))
        next_m = tl.maximum(m, tl.max(scores, 0))
        old_weight = tl.exp(m - next_m)
        weights = tl.exp(scores - next_m)
        denominator = denominator * old_weight + tl.sum(weights, 0)
        numerator = numerator * old_weight + tl.sum(weights[:, None] * sv, 0)

    tl.store(out_ptr + branch * o_stride_b + head * o_stride_h + dims,
             numerator / denominator, mask=dims < head_dim)


def shared_prefix_attention(
    queries: torch.Tensor,
    prefix_keys: torch.Tensor,
    prefix_values: torch.Tensor,
    suffix_keys: torch.Tensor | None = None,
    suffix_values: torch.Tensor | None = None,
    suffix_lengths: torch.Tensor | None = None,
    *,
    branches_per_program: int = 16,
) -> torch.Tensor:
    """Compute attention for the final token of independent prefix-sharing branches.

    Shapes: queries [B,Hq,D], prefix K/V [L,Hkv,D], suffix K/V
    [B,S,Hkv,D]. Every query can see the entire prefix and only its own
    first ``suffix_lengths[b]`` suffix tokens. Use ``branches_per_program=1``
    for the independent-query baseline; 16 shares prefix loads across queries.
    """
    if queries.ndim != 3 or prefix_keys.ndim != 3 or prefix_values.shape != prefix_keys.shape:
        raise ValueError("Expected Q [B,Hq,D] and matching prefix K/V [L,Hkv,D]")
    batch, q_heads, head_dim = queries.shape
    prefix_len, kv_heads, kv_dim = prefix_keys.shape
    if batch < 1 or prefix_len < 1 or kv_dim != head_dim or q_heads % kv_heads:
        raise ValueError("Invalid batch, prefix length, head dimension, or GQA head count")
    if branches_per_program not in (1, 16):
        raise ValueError("branches_per_program must be 1 or 16")
    if not queries.is_cuda or any(x.device != queries.device for x in (prefix_keys, prefix_values)):
        raise ValueError("All tensors must be on the same CUDA device")
    if queries.dtype not in (torch.float16, torch.bfloat16) or any(
        x.dtype != queries.dtype for x in (prefix_keys, prefix_values)
    ):
        raise ValueError("Q and K/V must share float16 or bfloat16 dtype")
    if suffix_keys is None:
        if suffix_values is not None or suffix_lengths is not None:
            raise ValueError("Supply suffix keys, values, and lengths together")
        suffix_cap = 0
        suffix_keys = queries
        suffix_values = queries
        suffix_lengths = torch.empty(0, dtype=torch.int32, device=queries.device)
    else:
        if suffix_values is None or suffix_lengths is None or suffix_keys.ndim != 4:
            raise ValueError("Supply suffix keys, values, and lengths together")
        if suffix_values.shape != suffix_keys.shape or suffix_keys.shape[:1] != (batch,):
            raise ValueError("Suffix K/V must have matching batch and shape")
        suffix_cap = suffix_keys.shape[1]
        if suffix_keys.shape[2:] != (kv_heads, head_dim) or suffix_lengths.shape != (batch,):
            raise ValueError("Suffix head shape or lengths are invalid")
        if any(x.device != queries.device for x in (suffix_keys, suffix_values, suffix_lengths)):
            raise ValueError("Suffix tensors must be on the Q device")
        if suffix_keys.dtype != queries.dtype or suffix_values.dtype != queries.dtype:
            raise ValueError("Suffix K/V dtype must match Q")
        if suffix_lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("Suffix lengths must be integer")
        # Checking CUDA values here would synchronize every inference call.
        # Callers must supply lengths in [0, suffix_cap].

    queries = queries.contiguous()
    prefix_keys = prefix_keys.contiguous()
    prefix_values = prefix_values.contiguous()
    suffix_keys = suffix_keys.contiguous()
    suffix_values = suffix_values.contiguous()
    output = torch.empty_like(queries)
    kernel = _independent_attention_kernel if branches_per_program == 1 else _attention_kernel
    common_args = (
        queries, prefix_keys, prefix_values, suffix_keys, suffix_values, suffix_lengths, output,
        *queries.stride()[:2], *prefix_keys.stride()[:2],
        *suffix_keys.stride()[:3], *output.stride()[:2],
    )
    if branches_per_program == 1:
        kernel[(batch, q_heads)](
            *common_args, q_heads, kv_heads, prefix_len, suffix_cap, head_dim,
            1.0 / math.sqrt(head_dim), 64, triton.next_power_of_2(head_dim),
            triton.next_power_of_2(max(1, suffix_cap)), num_warps=4,
        )
    else:
        kernel[(triton.cdiv(batch, branches_per_program), q_heads)](
            *common_args, batch, q_heads, kv_heads, prefix_len, suffix_cap, head_dim,
            1.0 / math.sqrt(head_dim), branches_per_program, 64,
            triton.next_power_of_2(head_dim),
            triton.next_power_of_2(max(1, suffix_cap)), num_warps=4,
        )
    return output
