"""Clear, slow attention reference for numerical checks."""

import torch


def reference_attention(q, pk, pv, sk=None, sv=None, lengths=None):
    batch, q_heads, dim = q.shape
    repeats = q_heads // pk.shape[1]
    pk = pk.repeat_interleave(repeats, dim=1).float()
    pv = pv.repeat_interleave(repeats, dim=1).float()
    scale = dim ** -0.5
    prefix_scores = torch.einsum("bhd,lhd->bhl", q.float(), pk) * scale
    if sk is None:
        return torch.einsum("bhl,lhd->bhd", prefix_scores.softmax(-1), pv).to(q.dtype)
    sk = sk.repeat_interleave(repeats, dim=2).float()
    sv = sv.repeat_interleave(repeats, dim=2).float()
    suffix_scores = torch.einsum("bhd,bshd->bhs", q.float(), sk) * scale
    suffix_scores = suffix_scores.masked_fill(
        torch.arange(sk.shape[1], device=q.device)[None, None, :] >= lengths[:, None, None],
        float("-inf"),
    )
    probs = torch.cat((prefix_scores, suffix_scores), dim=-1).softmax(-1)
    result = torch.einsum("bhl,lhd->bhd", probs[..., :pk.shape[0]], pv)
    result += torch.einsum("bhs,bshd->bhd", probs[..., pk.shape[0]:], sv)
    return result.to(q.dtype)
