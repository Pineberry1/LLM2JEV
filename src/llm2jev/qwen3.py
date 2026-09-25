"""Experimental Qwen3 binary scoring with one shared prefix and many branches.

The common prefix is prefetched once with Transformers. Each branch suffix is
processed token by token. At every layer, the branch queries share that layer's
prefix K/V while retaining their own suffix K/V. Only selected vocabulary rows
are evaluated after the final token.
"""

import torch
import torch.nn.functional as F

from .attention import shared_prefix_attention


class Qwen3BranchEngine:
    def __init__(self, model):
        if model.config.model_type != "qwen3":
            raise ValueError("Only Qwen3 models are supported")
        if model.training:
            raise ValueError("Call model.eval() before creating the engine")
        self.model = model
        self.prefix_keys = None
        self.prefix_values = None
        self.prefix_len = 0

    @torch.inference_mode()
    def prepare_prefix(self, prefix_ids: torch.Tensor) -> None:
        """Prefill one shared prefix; this work is excluded from branch timing."""
        if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1 or prefix_ids.shape[1] < 1:
            raise ValueError("prefix_ids must have shape [1, L] with L > 0")
        cache = self.model.model(input_ids=prefix_ids, use_cache=True).past_key_values
        self.prefix_keys = [layer.keys[0].transpose(0, 1).contiguous() for layer in cache.layers]
        self.prefix_values = [layer.values[0].transpose(0, 1).contiguous() for layer in cache.layers]
        self.prefix_len = prefix_ids.shape[1]

    @torch.inference_mode()
    def score_suffixes(
        self,
        suffix_ids: torch.Tensor,
        candidate_token_ids: torch.Tensor,
        suffix_lengths: torch.Tensor | None = None,
        *,
        branches_per_program: int = 16,
    ) -> torch.Tensor:
        """Return normalized candidate probabilities with shape [B, C]."""
        if self.prefix_keys is None:
            raise ValueError("Call prepare_prefix first")
        if suffix_ids.ndim != 2 or suffix_ids.shape[0] < 1 or suffix_ids.shape[1] < 1:
            raise ValueError("suffix_ids must have shape [B, S] with B,S > 0")
        if candidate_token_ids.ndim != 1 or candidate_token_ids.numel() < 2:
            raise ValueError("Provide at least two candidate token IDs")
        if suffix_lengths is not None and suffix_lengths.shape != (suffix_ids.shape[0],):
            raise ValueError("suffix_lengths must have one entry per branch")
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        model = self.model
        batch, suffix_len = suffix_ids.shape
        suffix_keys = [None] * len(model.model.layers)
        suffix_values = [None] * len(model.model.layers)
        hidden = None
        final_hidden = None

        for position in range(suffix_len):
            hidden = model.model.embed_tokens(suffix_ids[:, position:position + 1])
            position_ids = torch.full((batch, 1), self.prefix_len + position,
                                      device=suffix_ids.device, dtype=torch.long)
            cos, sin = model.model.rotary_emb(hidden, position_ids)
            lengths = torch.full((batch,), position + 1, device=suffix_ids.device,
                                 dtype=torch.int32)

            for index, layer in enumerate(model.model.layers):
                residual = hidden
                normalized = layer.input_layernorm(hidden)
                attn = layer.self_attn
                head_dim = attn.head_dim
                q_heads = attn.config.num_attention_heads
                kv_heads = attn.config.num_key_value_heads
                q = attn.q_norm(attn.q_proj(normalized).view(batch, 1, q_heads, head_dim)).transpose(1, 2)
                k = attn.k_norm(attn.k_proj(normalized).view(batch, 1, kv_heads, head_dim)).transpose(1, 2)
                v = attn.v_proj(normalized).view(batch, 1, kv_heads, head_dim).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                current_k = k.transpose(1, 2)
                current_v = v.transpose(1, 2)
                suffix_keys[index] = (current_k if suffix_keys[index] is None else
                                      torch.cat((suffix_keys[index], current_k), dim=1))
                suffix_values[index] = (current_v if suffix_values[index] is None else
                                        torch.cat((suffix_values[index], current_v), dim=1))
                context = shared_prefix_attention(
                    q[:, :, 0, :], self.prefix_keys[index], self.prefix_values[index],
                    suffix_keys[index], suffix_values[index], lengths,
                    branches_per_program=branches_per_program,
                )
                hidden = residual + attn.o_proj(context.reshape(batch, 1, -1))
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

            if suffix_lengths is not None:
                if final_hidden is None:
                    final_hidden = torch.zeros_like(hidden)
                final_hidden = torch.where(
                    (suffix_lengths == position + 1)[:, None, None], hidden, final_hidden,
                )

        last_hidden = model.model.norm(final_hidden if final_hidden is not None else hidden)[:, 0, :]
        selected_weights = model.lm_head.weight.index_select(0, candidate_token_ids)
        logits = F.linear(last_hidden, selected_weights).float()
        return logits.softmax(-1)
