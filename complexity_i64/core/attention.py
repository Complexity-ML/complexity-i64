"""
Complexity-I64 :: Integer Attention

QKV projections: INT8 matmuls (fused QKV + fused mu_QKV)
O projection: INT8 matmul
Attention core: float (softmax + dot products — irreducible)
RoPE: float (cos/sin rotation — irreducible)

INL - 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from complexity_i64.core.integer_ops import int8_linear, quantize_weight_int8


class I64Attention(nn.Module):
    """
    Integer-native Mu-Guided Attention.

    INT8: QKV projection, mu projection, O projection (5 matmuls → 3 fused INT8)
    Float: RoPE, QK dot product, softmax, attention × V
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads

        # QKV projections (will be fused + quantized)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        # Mu-guided projections
        self.mu_to_q = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.mu_to_k = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.mu_to_v = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        for proj in [self.mu_to_q, self.mu_to_k, self.mu_to_v]:
            nn.init.normal_(proj.weight, std=0.01)

        # QK Norm
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            from complexity_i64.core.normalization import I64RMSNorm
            self.q_norm = I64RMSNorm(self.head_dim)
            self.k_norm = I64RMSNorm(self.head_dim)

        # RoPE
        self._init_rope(max_position_embeddings, rope_theta)

    def _init_rope(self, max_pos: int, theta: float):
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._max_pos = max_pos

    def _apply_rope(self, q, k, positions):
        """RoPE — float, irreducible. q,k: (batch, heads, seq, head_dim)."""
        # positions: (batch, seq) or (seq,) — use first batch's positions (same for all)
        if positions.dim() == 2:
            pos = positions[0].float()
        else:
            pos = positions.float()
        freqs = torch.outer(pos, self.inv_freq.to(q.device))
        # (seq, head_dim//2) -> (1, 1, seq, head_dim//2) for broadcast
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)

        def rotate(x, cos, sin):
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

        return rotate(q, cos, sin), rotate(k, cos, sin)

    def _project_qkv(self, hidden, mu_prev):
        """QKV + mu projections — INT8 if quantized."""
        if hasattr(self, 'qkv_int8'):
            qkv = int8_linear(hidden, self.qkv_int8, self.qkv_scale)
            q = qkv[:, :self.q_size]
            k = qkv[:, self.q_size:self.q_size + self.kv_size]
            v = qkv[:, self.q_size + self.kv_size:]

            if mu_prev is not None and hasattr(self, 'mu_qkv_int8'):
                mu_qkv = int8_linear(mu_prev, self.mu_qkv_int8, self.mu_qkv_scale)
                q = q + mu_qkv[:, :self.q_size]
                k = k + mu_qkv[:, self.q_size:self.q_size + self.kv_size]
                v = v + mu_qkv[:, self.q_size + self.kv_size:]
        else:
            q = self.q_proj(hidden)
            k = self.k_proj(hidden)
            v = self.v_proj(hidden)
            if mu_prev is not None:
                q = q + self.mu_to_q(mu_prev)
                k = k + self.mu_to_k(mu_prev)
                v = v + self.mu_to_v(mu_prev)

        return q, k, v

    def _o_proj_forward(self, out):
        """O projection — INT8 if quantized."""
        if hasattr(self, 'o_int8'):
            return int8_linear(out, self.o_int8, self.o_scale)
        return self.o_proj(out)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        mu_prev: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len = hidden.shape[0], hidden.shape[1] if hidden.dim() == 3 else 1
        is_2d = hidden.dim() == 2

        if is_2d:
            hidden = hidden.unsqueeze(1)
            if mu_prev is not None:
                mu_prev = mu_prev.unsqueeze(1)
            bsz, seq_len = hidden.shape[0], 1

        q, k, v = self._project_qkv(
            hidden.reshape(-1, self.hidden_size),
            mu_prev.reshape(-1, self.hidden_size) if mu_prev is not None else None,
        )

        # Reshape to heads
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE (float — irreducible)
        q, k = self._apply_rope(q, k, positions)

        # KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        new_kv = (k, v) if use_cache else None

        # GQA expand
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Attention (float — softmax is irreducible)
        # Always use causal masking; attention_mask handles padding on top
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=(attention_mask is None and past_key_value is None),
        )

        # O projection (INT8)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        out = self._o_proj_forward(attn_output.reshape(-1, self.num_heads * self.head_dim))
        out = out.view(bsz, seq_len, self.hidden_size)

        if is_2d:
            out = out.squeeze(1)

        return out, new_kv

    def quantize(self):
        """Fuse and quantize QKV + mu + O to INT8."""
        # Fused QKV
        q_w = self.q_proj.weight.data
        k_w = self.k_proj.weight.data
        v_w = self.v_proj.weight.data
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        qkv_q, qkv_s = quantize_weight_int8(qkv_w)
        self.register_buffer("qkv_int8", qkv_q)
        self.register_buffer("qkv_scale", qkv_s)
        self.q_size = q_w.shape[0]
        self.kv_size = k_w.shape[0]

        # Fused mu_QKV
        mu_q_w = self.mu_to_q.weight.data
        mu_k_w = self.mu_to_k.weight.data
        mu_v_w = self.mu_to_v.weight.data
        mu_qkv_w = torch.cat([mu_q_w, mu_k_w, mu_v_w], dim=0)
        mu_qkv_q, mu_qkv_s = quantize_weight_int8(mu_qkv_w)
        self.register_buffer("mu_qkv_int8", mu_qkv_q)
        self.register_buffer("mu_qkv_scale", mu_qkv_s)

        # O projection
        o_q, o_s = quantize_weight_int8(self.o_proj.weight.data)
        self.register_buffer("o_int8", o_q)
        self.register_buffer("o_scale", o_s)

        # Free float weights
        del self.q_proj
        del self.k_proj
        del self.v_proj
        del self.mu_to_q
        del self.mu_to_k
        del self.mu_to_v
        # Keep o_proj deleted separately (its forward is handled by _o_proj_forward)
        del self.o_proj
