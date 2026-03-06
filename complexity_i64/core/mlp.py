"""
Complexity-I64 :: Integer MLP

SwiGLU MLP where every matmul is INT8 and SiLU is a LUT lookup.

Forward: y = down_i8(silu_lut(gate_i8(x)) * up_i8(x))

Three INT8 matmuls + one LUT lookup. Zero float compute in the MLP.

INL - 2025
"""

import torch
import torch.nn as nn
from typing import Optional

from complexity_i64.core.integer_ops import (
    int8_linear, int8_fused_gate_up, silu_multiply_integer,
    quantize_weight_int8,
)


class I64MLP(nn.Module):
    """
    Integer SwiGLU MLP.

    Weights stored as INT8 natively. gate+up fused into single matmul.
    SiLU computed via 2049-entry LUT (zero FLOPs).
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Float weights for training — quantized post-init for inference
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'gate_up_int8'):
            return self._forward_int8(x)
        # Float fallback (training)
        import torch.nn.functional as F
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def _forward_int8(self, x: torch.Tensor) -> torch.Tensor:
        """Full INT8 path: fused gate+up → LUT SiLU×up → INT8 down."""
        gate, up = int8_fused_gate_up(
            x, self.gate_up_int8, self.gate_up_scale, self.intermediate_size,
        )
        inter = silu_multiply_integer(gate, up)
        return int8_linear(inter, self.down_int8, self.down_scale)

    def quantize(self):
        """Convert float weights to INT8 for inference."""
        gq, gs = quantize_weight_int8(self.gate_proj.weight.data)
        uq, us = quantize_weight_int8(self.up_proj.weight.data)
        dq, ds = quantize_weight_int8(self.down_proj.weight.data)

        # Fused gate+up
        self.register_buffer("gate_up_int8", torch.cat([gq, uq], dim=0))
        self.register_buffer("gate_up_scale", torch.cat([gs, us]))
        self.register_buffer("down_int8", dq)
        self.register_buffer("down_scale", ds)

        # Free float weights
        self.gate_proj.weight = None
        self.up_proj.weight = None
        self.down_proj.weight = None


class I64TokenRoutedMLP(nn.Module):
    """
    Integer token-routed MLP. i64 routing + INT8 expert compute.

    Routing: expert_id = token_id % num_experts (pure integer, no gate)
    Expert compute: INT8 SwiGLU per expert
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_experts: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_inter = intermediate_size // num_experts

        # Expert weights (will be quantized to INT8)
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, 2 * self.expert_inter)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, self.expert_inter, hidden_size)
        )

        # i64 routing table
        self.register_buffer(
            "token_to_expert",
            torch.arange(vocab_size, dtype=torch.long) % num_experts,
        )

        nn.init.kaiming_uniform_(self.gate_up_proj, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_proj, a=5**0.5)

    def route(self, token_ids: Optional[torch.Tensor], num_tokens: int,
              device: torch.device, mu: Optional[torch.Tensor] = None) -> torch.Tensor:
        if token_ids is None:
            return torch.zeros(num_tokens, dtype=torch.long, device=device)
        ids = token_ids.clamp(0, self.token_to_expert.shape[0] - 1)
        base_ids = self.token_to_expert[ids]

        # Mu-guided routing override (if mu_router exists)
        if mu is not None and hasattr(self, 'mu_router_int8'):
            from complexity_i64.core.integer_ops import int8_linear
            mu_logits = int8_linear(mu, self.mu_router_int8, self.mu_router_scale)
            import torch.nn.functional as F
            base_one_hot = F.one_hot(base_ids, self.num_experts).float()
            combined = base_one_hot * 10.0 + mu_logits
            return combined.argmax(dim=-1)

        return base_ids

    def forward(self, x: torch.Tensor, token_ids: Optional[torch.Tensor] = None,
                mu: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        expert_ids = self.route(token_ids, x.shape[0], x.device, mu=mu)
        return self._expert_forward(x, expert_ids)

    def _expert_forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """Dispatch tokens to experts, compute, gather."""
        if hasattr(self, 'gate_up_int8'):
            return self._expert_forward_int8(x, expert_ids)

        output = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = expert_ids == e
            if not mask.any():
                continue
            xe = x[mask]
            gu = torch.mm(xe, self.gate_up_proj[e])
            gate, up = gu.split(self.expert_inter, dim=-1)
            import torch.nn.functional as F
            inter = F.silu(gate) * up
            output[mask] = torch.mm(inter, self.down_proj[e])
        return output

    def _expert_forward_int8(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """INT8 expert forward with LUT SiLU."""
        from complexity_i64.core.integer_ops import int8_linear, silu_multiply_integer
        output = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = expert_ids == e
            if not mask.any():
                continue
            xe = x[mask]
            gu = int8_linear(xe, self.gate_up_int8[e], self.gate_up_scale[e])
            gate, up = gu.split(self.expert_inter, dim=-1)
            inter = silu_multiply_integer(gate, up)
            output[mask] = int8_linear(inter, self.down_int8[e], self.down_scale[e])
        return output

    def quantize(self):
        """Quantize expert weights to INT8."""
        gu_q, gu_s, dn_q, dn_s = [], [], [], []
        for e in range(self.num_experts):
            gq, gs = quantize_weight_int8(self.gate_up_proj[e].t())  # (2*inter, hidden)
            dq, ds = quantize_weight_int8(self.down_proj[e].t())     # (hidden, inter)
            gu_q.append(gq); gu_s.append(gs)
            dn_q.append(dq); dn_s.append(ds)

        self.register_buffer("gate_up_int8", torch.stack(gu_q))
        self.register_buffer("gate_up_scale", torch.stack(gu_s))
        self.register_buffer("down_int8", torch.stack(dn_q))
        self.register_buffer("down_scale", torch.stack(dn_s))
