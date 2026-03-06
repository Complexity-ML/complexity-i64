"""
Complexity-I64 :: Integer INL Dynamics

PID-like control with velocity tracking — all in integer.

Controller MLP: INT8 matmuls
Activations: LUT sigmoid, LUT softplus (zero FLOPs)
mu/velocity arithmetic: float (accumulation needs precision)

    error = h - mu(h)
    v_next = sigmoid_lut(alpha_raw) * v - clamp(softplus_lut(beta_raw), 2.0) * error
    h_next = h + dt * sigmoid_lut(gate_raw) * v_next

INL - 2025
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from complexity_i64.core.integer_ops import (
    int8_linear, quantize_weight_int8,
    sigmoid_integer, softplus_integer,
    _Q7,
)


class I64Dynamics(nn.Module):
    """
    Integer INL Dynamics.

    Controller: 2 INT8 matmuls (controller_in, controller_out)
    SiLU: LUT (controller hidden activation)
    Sigmoid/Softplus: LUT (alpha, beta, gate)
    mu_proj: INT8 matmul
    """

    def __init__(self, hidden_size: int, controller_hidden: int = 64, dt: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt

        self.mu = nn.Parameter(torch.zeros(hidden_size))
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)

        self.controller_in = nn.Linear(hidden_size * 2, controller_hidden)
        self.controller_out = nn.Linear(controller_hidden, hidden_size * 3)

        with torch.no_grad():
            bias = self.controller_out.bias
            bias[:hidden_size].fill_(2.2)
            bias[hidden_size:hidden_size*2].fill_(-2.2)
            bias[hidden_size*2:].fill_(0.0)
            self.controller_out.weight.normal_(0, 0.01)

    def forward(
        self, h: torch.Tensor, v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if v is None:
            v = torch.zeros_like(h)

        hv = torch.cat([h, v], dim=-1)

        # Controller: INT8 matmuls + LUT SiLU
        if hasattr(self, 'ctrl_in_int8'):
            ctrl = int8_linear(hv, self.ctrl_in_int8, self.ctrl_in_scale,
                               self.ctrl_in_bias)
            # SiLU via LUT
            from complexity_i64.core.integer_ops import silu_integer
            ctrl_q7 = (ctrl.float() * _Q7).round().to(torch.int32)
            ctrl = silu_integer(ctrl_q7).float() / _Q7
            ctrl_out = int8_linear(ctrl, self.ctrl_out_int8, self.ctrl_out_scale,
                                   self.ctrl_out_bias)
        else:
            import torch.nn.functional as F
            ctrl = F.silu(self.controller_in(hv))
            ctrl_out = self.controller_out(ctrl)

        alpha_raw, beta_raw, gate_raw = torch.split(ctrl_out, self.hidden_size, dim=-1)

        # LUT activations for alpha, beta, gate
        if hasattr(self, 'ctrl_in_int8'):
            alpha_q7 = (alpha_raw.float() * _Q7).round().to(torch.int32)
            alpha = sigmoid_integer(alpha_q7).float() / _Q7

            beta_q7 = (beta_raw.float() * _Q7).round().to(torch.int32)
            beta = softplus_integer(beta_q7).float() / _Q7
            beta = beta.clamp(max=2.0)

            gate_q7 = (gate_raw.float() * _Q7).round().to(torch.int32)
            gate = sigmoid_integer(gate_q7).float() / _Q7
        else:
            import torch.nn.functional as F
            alpha = torch.sigmoid(alpha_raw)
            beta = torch.clamp(F.softplus(beta_raw), max=2.0)
            gate = torch.sigmoid(gate_raw)

        # mu projection: INT8 if available
        if hasattr(self, 'mu_proj_int8'):
            mu_contextual = self.mu + int8_linear(h, self.mu_proj_int8, self.mu_proj_scale)
        else:
            mu_contextual = self.mu + self.mu_proj(h)

        error = h - mu_contextual
        v_next = alpha * v - beta * error
        v_next = torch.clamp(v_next, min=-10.0, max=10.0)
        h_next = h + self.dt * gate * v_next

        return h_next, v_next, mu_contextual

    def quantize(self):
        """Convert controller weights to INT8."""
        # Controller in
        ciq, cis = quantize_weight_int8(self.controller_in.weight.data)
        self.register_buffer("ctrl_in_int8", ciq)
        self.register_buffer("ctrl_in_scale", cis)

        # Controller out
        coq, cos = quantize_weight_int8(self.controller_out.weight.data)
        self.register_buffer("ctrl_out_int8", coq)
        self.register_buffer("ctrl_out_scale", cos)

        # mu_proj
        mpq, mps = quantize_weight_int8(self.mu_proj.weight.data)
        self.register_buffer("mu_proj_int8", mpq)
        self.register_buffer("mu_proj_scale", mps)

        # Free float weights (keep biases as standalone buffers)
        self.register_buffer("ctrl_in_bias", self.controller_in.bias.data.clone())
        self.register_buffer("ctrl_out_bias", self.controller_out.bias.data.clone())
        del self.controller_in
        del self.controller_out
        del self.mu_proj
