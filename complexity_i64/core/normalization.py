"""
Complexity-I64 :: Integer RMSNorm

Float rsqrt (irreducible) + Q12 INT16 weight multiply.
Output: float (for RoPE/softmax downstream) or INT8 (for next matmul).

INL - 2025
"""

import torch
import torch.nn as nn

_Q_NORM = 128      # Q7 for normalized values
_Q_WEIGHT = 4096   # Q12 for weights


class I64RMSNorm(nn.Module):
    """Integer RMSNorm with optional fused INT8 output."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Weights stored as Q12 INT16 (set by quantize_weight)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm → float output."""
        if hasattr(self, 'weight_q12'):
            return self._forward_integer(x)
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight

    def _forward_integer(self, x: torch.Tensor) -> torch.Tensor:
        """Integer path: float rsqrt + Q7×Q12 → Q19, dequant."""
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        xn = x.float() * norm
        xn_q7 = (xn * _Q_NORM).round().to(torch.int32)
        out_q19 = xn_q7 * self.weight_q12.to(torch.int32)
        return (out_q19.float() / (_Q_NORM * _Q_WEIGHT)).type_as(x)

    def forward_with_int8_output(self, x: torch.Tensor):
        """Fused RMSNorm + INT8 quantize. Returns (float_output, (int8, scale))."""
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        if hasattr(self, 'weight_q12'):
            xn = x.float() * norm
            xn_q7 = (xn * _Q_NORM).round().to(torch.int32)
            out_q19 = xn_q7 * self.weight_q12.to(torch.int32)
            float_out = (out_q19.float() / (_Q_NORM * _Q_WEIGHT)).type_as(x)
        else:
            float_out = (x.float() * norm).type_as(x) * self.weight

        # INT8 quantize the output
        from complexity_i64.core.integer_ops import quantize_activation_int8
        flat = float_out.reshape(-1, float_out.shape[-1])
        x_int8, x_scale = quantize_activation_int8(flat)
        return float_out, (x_int8, x_scale)

    def quantize_weight(self):
        """Convert float weight to Q12 INT16."""
        w = self.weight.data.float()
        w_q12 = (w * _Q_WEIGHT).round().clamp(-32768, 32767).to(torch.int16)
        self.register_buffer('weight_q12', w_q12)
