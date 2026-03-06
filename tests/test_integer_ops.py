"""
Tests for INT8 primitives: quantize, LUT activations, int8_linear.

INL - 2025
"""

import torch
import torch.nn.functional as F
import pytest

from complexity_i64.core.integer_ops import (
    quantize_weight_int8,
    quantize_activation_int8,
    int8_linear,
    int8_fused_gate_up,
    silu_integer,
    sigmoid_integer,
    softplus_integer,
    silu_multiply_integer,
    _Q7,
)


# ============================================================================
# Quantize / Dequantize round-trip
# ============================================================================

class TestQuantize:
    def test_weight_round_trip(self):
        """Quantize → dequantize should approximate original."""
        w = torch.randn(64, 128)
        wq, ws = quantize_weight_int8(w)
        w_recon = wq.float() * ws.unsqueeze(-1)
        # Per-channel INT8: expect < 1% relative error on well-distributed weights
        rel_err = (w - w_recon).abs().mean() / w.abs().mean()
        assert rel_err < 0.05, f"Weight round-trip error too high: {rel_err:.4f}"

    def test_activation_round_trip(self):
        """Dynamic per-token quantize → dequantize."""
        x = torch.randn(32, 128)
        xq, xs = quantize_activation_int8(x)
        x_recon = xq.float() * xs.unsqueeze(-1)
        rel_err = (x - x_recon).abs().mean() / x.abs().mean()
        assert rel_err < 0.05, f"Activation round-trip error too high: {rel_err:.4f}"

    def test_quantize_range(self):
        """INT8 values must be in [-128, 127]."""
        w = torch.randn(32, 64) * 10
        wq, _ = quantize_weight_int8(w)
        assert wq.min() >= -128
        assert wq.max() <= 127
        assert wq.dtype == torch.int8

    def test_zero_input(self):
        """Quantizing zeros should produce zeros."""
        w = torch.zeros(16, 32)
        wq, ws = quantize_weight_int8(w)
        assert (wq == 0).all()


# ============================================================================
# INT8 Linear vs float
# ============================================================================

class TestInt8Linear:
    def test_vs_float_linear(self):
        """INT8 linear should approximate F.linear."""
        torch.manual_seed(42)
        x = torch.randn(8, 64)
        w = torch.randn(128, 64)

        wq, ws = quantize_weight_int8(w)

        out_int8 = int8_linear(x, wq, ws)
        out_float = F.linear(x, w)

        # INT8 introduces quantization noise; check correlation, not exact match
        cos_sim = F.cosine_similarity(out_int8.flatten(), out_float.flatten(), dim=0)
        assert cos_sim > 0.95, f"INT8 linear too far from float: cos_sim={cos_sim:.4f}"

    def test_with_bias(self):
        """INT8 linear with bias."""
        x = torch.randn(4, 32)
        w = torch.randn(64, 32)
        b = torch.randn(64)

        wq, ws = quantize_weight_int8(w)
        out = int8_linear(x, wq, ws, bias=b)
        assert out.shape == (4, 64)

    def test_batched(self):
        """INT8 linear on 3D input."""
        x = torch.randn(2, 8, 64)
        w = torch.randn(32, 64)
        wq, ws = quantize_weight_int8(w)
        out = int8_linear(x, wq, ws)
        assert out.shape == (2, 8, 32)

    def test_fused_gate_up(self):
        """Fused gate+up should match separate projections."""
        torch.manual_seed(42)
        x = torch.randn(8, 64)
        w_gate = torch.randn(32, 64)
        w_up = torch.randn(32, 64)

        wq_g, ws_g = quantize_weight_int8(w_gate)
        wq_u, ws_u = quantize_weight_int8(w_up)

        fused_int8 = torch.cat([wq_g, wq_u], dim=0)
        fused_scale = torch.cat([ws_g, ws_u])

        gate, up = int8_fused_gate_up(x, fused_int8, fused_scale, 32)

        gate_sep = int8_linear(x, wq_g, ws_g)
        up_sep = int8_linear(x, wq_u, ws_u)

        assert torch.allclose(gate, gate_sep, atol=1e-5)
        assert torch.allclose(up, up_sep, atol=1e-5)


# ============================================================================
# LUT activations vs float
# ============================================================================

class TestLUTActivations:
    def test_silu_accuracy(self):
        """LUT SiLU should approximate float SiLU."""
        x_float = torch.linspace(-4, 4, 100)
        x_q7 = (x_float * _Q7).round().to(torch.int32)

        lut_result = silu_integer(x_q7).float() / _Q7
        float_result = F.silu(x_float)

        max_err = (lut_result - float_result).abs().max()
        assert max_err < 0.02, f"SiLU LUT max error: {max_err:.4f}"

    def test_sigmoid_accuracy(self):
        """LUT sigmoid should approximate float sigmoid."""
        x_float = torch.linspace(-4, 4, 100)
        x_q7 = (x_float * _Q7).round().to(torch.int32)

        lut_result = sigmoid_integer(x_q7).float() / _Q7
        float_result = torch.sigmoid(x_float)

        max_err = (lut_result - float_result).abs().max()
        assert max_err < 0.02, f"Sigmoid LUT max error: {max_err:.4f}"

    def test_softplus_accuracy(self):
        """LUT softplus should approximate float softplus."""
        x_float = torch.linspace(-4, 4, 100)
        x_q7 = (x_float * _Q7).round().to(torch.int32)

        lut_result = softplus_integer(x_q7).float() / _Q7
        float_result = F.softplus(x_float)

        max_err = (lut_result - float_result).abs().max()
        assert max_err < 0.02, f"Softplus LUT max error: {max_err:.4f}"

    def test_silu_saturation(self):
        """SiLU should handle extreme values (clamped to LUT range)."""
        x_large = torch.tensor([5000, -5000], dtype=torch.int32)
        result = silu_integer(x_large)
        # x >> 0: silu(x) ≈ x, so result ≈ x
        assert result[0] == 5000
        # x << 0: silu(x) ≈ 0
        assert result[1] == 0

    def test_sigmoid_bounds(self):
        """Sigmoid output should be in [0, Q7]."""
        x = torch.randint(-2000, 2000, (100,), dtype=torch.int32)
        result = sigmoid_integer(x)
        assert (result >= 0).all()
        assert (result <= _Q7).all()

    def test_silu_multiply(self):
        """silu_multiply_integer should approximate float silu(gate) * up."""
        gate = torch.randn(16, 64)
        up = torch.randn(16, 64)

        int_result = silu_multiply_integer(gate, up)
        float_result = F.silu(gate) * up

        cos_sim = F.cosine_similarity(int_result.flatten(), float_result.flatten(), dim=0)
        assert cos_sim > 0.95, f"silu_multiply too far from float: cos_sim={cos_sim:.4f}"


# ============================================================================
# Module-level tests
# ============================================================================

class TestModuleQuantize:
    def test_attention_quantize(self):
        """Attention should work after quantize()."""
        from complexity_i64.core.attention import I64Attention
        attn = I64Attention(64, 4, 2)
        attn.quantize()

        x = torch.randn(2, 8, 64)
        pos = torch.arange(8).unsqueeze(0).expand(2, -1)
        out, _ = attn(x, pos)
        assert out.shape == (2, 8, 64)

    def test_mlp_quantize(self):
        """MLP should work after quantize()."""
        from complexity_i64.core.mlp import I64MLP
        mlp = I64MLP(64, 128)
        mlp.quantize()

        x = torch.randn(4, 64)
        out = mlp(x)
        assert out.shape == (4, 64)

    def test_dynamics_quantize(self):
        """Dynamics should work after quantize()."""
        from complexity_i64.core.dynamics import I64Dynamics
        dyn = I64Dynamics(64, controller_hidden=16)
        dyn.quantize()

        h = torch.randn(4, 64)
        h_next, v_next, mu = dyn(h)
        assert h_next.shape == (4, 64)
        assert v_next.shape == (4, 64)

    def test_full_model_forward(self):
        """Full model forward pass (tiny config)."""
        from complexity_i64.models.config import I64Config
        from complexity_i64.models.modeling import I64Model

        config = I64Config(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        )
        model = I64Model(config)

        input_ids = torch.randint(0, 256, (2, 16))
        out = model(input_ids)
        assert out.logits.shape == (2, 16, 256)

    def test_full_model_quantize_forward(self):
        """Full model should work after quantize_all()."""
        from complexity_i64.models.config import I64Config
        from complexity_i64.models.modeling import I64Model

        config = I64Config(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        )
        model = I64Model(config)
        model.quantize_all()

        input_ids = torch.randint(0, 256, (2, 16))
        out = model(input_ids)
        assert out.logits.shape == (2, 16, 256)
