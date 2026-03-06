"""
Complexity-I64 :: Integer-Native Model

Same architecture as Complexity Deep, projected into integer:
- Every matmul → INT8 (torch._int_mm)
- SiLU, sigmoid, softplus → LUT lookups (zero FLOPs)
- RMSNorm weights → Q12 INT16
- Float only where irreducible: rsqrt, RoPE, softmax

Compatible with complexity-deep checkpoints:
    1. Load float weights from checkpoint
    2. Call model.quantize_all() → converts everything to INT8
    3. Inference runs in integer

INL - 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

from complexity_i64.models.config import I64Config
from complexity_i64.core.normalization import I64RMSNorm
from complexity_i64.core.attention import I64Attention
from complexity_i64.core.dynamics import I64Dynamics
from complexity_i64.core.mlp import I64TokenRoutedMLP, I64MLP


@dataclass
class I64Output:
    last_hidden_state: torch.Tensor
    last_velocity_state: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None


@dataclass
class I64CausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    velocity_state: Optional[torch.Tensor] = None


class I64DecoderLayer(nn.Module):
    """
    Integer decoder layer:
      1. I64RMSNorm → I64Attention (INT8 QKV, float softmax, INT8 O)
      2. I64Dynamics (INT8 controller, LUT activations)
      3. I64RMSNorm → I64TokenRoutedMLP (INT8 experts, LUT SiLU)
    """

    def __init__(self, config: I64Config):
        super().__init__()
        self.input_layernorm = I64RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.self_attn = I64Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            use_qk_norm=config.use_qk_norm,
        )

        self.dynamics = I64Dynamics(
            hidden_size=config.hidden_size,
            controller_hidden=config.dynamics_controller_hidden,
            dt=config.dynamics_dt,
        )

        self.post_attention_layernorm = I64RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.use_token_routed_mlp:
            self.mlp = I64TokenRoutedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                vocab_size=config.vocab_size,
            )
        else:
            self.mlp = I64MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        mu_prev: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:

        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden, new_kv = self.self_attn(
            hidden, positions, mu_prev=mu_prev,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        hidden, velocity, mu_current = self.dynamics(hidden, velocity)
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)

        if isinstance(self.mlp, I64TokenRoutedMLP):
            # Flatten for token routing
            shape = hidden.shape
            flat = hidden.reshape(-1, shape[-1])
            tid = token_ids.reshape(-1) if token_ids is not None else None
            mu_flat = mu_current.reshape(-1, shape[-1]) if mu_current is not None else None
            hidden = self.mlp(flat, token_ids=tid, mu=mu_flat).reshape(shape)
        else:
            hidden = self.mlp(hidden)

        hidden = residual + hidden

        return hidden, velocity, mu_current, new_kv

    def quantize(self):
        """Quantize all sub-modules to INT8."""
        self.input_layernorm.quantize_weight()
        self.self_attn.quantize()
        self.dynamics.quantize()
        self.post_attention_layernorm.quantize_weight()
        self.mlp.quantize()


class I64Model(nn.Module):
    """
    Complexity-I64: Integer-native transformer.

    Load complexity-deep checkpoint → quantize_all() → pure integer inference.
    """

    def __init__(self, config: I64Config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            I64DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

        self.norm = I64RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        velocity_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> I64CausalLMOutput:

        hidden = self.embed_tokens(input_ids.long())

        if positions is None:
            seq_len = hidden.shape[1] if hidden.dim() == 3 else 1
            offset = past_key_values[0][0].shape[2] if past_key_values else 0
            positions = torch.arange(offset, offset + seq_len, device=hidden.device)
            if hidden.dim() == 3:
                positions = positions.unsqueeze(0).expand(hidden.shape[0], -1)

        velocity = velocity_state
        if velocity is None:
            velocity = torch.zeros_like(hidden)

        mu_prev = None
        mu_residual = None
        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values else None
            hidden, velocity, mu_current, new_kv = layer(
                hidden, positions, velocity,
                token_ids=input_ids,
                mu_prev=mu_prev,
                attention_mask=attention_mask,
                past_key_value=pkv,
                use_cache=use_cache,
            )

            # Mu Residual Highway
            if mu_residual is None:
                mu_residual = mu_current.clone()
            else:
                mu_residual = mu_residual + mu_current
            mu_prev = mu_current + 0.1 * mu_residual

            if use_cache:
                new_past_key_values.append(new_kv)

        hidden = self.norm(hidden)

        # Logits: INT8 if quantized
        logits = self._compute_logits(hidden)

        return I64CausalLMOutput(
            logits=logits,
            past_key_values=new_past_key_values,
            velocity_state=velocity,
        )

    def _compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'lm_head_int8'):
            from complexity_i64.core.integer_ops import int8_linear
            return int8_linear(hidden.reshape(-1, hidden.shape[-1]),
                               self.lm_head_int8, self.lm_head_scale).reshape(*hidden.shape[:-1], -1)
        if self.tie_word_embeddings:
            if hasattr(self, 'embed_int8'):
                from complexity_i64.core.integer_ops import int8_linear
                return int8_linear(hidden.reshape(-1, hidden.shape[-1]),
                                   self.embed_int8, self.embed_scale).reshape(*hidden.shape[:-1], -1)
            return F.linear(hidden.float(), self.embed_tokens.weight.float())
        return F.linear(hidden.float(), self.lm_head.weight.float())

    def quantize_all(self):
        """Convert entire model to INT8. Call after loading checkpoint."""
        for layer in self.layers:
            layer.quantize()
        self.norm.quantize_weight()

        # lm_head
        from complexity_i64.core.integer_ops import quantize_weight_int8
        if self.tie_word_embeddings:
            wq, ws = quantize_weight_int8(self.embed_tokens.weight.data)
            self.register_buffer("embed_int8", wq)
            self.register_buffer("embed_scale", ws)
        elif hasattr(self, 'lm_head'):
            wq, ws = quantize_weight_int8(self.lm_head.weight.data)
            self.register_buffer("lm_head_int8", wq)
            self.register_buffer("lm_head_scale", ws)
            self.lm_head.weight = None

        self.requires_grad_(False)
        print(f"  Quantized to INT8: {self.num_parameters():,} params")

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def from_complexity_deep(checkpoint_path: str, config: I64Config, device="cpu"):
        """Load a complexity-deep checkpoint into I64Model and quantize."""
        model = I64Model(config)

        # Load weights (complexity-deep → i64 mapping is 1:1)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Remap keys if needed (complexity_deep naming → i64 naming)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.quantize_all()
        return model
