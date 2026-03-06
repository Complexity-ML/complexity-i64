"""
Complexity-I64 :: Model

Train in float32 (standard PyTorch). Deploy in INT8 (quantize_all).

Architecture = Complexity Deep projected into integer:
- Every matmul → INT8 (torch._int_mm) after quantization
- SiLU, sigmoid, softplus → LUT lookups (zero FLOPs) after quantization
- RMSNorm weights → Q12 INT16 after quantization
- Float only where irreducible: rsqrt, RoPE, softmax

Compatible with complexity-deep checkpoints (same weight names).

INL - 2025
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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
    Complexity-I64 model.

    Training: float32 forward/backward (standard PyTorch).
    Inference: quantize_all() → INT8 matmuls, LUT activations.
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

        # Tie weights
        if config.tie_word_embeddings:
            # lm_head shares embed_tokens weight
            pass

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        velocity_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
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

            # Mu Residual Highway (EMA + norm clamp to prevent divergence)
            if mu_residual is None:
                mu_residual = mu_current
            else:
                mu_residual = 0.9 * mu_residual + 0.1 * mu_current
            # Norm clamp: preserve direction, bound magnitude
            mu_norm = mu_residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            mu_residual = torch.where(mu_norm > 10.0, mu_residual * (10.0 / mu_norm), mu_residual)
            mu_prev = mu_current + 0.1 * mu_residual

            if use_cache:
                new_past_key_values.append(new_kv)

        hidden = self.norm(hidden)

        # Logits: INT8 if quantized
        logits = self._compute_logits(hidden)

        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return I64CausalLMOutput(
            loss=loss,
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
            del self.lm_head

        self.requires_grad_(False)
        logger.info("Quantized to INT8: %d params", self.num_parameters())

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively with KV cache."""
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        past_key_values = None
        velocity_state = None

        for _ in range(max_new_tokens):
            if past_key_values is not None:
                curr_ids = input_ids[:, -1:]
                if velocity_state is not None:
                    velocity_state = velocity_state[:, -1:, :]
            else:
                curr_ids = input_ids

            outputs = self.forward(
                curr_ids,
                past_key_values=past_key_values,
                velocity_state=velocity_state,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            velocity_state = outputs.velocity_state

            raw_logits = outputs.logits[:, -1, :].clone()
            logits = raw_logits / temperature

            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            if do_sample:
                probs = F.softmax(logits, dim=-1)
                probs = torch.clamp(probs, min=1e-8)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(raw_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if (next_token == eos_token_id).all():
                break

        return input_ids

    def save_pretrained(self, save_path: str):
        """Save model weights + config.json."""
        import json
        from pathlib import Path
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cpu") -> "I64Model":
        """Load model from saved checkpoint directory."""
        import json
        from pathlib import Path
        path = Path(load_path)
        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)
        config = I64Config.from_dict(config_dict)
        model = cls(config)
        pt_path = path / "model.pt"
        if pt_path.exists():
            state_dict = torch.load(pt_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        model = model.to(device)
        return model

    @staticmethod
    def from_complexity_deep(checkpoint_path: str, config: I64Config, device="cpu"):
        """Load a complexity-deep checkpoint and quantize to INT8."""
        model = I64Model(config)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.quantize_all()
        return model


# Convenience factory
def create_i64_model(config_path: str, vocab_size: Optional[int] = None) -> I64Model:
    """Create an I64Model from a YAML config file."""
    import yaml
    with open(config_path) as f:
        model_yaml = yaml.safe_load(f)["model"]
    config = I64Config.from_dict(model_yaml)
    if vocab_size is not None:
        config.vocab_size = vocab_size
    return I64Model(config)
