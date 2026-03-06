"""
Complexity-I64 :: Config

Same architecture as complexity-deep but with integer compute mode.
Compatible with complexity-deep checkpoints (load float → quantize).

INL - 2025
"""

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class I64Config:
    """Complexity-I64 model config. Mirrors complexity-deep for checkpoint compat."""

    model_type: str = "complexity-i64"
    architecture: str = "I64ForCausalLM"

    # Dimensions (same as complexity-deep)
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8

    # Positions
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0

    # Norms
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    # Embeddings
    tie_word_embeddings: bool = True
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 0

    # i64 routing
    use_token_routed_mlp: bool = True
    num_experts: int = 4

    # Attention
    use_qk_norm: bool = True

    # INL Dynamics
    dynamics_dt: float = 0.1
    dynamics_controller_hidden: int = 64

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @staticmethod
    def from_json(path: str) -> "I64Config":
        with open(path) as f:
            data = json.load(f)
        config = I64Config()
        for key, val in data.items():
            if key in ("parameters", "innovations"):
                continue
            if hasattr(config, key):
                setattr(config, key, val)
        return config

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    # Presets
    @classmethod
    def i64_1b(cls) -> "I64Config":
        return cls(hidden_size=2048, intermediate_size=5632,
                   num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=8)

    @classmethod
    def i64_150m(cls) -> "I64Config":
        return cls(vocab_size=100000, hidden_size=768, intermediate_size=2048,
                   num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4)
