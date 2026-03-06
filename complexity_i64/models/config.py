"""
Complexity-I64 :: Config

Same architecture as complexity-deep but with integer compute mode.
Compatible with complexity-deep checkpoints (load float → quantize).

Training: float32 (standard PyTorch)
Inference: INT8 via quantize_all()

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

    # Dimensions
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
    initializer_range: float = 0.02
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 0

    # Token-Routed MLP
    use_token_routed_mlp: bool = True
    num_experts: int = 4

    # Attention
    use_qk_norm: bool = True
    use_sdpa: bool = True
    attention_dropout: float = 0.0

    # INL Dynamics
    dynamics_alpha: float = 0.9
    dynamics_beta: float = 0.1
    dynamics_gate: float = 0.5
    dynamics_dt: float = 0.1
    dynamics_controller_hidden: int = 64

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def expert_intermediate_size(self) -> int:
        return self.intermediate_size // self.num_experts

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

    @classmethod
    def from_dict(cls, config_dict: dict) -> "I64Config":
        import inspect
        valid_params = set(inspect.signature(cls.__init__).parameters.keys())
        valid_params.discard('self')
        filtered = {k: v for k, v in config_dict.items() if k in valid_params}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> "I64Config":
        """Load config from a YAML file (configs/pretrain/*.yaml)."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)["model"]
        return cls.from_dict(data)
