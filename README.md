# Complexity-I64

Integer-native Complexity architecture. Train in float32, deploy in INT8.

## Architecture

Complexity-I64 projects the [Complexity-Deep](../complexity-deep/) architecture into integer arithmetic:

- **Every matmul** → INT8 via `torch._int_mm` (INT8×INT8→INT32)
- **SiLU, sigmoid, softplus** → LUT lookups (2049 entries, zero FLOPs)
- **RMSNorm weights** → Q12 INT16
- **Float only where irreducible**: rsqrt, RoPE, softmax

### Key Components

| Component | Training (float32) | Inference (INT8) |
|---|---|---|
| QKV + O projections | `nn.Linear` | Fused INT8 matmul |
| MLP (SwiGLU) | `nn.Linear` + `F.silu` | INT8 matmul + LUT SiLU |
| Token-Routed MLP | Per-expert `nn.Linear` | Per-expert INT8 matmul |
| Dynamics controller | `nn.Linear` | INT8 matmul + LUT activations |
| RMSNorm | float weight | Q12 INT16 weight multiply |

### Innovations (from Complexity-Deep)

1. **Token-Routed MLP** — Deterministic routing (`token_id % num_experts`). Guarantees perfect load balance by construction, no auxiliary loss needed. Experts specialize via orthogonal gradient updates on disjoint token subsets.

2. **Mu-Guided Attention** — Latent state mu from dynamics influences Q, K, V projections, creating bidirectional information flow between layers.

3. **PID-Style Dynamic Scaler** — Adaptive controller (alpha/beta/gate) stabilizes training via velocity tracking and error correction.

## Quick Start

```bash
pip install -e .
```

### Pre-training

```bash
python train.py \
    --model configs/pretrain/1b.yaml \
    --data configs/data/pretrain.yaml \
    --lr 3e-5 --batch-size 32 --max-steps 100000
```

### SFT (Conversational Fine-tuning)

```bash
python sft.py --config configs/sft/chat.yaml
python sft.py --config configs/sft/chat_lora.yaml  # with LoRA
```

### Quantize for Inference

```python
from complexity_i64 import I64Model

model = I64Model.from_pretrained("./checkpoints/final")
model.quantize_all()  # float32 → INT8
output = model.generate(input_ids, max_new_tokens=100)
```

### Train Tokenizer

```bash
pip install -e ".[tokenizer]"
python train_tokenizer.py --dataset Pacific-Prime/edu-web --vocab-size 32000
```

## Model Sizes

| Config | Params | Layers | Hidden | Heads | KV Heads | Experts |
|--------|--------|--------|--------|-------|----------|---------|
| 1b | ~1.5B | 24 | 2048 | 16 | 8 | 4 |
| 3b | ~3.1B | 32 | 2560 | 32 | 8 | 8 |
| 7b | ~7.8B | 32 | 4096 | 32 | 8 | 8 |

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Project Structure

```
complexity_i64/
    core/
        integer_ops.py    # INT8 primitives, quantize/dequant, LUT activations
        attention.py       # I64Attention (INT8 QKV + float softmax)
        mlp.py             # I64MLP + I64TokenRoutedMLP (INT8 SwiGLU)
        dynamics.py        # I64Dynamics (PID controller, INT8 + LUT)
        normalization.py   # I64RMSNorm (Q12 INT16 weights)
    models/
        config.py          # I64Config (YAML-driven)
        modeling.py        # I64Model (full causal LM)
    training/
        trainer.py         # Training loop (pre-train + SFT)
        utils.py           # Optimizer, scheduler, checkpoint utils
    data/
        datasets.py        # Streaming, pre-tokenized, conversational SFT
configs/
    pretrain/
        1b.yaml            # ~1.5B params
        3b.yaml            # ~3.1B params
        7b.yaml            # ~7.8B params
    data/pretrain.yaml     # Data pipeline config
    sft/chat.yaml          # SFT config
    sft/chat_lora.yaml     # SFT with LoRA config
```

## Checkpoint Compatibility

Complexity-I64 uses the same weight names as Complexity-Deep. Load a float checkpoint and quantize:

```python
model = I64Model.from_complexity_deep("path/to/checkpoint.pt", config)
# Returns a fully quantized INT8 model
```

INL - 2025
