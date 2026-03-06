"""
Complexity-I64 :: Distributed Training (FSDP)

PyTorch FSDP for multi-GPU / multi-node training.
Launch via: torchrun --nproc_per_node=N train.py ...

INL - 2025
"""

import os
import logging
import functools

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType

logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed process group. Called by each rank."""
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    logger.info("Rank %d/%d initialized (local_rank=%d)", rank, world_size, local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    """Destroy the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """True if this is rank 0 or non-distributed."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_fsdp_mixed_precision(bf16: bool = False) -> MixedPrecision:
    """Mixed precision policy for FSDP."""
    if bf16 and torch.cuda.is_bf16_supported():
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    return MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )


def wrap_model_fsdp(
    model,
    bf16: bool = False,
    sharding_strategy: str = "full_shard",
) -> FSDP:
    """
    Wrap an I64Model with FSDP.

    Wraps each I64DecoderLayer individually for optimal memory/compute.
    """
    from complexity_i64.models.modeling import I64DecoderLayer

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={I64DecoderLayer},
    )

    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=get_fsdp_mixed_precision(bf16),
        sharding_strategy=strategy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        limit_all_gathers=True,
    )

    if is_main_process():
        logger.info("FSDP wrapped with strategy=%s, bf16=%s", sharding_strategy, bf16)

    return fsdp_model


def save_fsdp_checkpoint(model, optimizer, scheduler, step, checkpoint_dir, scaler=None):
    """Save full state dict checkpoint from FSDP model (only rank 0 writes)."""
    from pathlib import Path

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)

    if is_main_process():
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "scheduler_state_dict": scheduler.state_dict(),
        }
        if scaler is not None:
            save_dict["scaler_state_dict"] = scaler.state_dict()

        ckpt_path = path / f"step_{step}.pt"
        torch.save(save_dict, ckpt_path)
        torch.save(save_dict, path / "last.pt")
        logger.info("Saved FSDP checkpoint: %s", ckpt_path)

    dist.barrier()


def load_fsdp_checkpoint(model, optimizer, checkpoint_path, device="cpu"):
    """Load a full state dict checkpoint into FSDP model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model.load_state_dict(checkpoint["model_state_dict"])

        optim_state = FSDP.optim_state_dict_to_load(
            model, optimizer, checkpoint["optimizer_state_dict"]
        )
        optimizer.load_state_dict(optim_state)

    step = checkpoint.get("step", 0)
    logger.info("Loaded FSDP checkpoint from step %d", step)
    return step


def reduce_mean(tensor):
    """All-reduce a tensor and return the mean across ranks."""
    if not dist.is_initialized():
        return tensor
    t = tensor.clone().detach()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t
