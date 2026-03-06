"""
Complexity-I64 :: Training Utilities

Optimizer, scheduler, and checkpoint management.

INL - 2025
"""

import os
import re
import math
import glob
import logging

import torch
from torch.optim import AdamW

logger = logging.getLogger(__name__)


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
) -> AdamW:
    """
    Create AdamW optimizer with selective weight decay.
    No decay on: bias, norm layers, mu equilibrium parameters.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or ('.mu' in name and 'mu_proj' not in name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )

    logger.info("Optimizer: %d params with decay, %d without", len(decay_params), len(no_decay_params))
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 500,
    max_steps: int = 100000,
    restart_period: int = 50000,
    restart_mult: int = 2,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.

    Types: cosine, constant, cosine_restarts
    """
    if scheduler_type == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        logger.info("Scheduler: constant LR")

    elif scheduler_type == "cosine_restarts":
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        base_lr = optimizer.param_groups[0]['lr']
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=restart_period,
            T_mult=restart_mult,
            eta_min=base_lr * min_lr_ratio,
        )
        logger.info("Scheduler: cosine restarts (T_0=%d, T_mult=%d)", restart_period, restart_mult)

    else:  # cosine with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        logger.info("Scheduler: cosine (warmup=%d, max_steps=%d)", warmup_steps, max_steps)

    return scheduler


def cleanup_old_checkpoints(output_dir: str, keep_last: int = 5):
    """Keep only the last N checkpoints."""
    pattern = os.path.join(output_dir, "*.pt")
    checkpoints = glob.glob(pattern)

    checkpoints = [c for c in checkpoints if os.path.basename(c) not in ("last.pt", "final.pt")]

    if len(checkpoints) <= keep_last:
        return

    def sort_key(path):
        match = re.search(r'(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else 0

    checkpoints.sort(key=sort_key)

    for ckpt in checkpoints[:-keep_last]:
        try:
            os.remove(ckpt)
            logger.info("Deleted old checkpoint: %s", ckpt)
        except OSError:
            pass
