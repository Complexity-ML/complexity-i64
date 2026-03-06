"""
Complexity-I64 :: Pre-Training Script

Train an I64 model from scratch in float32.
After training, call model.quantize_all() for INT8 inference.

Single GPU:
    python train.py --model configs/pretrain/1b.yaml \
                    --data configs/data/pretrain.yaml \
                    --lr 3e-5 --batch-size 32

Multi-GPU (FSDP):
    torchrun --nproc_per_node=4 train.py \
        --model configs/pretrain/3b.yaml \
        --data configs/data/pretrain.yaml \
        --lr 3e-5 --batch-size 8 --fsdp

Multi-node:
    torchrun --nnodes=2 --nproc_per_node=4 \
        --rdzv_backend=c10d --rdzv_endpoint=HOST:PORT \
        train.py --model configs/pretrain/7b.yaml \
        --data configs/data/pretrain.yaml --fsdp --bf16

INL - 2025
"""

import os
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from complexity_i64.models.modeling import I64Model
from complexity_i64.models.config import I64Config
from complexity_i64.data.datasets import (
    PreTokenizedDataset,
    StreamingTextDataset,
    collate_fn,
)
from complexity_i64.training.trainer import Trainer
from complexity_i64.training.utils import create_optimizer, create_scheduler
from complexity_i64.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    wrap_model_fsdp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("complexity_i64.train")


def main():
    parser = argparse.ArgumentParser(description="Train Complexity-I64")

    # Configs
    parser.add_argument("--model", type=str, required=True,
                        help="Model architecture YAML (configs/pretrain/default.yaml)")
    parser.add_argument("--data", type=str, required=True,
                        help="Dataset YAML (configs/data/pretrain.yaml)")

    # Training (CLI)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "constant", "cosine_restarts"])

    # Precision
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-amp", action="store_true")

    # Paths
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--tensorboard-dir", type=str, default="./runs")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-data", type=str, default=None,
                        help="Eval dataset YAML (same format as --data)")

    # Resume
    parser.add_argument("--resume", type=str, default=None)

    # Distributed (FSDP)
    parser.add_argument("--fsdp", action="store_true", help="Enable FSDP distributed training")
    parser.add_argument("--sharding-strategy", type=str, default="full_shard",
                        choices=["full_shard", "shard_grad_op", "no_shard"])
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing (saves VRAM)")

    # Other
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")

    args = parser.parse_args()

    # Distributed setup
    use_distributed = args.fsdp and torch.cuda.is_available()
    if use_distributed:
        rank, world_size, local_rank = setup_distributed()
    else:
        rank, world_size, local_rank = 0, 1, 0

    # Load model config
    with open(args.model) as f:
        model_yaml = yaml.safe_load(f)["model"]

    # Load data config
    with open(args.data) as f:
        data_yaml = yaml.safe_load(f)

    # Device
    if use_distributed:
        device = torch.device(f"cuda:{local_rank}")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if is_main_process():
        logger.info("Model config: %s", args.model)
        logger.info("Data config: %s", args.data)
        logger.info("Device: %s (world_size=%d)", device, world_size)
        if torch.cuda.is_available():
            logger.info("GPU: %s", torch.cuda.get_device_name())

    # Tokenizer
    if os.path.exists(args.tokenizer):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    else:
        if is_main_process():
            logger.warning("Tokenizer not found at %s, using GPT-2", args.tokenizer)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if is_main_process():
        logger.info("Vocab size: %d", len(tokenizer))

    # Model from YAML config
    config = I64Config.from_dict(model_yaml)
    config.vocab_size = model_yaml.get("vocab_size", len(tokenizer))
    model = I64Model(config)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        if is_main_process():
            logger.info("Gradient checkpointing enabled")

    if use_distributed:
        model = wrap_model_fsdp(model, bf16=args.bf16, sharding_strategy=args.sharding_strategy)
    else:
        model = model.to(device)

    if is_main_process():
        logger.info("Parameters: %d", sum(p.numel() for p in model.parameters()))

    # Optimizer & scheduler
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    # Dataset
    source = data_yaml.get("source", "streaming")
    max_length = data_yaml.get("max_length", 512)

    if source == "pretokenized":
        train_dataset = PreTokenizedDataset(data_yaml["data_dir"], max_length=max_length)
        sampler = DistributedSampler(train_dataset, shuffle=True) if use_distributed else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=data_yaml.get("num_workers", 4),
            pin_memory=True,
        )
    else:
        train_dataset = StreamingTextDataset(
            dataset_name=data_yaml["dataset"],
            tokenizer=tokenizer,
            max_length=max_length,
            text_field=data_yaml.get("text_field", "text"),
            subset=data_yaml.get("subset", None),
            exclude_sources=data_yaml.get("exclude_sources", None),
            token=args.token,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=0,
        )

    # Eval dataset (optional)
    eval_loader = None
    if args.eval_data:
        with open(args.eval_data) as f:
            eval_yaml = yaml.safe_load(f)
        eval_source = eval_yaml.get("source", "streaming")
        eval_max_length = eval_yaml.get("max_length", max_length)

        if eval_source == "pretokenized":
            eval_dataset = PreTokenizedDataset(eval_yaml["data_dir"], max_length=eval_max_length)
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=eval_yaml.get("num_workers", 2),
                pin_memory=True,
            )
        else:
            eval_dataset = StreamingTextDataset(
                dataset_name=eval_yaml["dataset"],
                tokenizer=tokenizer,
                max_length=eval_max_length,
                text_field=eval_yaml.get("text_field", "text"),
                subset=eval_yaml.get("subset", None),
                split=eval_yaml.get("split", "validation"),
                token=args.token,
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                num_workers=0,
            )
        if is_main_process():
            logger.info("Eval dataset: %s", args.eval_data)

    # Trainer
    run_name = f"i64_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.tensorboard_dir,
        run_name=run_name,
        use_amp=not args.no_amp,
        bf16=args.bf16,
        gradient_accumulation=args.gradient_accumulation,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        distributed=use_distributed,
    )

    # Resume
    start_step = 0
    if args.resume:
        start_step = trainer.resume(args.resume, lr=args.lr)

    if is_main_process():
        logger.info("Starting training — batch=%d, grad_accum=%d, lr=%.2e, max_steps=%d, gpus=%d",
                    args.batch_size, args.gradient_accumulation, args.lr, args.max_steps, world_size)

    final_step = trainer.train(
        train_loader,
        max_steps=args.max_steps,
        start_step=start_step,
        eval_loader=eval_loader,
        eval_interval=args.eval_interval,
    )

    if is_main_process():
        # Save final checkpoint — unwrap FSDP if needed
        if use_distributed:
            from complexity_i64.training.distributed import save_fsdp_checkpoint
            save_fsdp_checkpoint(model, optimizer, scheduler, final_step, args.checkpoint_dir)
        else:
            model.save_pretrained(str(Path(args.checkpoint_dir) / "final"))
        logger.info("Training complete! Final step: %d", final_step)

    if use_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
