"""
Complexity-I64 :: Pre-Training Script

Train an I64 model from scratch in float32.
After training, call model.quantize_all() for INT8 inference.

All parameters come from YAML config. No hardcoded values.

Usage:
    python train.py --config configs/pretrain/default.yaml
    python train.py --config configs/pretrain/default.yaml --resume ./checkpoints/last.pt

INL - 2025
"""

import os
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from complexity_i64.models.modeling import create_i64_model
from complexity_i64.data.datasets import (
    PreTokenizedDataset,
    StreamingTextDataset,
    collate_fn,
)
from complexity_i64.training.trainer import Trainer
from complexity_i64.training.utils import create_optimizer, create_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("complexity_i64.train")


def main():
    parser = argparse.ArgumentParser(description="Train Complexity-I64")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("Config: %s", args.config)
    logger.info("Device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name())

    # Tokenizer
    tokenizer_path = paths_cfg["tokenizer"]
    if os.path.exists(tokenizer_path):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        logger.warning("Tokenizer not found at %s, using GPT-2", tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    vocab_size = model_cfg.get("vocab_size", len(tokenizer))
    logger.info("Vocab size: %d", vocab_size)

    # Model
    model = create_i64_model(size=model_cfg["size"], vocab_size=vocab_size)
    model = model.to(device)
    logger.info("Parameters: %d", model.num_parameters())

    # Optimizer & scheduler
    optimizer = create_optimizer(
        model,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.1),
    )
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=train_cfg.get("scheduler", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 500),
        max_steps=train_cfg.get("max_steps", 100000),
        restart_period=train_cfg.get("restart_period", 50000),
        restart_mult=train_cfg.get("restart_mult", 2),
    )

    # Dataset
    source = data_cfg.get("source", "streaming")
    if source == "pretokenized":
        train_dataset = PreTokenizedDataset(
            data_cfg["data_dir"],
            max_length=data_cfg.get("max_length", 512),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=True,
        )
    else:
        train_dataset = StreamingTextDataset(
            dataset_name=data_cfg["dataset"],
            tokenizer=tokenizer,
            max_length=data_cfg.get("max_length", 512),
            text_field=data_cfg.get("text_field", "text"),
            token=args.token,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            collate_fn=collate_fn,
            num_workers=0,
        )

    # Trainer
    run_name = f"i64_{model_cfg['size']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=paths_cfg.get("checkpoint_dir", "./checkpoints"),
        log_dir=paths_cfg.get("tensorboard_dir", "./runs"),
        run_name=run_name,
        use_amp=train_cfg.get("use_amp", True),
        bf16=train_cfg.get("bf16", False),
        gradient_accumulation=train_cfg.get("gradient_accumulation", 1),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        log_interval=train_cfg.get("log_interval", 50),
        save_interval=train_cfg.get("save_interval", 5000),
    )

    # Resume
    start_step = 0
    if args.resume:
        start_step = trainer.resume(args.resume, lr=train_cfg["learning_rate"])

    # Train
    logger.info("Starting training — batch=%d, grad_accum=%d, max_steps=%d",
                train_cfg["batch_size"],
                train_cfg.get("gradient_accumulation", 1),
                train_cfg.get("max_steps", 100000))

    final_step = trainer.train(
        train_loader,
        max_steps=train_cfg.get("max_steps", 100000),
        start_step=start_step,
    )

    # Save final
    checkpoint_dir = paths_cfg.get("checkpoint_dir", "./checkpoints")
    model.save_pretrained(str(Path(checkpoint_dir) / "final"))
    logger.info("Training complete! Final step: %d", final_step)


if __name__ == "__main__":
    main()
