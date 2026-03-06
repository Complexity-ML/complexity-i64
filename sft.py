"""
Complexity-I64 :: Conversational SFT

Fine-tune with multi-turn conversations + optional LoRA.
All parameters from YAML config.

Usage:
    python sft.py --config configs/sft/chat.yaml
    python sft.py --config configs/sft/chat_lora.yaml

INL - 2025
"""

import os
import yaml
import logging
import functools
import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from complexity_i64.models.modeling import I64Model
from complexity_i64.data.datasets import (
    ConversationalDataset,
    CHAT_TEMPLATES,
    collate_sft,
)
from complexity_i64.training.trainer import Trainer
from complexity_i64.training.utils import (
    create_optimizer,
    create_scheduler,
    cleanup_old_checkpoints,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("complexity_i64.sft")


def main():
    parser = argparse.ArgumentParser(description="Complexity-I64 Conversational SFT")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    lora_cfg = cfg.get("lora", {})
    data_cfg = cfg["data"]

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("Config: %s", args.config)
    logger.info("Device: %s", device)

    # Tokenizer
    tokenizer_path = model_cfg.get("tokenizer", "./tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        logger.warning("Tokenizer not found at %s, using GPT-2", tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    logger.info("Vocab size: %d", len(tokenizer))

    # Model
    checkpoint = model_cfg["checkpoint"]
    logger.info("Loading model from %s", checkpoint)
    model = I64Model.from_pretrained(checkpoint, device=str(device))
    logger.info("Parameters: %d", model.num_parameters())

    # LoRA
    use_lora = lora_cfg.get("enabled", False)
    if use_lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(
                r=lora_cfg["rank"],
                lora_alpha=lora_cfg["alpha"],
                lora_dropout=lora_cfg.get("dropout", 0.05),
                target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, peft_config)
            logger.info("LoRA enabled: rank=%d, alpha=%d", lora_cfg["rank"], lora_cfg["alpha"])
        except ImportError:
            logger.warning("peft not installed, training without LoRA")
            use_lora = False

    model = model.to(device)

    # Dataset
    template_name = data_cfg.get("template", "default")
    chat_template = CHAT_TEMPLATES.get(template_name, CHAT_TEMPLATES["default"])

    datasets_config = data_cfg.get("datasets", [])
    dataset = ConversationalDataset.from_multiple_datasets(
        datasets_config=datasets_config,
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_length=train_cfg.get("max_length", 2048),
        max_samples=data_cfg.get("max_samples", None),
        token=args.token,
        mask_user=True,
    )

    pad_id = tokenizer.pad_token_id or 0
    train_loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=functools.partial(collate_sft, pad_token_id=pad_id),
        num_workers=2,
        pin_memory=True,
    )

    # Optimizer & scheduler
    lr = train_cfg["learning_rate"]
    epochs = train_cfg["epochs"]
    grad_accum = train_cfg.get("gradient_accumulation", 1)
    optimizer = create_optimizer(model, lr=lr)

    total_steps = len(dataset) // (train_cfg["batch_size"] * grad_accum) * epochs
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.1))

    scheduler = create_scheduler(
        optimizer,
        scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_steps=total_steps,
    )

    # Trainer
    output_dir = model_cfg.get("output", "./checkpoints-sft")
    run_name = f"i64_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=output_dir,
        run_name=run_name,
        use_amp=True,
        bf16=train_cfg.get("bf16", True),
        gradient_accumulation=grad_accum,
        log_interval=10,
        save_interval=total_steps + 1,
    )

    logger.info("Starting SFT — epochs=%d, batch=%d, grad_accum=%d, lr=%.2e",
                epochs, train_cfg["batch_size"], grad_accum, lr)

    trainer.train_epochs(train_loader, epochs=epochs)

    # Save
    if use_lora:
        model.save_pretrained(str(Path(output_dir) / "lora_final"))
        logger.info("LoRA adapter saved to %s/lora_final", output_dir)
    else:
        model.save_pretrained(str(Path(output_dir) / "final"))

    cleanup_old_checkpoints(output_dir)
    logger.info("SFT complete!")


if __name__ == "__main__":
    main()
