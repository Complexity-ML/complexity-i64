"""
Complexity-I64 :: Trainer

Handles both pre-training and SFT training loops.
Float32 training (or mixed precision BF16/FP16).

INL - 2025
"""

import math
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False


class Trainer:
    """
    Training loop for I64Model.

    Supports:
    - Mixed precision (FP16/BF16)
    - Gradient accumulation
    - Checkpoint save/resume
    - TensorBoard logging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./runs",
        run_name: str = "i64_train",
        use_amp: bool = True,
        bf16: bool = False,
        gradient_accumulation: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 50,
        save_interval: int = 5000,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_accumulation = gradient_accumulation
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision
        self.use_amp = use_amp and AMP_AVAILABLE and torch.cuda.is_available()
        if self.use_amp:
            if bf16 and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                self.scaler = None
                logger.info("Using BF16 mixed precision")
            else:
                self.amp_dtype = torch.float16
                self.scaler = GradScaler('cuda')
                logger.info("Using FP16 mixed precision")
        else:
            self.amp_dtype = torch.float32
            self.scaler = None
            logger.info("Using FP32 (no mixed precision)")

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(Path(log_dir) / run_name))
        self.global_step = 0

    def train(
        self,
        train_loader: DataLoader,
        max_steps: int = 100000,
        start_step: int = 0,
        eval_loader: Optional[DataLoader] = None,
        eval_interval: int = 1000,
    ) -> int:
        """Pre-training loop (step-based)."""
        self.model.train()
        self.global_step = start_step
        total_loss = 0.0
        start_time = time.time()
        pbar = tqdm(total=max_steps, initial=start_step, desc="Training")

        self.optimizer.zero_grad()
        accum_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if self.global_step >= max_steps:
                break

            loss = self._forward_backward(batch)
            if loss is None:
                continue

            accum_loss += loss

            if (batch_idx + 1) % self.gradient_accumulation == 0:
                self._optimizer_step()
                total_loss += accum_loss
                accum_loss = 0.0
                self.global_step += 1
                pbar.update(1)

                if self.global_step % self.log_interval == 0:
                    self._log_metrics(total_loss, start_time, batch)
                    total_loss = 0.0

                if self.global_step % self.save_interval == 0:
                    self._save_checkpoint()

                if eval_loader and self.global_step % eval_interval == 0:
                    self._evaluate(eval_loader)
                    self.model.train()

        pbar.close()
        return self.global_step

    def train_epochs(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        eval_loader: Optional[DataLoader] = None,
    ) -> int:
        """SFT training loop (epoch-based)."""
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(pbar):
                if batch is None:
                    continue

                loss = self._forward_backward(batch)
                if loss is None:
                    continue

                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    self._optimizer_step()
                    self.global_step += 1

                    total_loss += loss
                    num_batches += 1

                    if self.global_step % 10 == 0:
                        current_loss = loss
                        ppl = math.exp(min(current_loss, 20))
                        lr = self.scheduler.get_last_lr()[0]
                        pbar.set_postfix(loss=f"{current_loss:.4f}", ppl=f"{ppl:.2f}", lr=f"{lr:.2e}")
                        self.writer.add_scalar("train/loss", current_loss, self.global_step)
                        self.writer.add_scalar("train/perplexity", ppl, self.global_step)
                        self.writer.add_scalar("train/lr", lr, self.global_step)

            # End of epoch
            avg_loss = total_loss / max(num_batches, 1)
            ppl = math.exp(min(avg_loss, 20))
            logger.info("Epoch %d — loss: %.4f, ppl: %.2f", epoch + 1, avg_loss, ppl)
            self.writer.add_scalar("epoch/loss", avg_loss, epoch)

            if eval_loader:
                self._evaluate(eval_loader)

            self._save_checkpoint(tag=f"epoch{epoch + 1}")

        return self.global_step

    def _forward_backward(self, batch) -> Optional[float]:
        """Forward + backward pass. Returns scalar loss."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        try:
            if self.use_amp:
                with autocast('cuda', dtype=self.amp_dtype):
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / self.gradient_accumulation
            else:
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / self.gradient_accumulation
        except Exception as e:
            logger.error("Forward error: %s", e)
            return None

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/Inf loss, skipping batch")
            self.optimizer.zero_grad()
            return None

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item()

    def _optimizer_step(self):
        """Gradient clip + optimizer step."""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

    def _log_metrics(self, total_loss: float, start_time: float, batch: dict):
        """Log to TensorBoard."""
        avg_loss = total_loss / self.log_interval
        elapsed = time.time() - start_time
        ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')

        batch_size = batch["input_ids"].shape[0]
        seq_len = batch["input_ids"].shape[1]
        tokens_per_sec = (self.global_step * self.gradient_accumulation * batch_size * seq_len) / max(elapsed, 1)

        self.writer.add_scalar("train/loss", avg_loss, self.global_step)
        self.writer.add_scalar("train/perplexity", ppl, self.global_step)
        self.writer.add_scalar("train/learning_rate", self.scheduler.get_last_lr()[0], self.global_step)
        self.writer.add_scalar("train/tokens_per_sec", tokens_per_sec, self.global_step)

        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1e9
            self.writer.add_scalar("train/memory_gb", mem, self.global_step)

    def _evaluate(self, eval_loader: DataLoader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
                if batch is None:
                    continue
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                if self.use_amp:
                    with autocast('cuda', dtype=self.amp_dtype):
                        outputs = self.model(input_ids, labels=labels)
                else:
                    outputs = self.model(input_ids, labels=labels)

                if not torch.isnan(outputs.loss):
                    total_loss += outputs.loss.item()
                    num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        ppl = math.exp(min(avg_loss, 20))
        logger.info("Eval — loss: %.4f, ppl: %.2f", avg_loss, ppl)
        self.writer.add_scalar("eval/loss", avg_loss, self.global_step)
        self.writer.add_scalar("eval/perplexity", ppl, self.global_step)

    def _save_checkpoint(self, tag: Optional[str] = None):
        """Save checkpoint."""
        name = tag or f"step_{self.global_step}"
        path = self.checkpoint_dir / f"{name}.pt"

        save_dict = {
            "step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        if self.scaler:
            save_dict["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(save_dict, path)
        torch.save(save_dict, self.checkpoint_dir / "last.pt")
        logger.info("Saved checkpoint: %s", path)

    def resume(self, checkpoint_path: str, lr: Optional[float] = None):
        """Resume training from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        missing, unexpected = self.model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if missing:
            logger.info("New parameters: %d", len(missing))
        if unexpected:
            logger.warning("Unexpected keys: %d", len(unexpected))

        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            logger.warning("Optimizer structure changed, using fresh optimizer")

        self.global_step = checkpoint["step"]

        if lr is not None:
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
                pg['initial_lr'] = lr

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info("Resumed at step %d", self.global_step)
        return self.global_step
