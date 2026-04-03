"""Shared training loop for all benchmark modes."""

import json
import os
import time
import logging

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler

from .utils.config import BenchmarkConfig
from .utils.metrics import BenchmarkMetrics, StepRecord
from .utils.gpu_monitor import GPUMonitor
from .utils.logging_utils import (
    print_phase,
    print_step_progress,
    print_eval_progress,
    print_final_summary,
    print_failure_banner,
    print_success_banner,
    console,
)
from .data.prompt_format import format_train_prompt

logger = logging.getLogger("benchmark")


class TextDataset(Dataset):
    """Tokenized dataset for training."""

    def __init__(self, data_path: str, tokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples = []

        with open(data_path) as f:
            for line in f:
                rec = json.loads(line)
                text = format_train_prompt(rec["instruction"], rec["input"], rec["output"])
                self.examples.append(text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def run_training(
    model: torch.nn.Module,
    tokenizer,
    config: BenchmarkConfig,
) -> BenchmarkMetrics:
    """Execute the training loop and return metrics."""

    metrics = BenchmarkMetrics(
        machine_label=config.machine_label,
        mode=config.mode,
        model_name=config.model_name,
        dataset_name=config.dataset_name,
        seed=config.seed,
        seq_len=config.seq_len,
        micro_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        effective_batch_size=config.effective_batch_size,
        max_steps=config.max_steps,
        fairness_notes=config.fairness_notes.copy(),
        _warmup_steps=config.warmup_steps,
    )

    # Get GPU total memory for display
    gpu_total_gb = 0
    if torch.cuda.is_available():
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # --- Data setup ---
    print_phase("Preparing Data")

    train_path = os.path.join(config.data_dir, "train.jsonl")
    val_path = os.path.join(config.data_dir, "val.jsonl")

    train_dataset = TextDataset(train_path, tokenizer, config.seq_len)
    console.print(f"  Train samples: {len(train_dataset):,}")

    val_dataset = None
    if os.path.exists(val_path):
        val_dataset = TextDataset(val_path, tokenizer, config.seq_len)
        console.print(f"  Val samples:   {len(val_dataset):,}")

    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        num_workers=config.dataloader_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    # --- Optimizer setup ---
    print_phase("Setting Up Optimizer")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    warmup_steps_sched = int(config.max_steps * config.warmup_ratio)

    scheduler = get_scheduler(
        config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps_sched,
        num_training_steps=config.max_steps,
    )

    console.print(f"  Optimizer:        AdamW")
    console.print(f"  LR schedule:      {config.lr_scheduler_type}")
    console.print(f"  Warmup steps:     {warmup_steps_sched}")
    console.print(f"  Max steps:        {config.max_steps}")
    console.print(f"  Timing warmup:    {config.warmup_steps} steps (excluded from metrics)")
    console.print()

    # --- GPU monitor ---
    gpu_monitor = GPUMonitor()
    gpu_monitor.start()

    # --- Training loop ---
    print_phase("Training", f"Steps 1-{config.max_steps}")

    model.train()
    metrics.start_timer()
    metrics.status = "running"

    global_step = 0
    accum_loss = 0.0
    accum_tokens = 0
    micro_step = 0
    data_iter = iter(train_loader)
    step_start = time.time()

    try:
        while global_step < config.max_steps:
            # Get batch, cycle through data if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Move to device
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

            accum_loss += outputs.loss.item()
            accum_tokens += (labels != -100).sum().item()
            micro_step += 1

            # Optimizer step
            if micro_step % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                step_time = time.time() - step_start
                avg_loss = accum_loss / config.gradient_accumulation_steps

                is_warmup = global_step <= config.warmup_steps

                # GPU memory
                gpu_mem_mb = 0
                if torch.cuda.is_available():
                    gpu_mem_mb = torch.cuda.max_memory_allocated(0) / 1024**2

                # Record step
                record = StepRecord(
                    step=global_step,
                    loss=avg_loss,
                    learning_rate=scheduler.get_last_lr()[0],
                    step_time_sec=step_time,
                    gpu_memory_mb=gpu_mem_mb,
                    tokens_processed=accum_tokens,
                    is_warmup=is_warmup,
                )
                metrics.record_step(record)

                # Log progress
                if global_step % config.logging_steps == 0 or global_step == 1:
                    elapsed = time.time() - metrics._start_time
                    # Use non-warmup steps for ETA
                    timed_steps = [
                        r for r in metrics.step_records if not r.is_warmup
                    ]
                    if timed_steps:
                        avg_t = sum(r.step_time_sec for r in timed_steps) / len(timed_steps)
                        remaining = config.max_steps - global_step
                        eta = remaining * avg_t
                        total_timed_sec = sum(r.step_time_sec for r in timed_steps)
                        tps = sum(r.tokens_processed for r in timed_steps) / total_timed_sec
                        sps = len(timed_steps) * config.micro_batch_size / total_timed_sec
                    else:
                        avg_t = step_time
                        eta = (config.max_steps - global_step) * step_time
                        tps = 0
                        sps = 0

                    print_step_progress(
                        step=global_step,
                        max_steps=config.max_steps,
                        loss=avg_loss,
                        lr=scheduler.get_last_lr()[0],
                        step_time=step_time,
                        elapsed=elapsed,
                        eta=eta,
                        gpu_mem_gb=gpu_mem_mb / 1024,
                        gpu_total_gb=gpu_total_gb,
                        tokens_per_sec=tps,
                        samples_per_sec=sps,
                        is_warmup=is_warmup,
                    )
                    logger.debug(
                        f"Step {global_step}/{config.max_steps} "
                        f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                        f"step_time={step_time:.2f}s gpu_mem={gpu_mem_mb:.0f}MB"
                    )

                # Validation
                if (
                    val_dataset
                    and config.eval_steps > 0
                    and global_step % config.eval_steps == 0
                ):
                    val_loss = _run_validation(model, val_dataset, config)
                    print_eval_progress(global_step, val_loss)
                    logger.info(
                        f"Validation @ step {global_step}: val_loss={val_loss:.4f}"
                    )
                    model.train()

                # Reset accumulators
                accum_loss = 0.0
                accum_tokens = 0
                step_start = time.time()

        # Training complete
        metrics.stop_timer()
        metrics.status = "success"
        metrics.compute_aggregates()

    except RuntimeError as e:
        metrics.stop_timer()
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            metrics.status = "oom"
            metrics.failure_reason = str(e)[:500]
            metrics.compute_aggregates()
            torch.cuda.empty_cache()
            print_failure_banner(
                config.mode,
                "CUDA Out of Memory",
                f"Try reducing --micro-batch-size (current: {config.micro_batch_size}) "
                f"or enabling --gradient-checkpointing",
            )
            logger.error(f"OOM at step {global_step}: {e}")
        else:
            metrics.status = "failed"
            metrics.failure_reason = str(e)[:500]
            metrics.compute_aggregates()
            print_failure_banner(config.mode, str(e)[:200])
            logger.error(f"Training failed at step {global_step}: {e}")

    except KeyboardInterrupt:
        metrics.stop_timer()
        metrics.status = "interrupted"
        metrics.failure_reason = f"User interrupted at step {global_step}"
        metrics.compute_aggregates()
        console.print("\n[yellow]Training interrupted by user[/]")
        logger.warning(f"Training interrupted at step {global_step}")

    finally:
        gpu_monitor.stop()

    return metrics


def _run_validation(
    model: torch.nn.Module,
    val_dataset: "TextDataset",
    config: BenchmarkConfig,
) -> float:
    """Run quick validation loss computation."""
    model.train(False)

    total_loss = 0.0
    count = 0
    max_batches = 20  # Cap for speed

    loader = DataLoader(
        val_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            count += 1

    return total_loss / max(count, 1)
