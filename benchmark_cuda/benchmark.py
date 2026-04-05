#!/usr/bin/env python3
"""CUDA Fine-Tuning Benchmark CLI.

Usage:
    python -m benchmark_cuda.benchmark run --machine-label gx10 --mode lora
    python -m benchmark_cuda.benchmark compare --results-dir ./results
    python -m benchmark_cuda.benchmark inspect --run ./results/gx10_lora_20260403_140000
"""

import os
import sys
import logging
import warnings
from typing import Optional

# Suppress all warnings for clean video output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import typer
import torch

from .utils.config import BenchmarkConfig
from .utils.system_info import save_system_info, get_gpu_info
from .utils.logging_utils import (
    setup_logging,
    print_banner,
    print_config_table,
    print_phase,
    print_final_summary,
    print_success_banner,
    print_failure_banner,
    console,
)
from .data.prepare import prepare_dataset
from .data.preset_questions import create_preset_questions
from .trainer import run_training

app = typer.Typer(
    name="benchmark",
    help="CUDA Fine-Tuning Benchmark: LoRA / QLoRA / Full FT",
    add_completion=False,
)


@app.command()
def run(
    machine_label: str = typer.Option("gx10", "--machine-label", "-m", help="Machine identifier"),
    mode: str = typer.Option("lora", "--mode", help="Training mode: lora, qlora, fullft"),
    model: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct", "--model", help="HuggingFace model name"
    ),
    max_steps: int = typer.Option(500, "--max-steps", help="Number of optimizer steps"),
    warmup_steps: int = typer.Option(3, "--warmup-steps", help="Steps excluded from timing"),
    seq_len: int = typer.Option(1024, "--seq-len", help="Maximum sequence length"),
    micro_batch_size: Optional[int] = typer.Option(None, "--micro-batch-size", help="Batch size per step"),
    grad_accum: Optional[int] = typer.Option(None, "--grad-accum", help="Gradient accumulation steps"),
    learning_rate: Optional[float] = typer.Option(None, "--learning-rate", "--lr", help="Learning rate"),
    dtype: str = typer.Option("bfloat16", "--dtype", help="Precision: bfloat16, float16, float32"),
    gradient_checkpointing: bool = typer.Option(False, "--gradient-checkpointing", help="Enable gradient checkpointing"),
    lora_r: int = typer.Option(16, "--lora-r", help="LoRA rank"),
    lora_alpha: int = typer.Option(32, "--lora-alpha", help="LoRA alpha"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    logging_steps: int = typer.Option(10, "--logging-steps", help="Log every N steps"),
    eval_steps: int = typer.Option(100, "--eval-steps", help="Validate every N steps"),
    skip_eval: bool = typer.Option(False, "--skip-eval", help="Skip post-training evaluation"),
    data_dir: str = typer.Option("./data/processed", "--data-dir", help="Processed data directory"),
    output_dir: str = typer.Option("./results", "--output-dir", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run 5 steps only, skip evaluation"),
):
    """Run a benchmark: train + evaluate."""

    # Validate mode
    if mode not in ("lora", "qlora", "fullft"):
        console.print(f"[red]Invalid mode: {mode}. Must be lora, qlora, or fullft[/]")
        raise typer.Exit(1)

    # Check CUDA
    if not torch.cuda.is_available():
        console.print("[red]CUDA is not available. This benchmark requires an NVIDIA GPU.[/]")
        raise typer.Exit(1)

    # Build config
    config = BenchmarkConfig(
        machine_label=machine_label,
        mode=mode,
        model_name=model,
        max_steps=5 if dry_run else max_steps,
        warmup_steps=min(warmup_steps, 1) if dry_run else warmup_steps,
        seq_len=seq_len,
        dtype=dtype,
        gradient_checkpointing=gradient_checkpointing,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        seed=seed,
        logging_steps=1 if dry_run else logging_steps,
        eval_steps=0 if dry_run else eval_steps,
        skip_eval=True if dry_run else skip_eval,
        data_dir=data_dir,
        output_dir=output_dir,
    )

    # Apply user overrides for batch size
    if micro_batch_size is not None:
        config.micro_batch_size = micro_batch_size
    if grad_accum is not None:
        config.gradient_accumulation_steps = grad_accum
    if learning_rate is not None:
        config.learning_rate = learning_rate

    # Record gradient checkpointing as fairness note
    if gradient_checkpointing:
        config.add_fairness_note("gradient_checkpointing enabled by user")

    # Create run directory
    os.makedirs(config.run_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(config.run_dir)

    # GPU info for banner
    gpu_info = get_gpu_info()
    gpu_name = ""
    gpu_mem_gb = 0
    if gpu_info.get("gpus"):
        gpu_name = gpu_info["gpus"][0]["name"]
        gpu_mem_gb = gpu_info["gpus"][0]["total_memory_gb"]

    # Print banner
    print_banner(
        machine_label=config.machine_label,
        mode=config.mode,
        model_name=config.model_name,
        dataset_name=config.dataset_name,
        max_steps=config.max_steps,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_mem_gb,
    )

    if dry_run:
        console.print("[yellow]DRY RUN: 5 steps, no evaluation[/]\n")

    # Print config
    print_config_table(config.to_dict())

    # Save system info
    print_phase("System Information")
    sys_info, gpu_info_saved = save_system_info(config.run_dir)
    console.print(f"  Hostname:  {sys_info.get('hostname', '?')}")
    console.print(f"  OS:        {sys_info.get('os', '?')}")
    console.print(f"  CPU:       {sys_info.get('cpu_model', '?')}")
    console.print(f"  RAM:       {sys_info.get('ram_gb', '?')} GB")
    console.print(f"  PyTorch:   {sys_info.get('pytorch_version', '?')}")
    console.print(f"  CUDA:      {sys_info.get('cuda_version', '?')}")
    if gpu_info_saved.get("c2c_mode"):
        console.print(f"  C2C Mode:  {gpu_info_saved['c2c_mode']}")
    console.print()

    # Prepare data
    print_phase("Dataset Preparation")
    manifest = prepare_dataset(data_dir=config.data_dir, seed=config.seed)
    console.print(f"  Dataset:     {config.dataset_name}")
    console.print(f"  Train:       {manifest['train_count']:,} samples")
    console.print(f"  Validation:  {manifest['val_count']:,} samples")
    console.print(f"  Test:        {manifest['test_count']:,} samples")
    console.print()

    # Prepare preset questions
    preset_path = create_preset_questions(
        data_dir=config.data_dir,
        eval_dir=os.path.dirname(config.eval_preset_path),
        seed=config.seed,
    )
    config.eval_preset_path = preset_path

    # Save config
    config.save()

    # Setup model based on mode
    try:
        if mode == "lora":
            from .modes.lora import setup_lora
            model_obj, tokenizer = setup_lora(config)
        elif mode == "qlora":
            from .modes.qlora import setup_qlora
            model_obj, tokenizer = setup_qlora(config)
        elif mode == "fullft":
            from .modes.fullft import setup_fullft
            model_obj, tokenizer = setup_fullft(config)
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "gated repo" in error_msg.lower() or "restricted" in error_msg.lower():
            print_failure_banner(
                mode,
                "HuggingFace authentication required",
                "Run: huggingface-cli login\n"
                "  You also need to accept the model license at:\n"
                f"  https://huggingface.co/{config.model_name}",
            )
            config.status = "failed"
            config.failure_reason = "HuggingFace authentication required"
        elif "out of memory" in error_msg.lower():
            print_failure_banner(mode, "OOM during model loading", "Try --gradient-checkpointing or smaller --micro-batch-size")
            config.status = "oom"
            config.failure_reason = error_msg[:500]
        else:
            print_failure_banner(mode, error_msg[:300])
            config.status = "failed"
            config.failure_reason = error_msg[:500]
        config.save()
        raise typer.Exit(1)

    # Run training
    metrics = run_training(model_obj, tokenizer, config)

    # Save metrics
    metrics.save(config.run_dir)

    # Print final training summary
    print_final_summary(metrics.to_dict())

    # Update config with final status
    config.status = metrics.status
    config.failure_reason = metrics.failure_reason
    config.save()

    # Run evaluation (unless skipped or failed)
    if not config.skip_eval and metrics.status == "success":
        try:
            from .evaluation.evaluate import run_full_evaluation
            run_full_evaluation(model_obj, tokenizer, config, config.run_dir)
        except Exception as e:
            console.print(f"[yellow]Evaluation failed: {e}[/]")
            logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final banner
    if metrics.status == "success":
        print_success_banner(config.run_dir)
    else:
        console.print(f"\n[dim]Partial results saved to: {config.run_dir}[/]")

    logger.info(f"Benchmark complete: status={metrics.status} run_dir={config.run_dir}")


@app.command()
def compare(
    results_dir: str = typer.Option("./results", "--results-dir", "-d", help="Results directory"),
    output_csv: Optional[str] = typer.Option(None, "--output-csv", help="Output CSV path"),
):
    """Compare multiple benchmark runs."""
    from .commands.compare import compare_runs
    compare_runs(results_dir, output_csv)


@app.command()
def inspect(
    run_dir: str = typer.Option(..., "--run", "-r", help="Path to a single run directory"),
):
    """Inspect a single benchmark run."""
    from .commands.inspect import inspect_run
    inspect_run(run_dir)


@app.command("cross-compare")
def cross_compare(
    results_dir: str = typer.Option("./results", "--results-dir", "-d", help="Results directory"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory for reports"),
):
    """Cross-mode comparison: Base vs LoRA vs QLoRA vs Full Fine-Tune."""
    from .evaluation.cross_compare import run_cross_compare
    run_cross_compare(results_dir, output_dir)


@app.command()
def prepare(
    data_dir: str = typer.Option("./data/processed", "--data-dir", help="Output data directory"),
    eval_dir: str = typer.Option("./data/eval", "--eval-dir", help="Evaluation data directory"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    force: bool = typer.Option(False, "--force", help="Re-prepare even if data exists"),
):
    """Prepare dataset and preset questions (without training)."""
    print_phase("Dataset Preparation")
    manifest = prepare_dataset(data_dir=data_dir, seed=seed, force=force)
    console.print(f"  Train:       {manifest['train_count']:,}")
    console.print(f"  Validation:  {manifest['val_count']:,}")
    console.print(f"  Test:        {manifest['test_count']:,}")
    console.print()

    print_phase("Preset Questions")
    path = create_preset_questions(data_dir=data_dir, eval_dir=eval_dir, seed=seed, force=force)
    console.print(f"  Saved to: {path}")
    console.print()


if __name__ == "__main__":
    app()
