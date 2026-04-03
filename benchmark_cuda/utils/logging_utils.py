"""Rich-based dual logging: terminal + file."""

import logging
import os
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

THEME = Theme({
    "info": "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "header": "bold magenta",
    "metric": "bold white",
})

console = Console(theme=THEME, width=100)


def setup_logging(run_dir: str, log_file: str = "train.log") -> logging.Logger:
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, log_file)

    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — everything
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    # Rich console handler
    rh = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rh.setLevel(logging.INFO)
    logger.addHandler(rh)

    return logger


def print_banner(
    machine_label: str,
    mode: str,
    model_name: str,
    dataset_name: str,
    max_steps: int,
    gpu_name: str = "",
    gpu_memory_gb: float = 0,
):
    mode_display = {"lora": "LoRA", "qlora": "QLoRA", "fullft": "Full Fine-Tune"}.get(
        mode, mode
    )
    lines = [
        f"[bold white]Machine:[/]      {machine_label}",
        f"[bold white]GPU:[/]          {gpu_name} ({gpu_memory_gb:.0f} GB)" if gpu_name else "",
        f"[bold white]Mode:[/]         {mode_display}",
        f"[bold white]Model:[/]        {model_name}",
        f"[bold white]Dataset:[/]      {dataset_name}",
        f"[bold white]Steps:[/]        {max_steps}",
    ]
    content = "\n".join(l for l in lines if l)
    console.print()
    console.print(Panel(
        content,
        title="[bold cyan]CUDA Fine-Tuning Benchmark[/]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


def print_config_table(config_dict: dict, title: str = "Run Configuration"):
    table = Table(title=title, show_header=True, border_style="dim")
    table.add_column("Parameter", style="cyan", width=30)
    table.add_column("Value", style="white", width=50)

    display_keys = [
        "mode", "model_name", "max_steps", "seq_len", "micro_batch_size",
        "gradient_accumulation_steps", "effective_batch_size", "learning_rate",
        "dtype", "attn_implementation", "seed", "gradient_checkpointing",
    ]
    if config_dict.get("mode") in ("lora", "qlora"):
        display_keys += ["lora_r", "lora_alpha", "lora_dropout"]
    if config_dict.get("mode") == "qlora":
        display_keys += ["qlora_bits", "qlora_quant_type"]

    for key in display_keys:
        if key in config_dict:
            table.add_row(key, str(config_dict[key]))

    console.print(table)
    console.print()


def print_phase(phase: str, detail: str = ""):
    text = f"[header]{phase}[/]"
    if detail:
        text += f" [dim]{detail}[/]"
    console.print()
    console.rule(text)
    console.print()


def print_step_progress(
    step: int,
    max_steps: int,
    loss: float,
    lr: float,
    step_time: float,
    elapsed: float,
    eta: float,
    gpu_mem_gb: float,
    gpu_total_gb: float,
    tokens_per_sec: float = 0,
    samples_per_sec: float = 0,
    is_warmup: bool = False,
):
    pct = step / max_steps * 100
    elapsed_str = _format_time(elapsed)
    eta_str = _format_time(eta)

    warmup_tag = " [dim](warmup)[/]" if is_warmup else ""

    line = (
        f"  Step [bold]{step:>5}[/]/{max_steps} "
        f"[{'yellow' if pct < 50 else 'green'}]({pct:5.1f}%)[/]{warmup_tag}  "
        f"Loss: [metric]{loss:.4f}[/]  "
        f"LR: {lr:.2e}  "
        f"Step: {step_time:.2f}s  "
        f"Elapsed: {elapsed_str}  "
        f"ETA: {eta_str}  "
        f"GPU: {gpu_mem_gb:.1f}/{gpu_total_gb:.0f}GB"
    )
    if tokens_per_sec > 0:
        line += f"  Tok/s: {tokens_per_sec:,.0f}"
    if samples_per_sec > 0:
        line += f"  Samp/s: {samples_per_sec:.1f}"

    console.print(line)


def print_eval_progress(step: int, val_loss: float):
    console.print(
        f"  [cyan]Eval @ step {step}[/]  Val Loss: [metric]{val_loss:.4f}[/]"
    )


def print_final_summary(metrics_dict: dict):
    status = metrics_dict.get("status", "unknown")
    status_style = {
        "success": "bold green",
        "oom": "bold red",
        "failed": "bold red",
        "interrupted": "bold yellow",
        "fallback_used": "bold yellow",
    }.get(status, "white")

    table = Table(
        title="Final Benchmark Results",
        show_header=False,
        border_style="green" if status == "success" else "red",
        padding=(0, 2),
    )
    table.add_column("Label", style="cyan", width=25)
    table.add_column("Value", style="white", width=45)

    rows = [
        ("Machine", metrics_dict.get("machine_label", "")),
        ("Mode", metrics_dict.get("mode", "")),
        ("Status", f"[{status_style}]{status}[/]"),
        ("Steps Completed", f"{metrics_dict.get('steps_completed', 0)} / {metrics_dict.get('max_steps', 0)}"),
        ("Total Time", _format_time(metrics_dict.get("total_wall_clock_sec", 0))),
        ("Avg Step Time", f"{metrics_dict.get('avg_step_time', 0):.4f}s"),
        ("Median Step Time", f"{metrics_dict.get('median_step_time', 0):.4f}s"),
        ("P95 Step Time", f"{metrics_dict.get('p95_step_time', 0):.4f}s"),
        ("Peak GPU Memory", f"{metrics_dict.get('peak_gpu_memory_gb', 0):.2f} GB"),
        ("Final Loss", f"{metrics_dict.get('final_loss', 0):.4f}"),
        ("Tokens/sec", f"{metrics_dict.get('tokens_per_sec', 0):,.1f}"),
        ("Samples/sec", f"{metrics_dict.get('samples_per_sec', 0):.2f}"),
    ]

    if metrics_dict.get("failure_reason"):
        rows.append(("Failure Reason", f"[red]{metrics_dict['failure_reason']}[/]"))
    if metrics_dict.get("fallbacks_used"):
        rows.append(("Fallbacks", ", ".join(metrics_dict["fallbacks_used"])))

    for label, value in rows:
        table.add_row(label, str(value))

    console.print()
    console.print(table)
    console.print()


def print_failure_banner(mode: str, reason: str, suggestion: str = ""):
    lines = [
        f"[bold red]Benchmark FAILED[/]",
        f"[white]Mode:[/] {mode}",
        f"[white]Reason:[/] {reason}",
    ]
    if suggestion:
        lines.append(f"[white]Suggestion:[/] {suggestion}")

    console.print()
    console.print(Panel(
        "\n".join(lines),
        title="[bold red]FAILURE[/]",
        border_style="red",
        padding=(1, 2),
    ))
    console.print()


def print_success_banner(run_dir: str):
    console.print()
    console.print(Panel(
        f"[bold green]Benchmark completed successfully![/]\n\n"
        f"Results saved to: [white]{run_dir}[/]",
        title="[bold green]SUCCESS[/]",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()


def _format_time(seconds: float) -> str:
    if seconds <= 0:
        return "--:--"
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"
