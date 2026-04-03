"""Inspect a single benchmark run."""

import json
import os

from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from ..utils.logging_utils import console


def inspect_run(run_dir: str):
    """Pretty-print details of a single benchmark run."""

    if not os.path.isdir(run_dir):
        console.print(f"[red]Run directory not found: {run_dir}[/]")
        return

    # Config
    config_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
        console.print(Panel(
            Syntax(json.dumps(config, indent=2), "json", theme="monokai"),
            title="[bold cyan]Run Configuration[/]",
            border_style="cyan",
        ))

    # Metrics
    metrics_path = os.path.join(run_dir, "benchmark_metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

        table = Table(title="Benchmark Metrics", border_style="green")
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="white", width=40)

        display_keys = [
            "machine_label", "mode", "status", "steps_completed", "max_steps",
            "total_wall_clock_sec", "avg_step_time", "median_step_time",
            "p95_step_time", "peak_gpu_memory_gb", "final_loss",
            "tokens_per_sec", "samples_per_sec",
        ]
        for k in display_keys:
            if k in metrics:
                table.add_row(k, str(metrics[k]))

        if metrics.get("failure_reason"):
            table.add_row("failure_reason", f"[red]{metrics['failure_reason']}[/]")
        if metrics.get("fallbacks_used"):
            table.add_row("fallbacks_used", str(metrics["fallbacks_used"]))
        if metrics.get("fairness_notes"):
            for note in metrics["fairness_notes"]:
                table.add_row("fairness_note", note)

        console.print()
        console.print(table)

    # System info
    sys_path = os.path.join(run_dir, "system_info.json")
    if os.path.isfile(sys_path):
        with open(sys_path) as f:
            sys_info = json.load(f)
        console.print()
        console.print(Panel(
            Syntax(json.dumps(sys_info, indent=2), "json", theme="monokai"),
            title="[bold cyan]System Info[/]",
            border_style="dim",
        ))

    # GPU info
    gpu_path = os.path.join(run_dir, "gpu_info.json")
    if os.path.isfile(gpu_path):
        with open(gpu_path) as f:
            gpu_info = json.load(f)
        console.print()
        console.print(Panel(
            Syntax(json.dumps(gpu_info, indent=2), "json", theme="monokai"),
            title="[bold cyan]GPU Info[/]",
            border_style="dim",
        ))

    # Evaluation summary
    eval_metrics_path = os.path.join(run_dir, "evaluation", "evaluation_metrics.json")
    if os.path.isfile(eval_metrics_path):
        with open(eval_metrics_path) as f:
            eval_data = json.load(f)

        console.print()
        console.print(Panel(
            Syntax(json.dumps(eval_data, indent=2), "json", theme="monokai"),
            title="[bold cyan]Evaluation Metrics[/]",
            border_style="green",
        ))

    # List all files
    console.print()
    console.print("[bold]Run artifacts:[/]")
    for root, dirs, files in os.walk(run_dir):
        rel = os.path.relpath(root, run_dir)
        for f in sorted(files):
            path = os.path.join(rel, f) if rel != "." else f
            size = os.path.getsize(os.path.join(root, f))
            console.print(f"  {path} ({_human_size(size)})")


def _human_size(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
