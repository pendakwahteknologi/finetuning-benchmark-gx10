"""Cross-run comparison: read multiple result dirs and produce summary table."""

import csv
import json
import os
from typing import Optional

from rich.table import Table

from ..utils.logging_utils import console


def compare_runs(results_dir: str, output_csv: Optional[str] = None):
    """Read all completed runs and produce a comparison table."""

    runs = []
    for entry in sorted(os.listdir(results_dir)):
        metrics_path = os.path.join(results_dir, entry, "benchmark_metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            # Try to load evaluation metrics too
            eval_path = os.path.join(
                results_dir, entry, "evaluation", "evaluation_metrics.json"
            )
            eval_metrics = {}
            if os.path.isfile(eval_path):
                with open(eval_path) as f:
                    eval_metrics = json.load(f)
            runs.append({"dir": entry, "metrics": m, "eval_metrics": eval_metrics})

    if not runs:
        console.print("[yellow]No completed runs found.[/]")
        return

    # Training comparison table
    table = Table(
        title="Benchmark Comparison: Training",
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("Machine", style="bold white", width=10)
    table.add_column("Mode", style="cyan", width=10)
    table.add_column("Time", width=12, justify="right")
    table.add_column("Avg Step", width=10, justify="right")
    table.add_column("Med Step", width=10, justify="right")
    table.add_column("Peak Mem", width=10, justify="right")
    table.add_column("Final Loss", width=10, justify="right")
    table.add_column("Tok/s", width=10, justify="right")
    table.add_column("Status", width=12)

    for run in runs:
        m = run["metrics"]
        status = m.get("status", "?")
        status_style = {
            "success": "green",
            "oom": "red",
            "failed": "red",
            "interrupted": "yellow",
        }.get(status, "white")

        wall = m.get("total_wall_clock_sec", 0)
        time_str = _format_time(wall)

        table.add_row(
            m.get("machine_label", "?"),
            m.get("mode", "?"),
            time_str,
            f"{m.get('avg_step_time', 0):.3f}s",
            f"{m.get('median_step_time', 0):.3f}s",
            f"{m.get('peak_gpu_memory_gb', 0):.1f} GB",
            f"{m.get('final_loss', 0):.4f}",
            f"{m.get('tokens_per_sec', 0):,.0f}",
            f"[{status_style}]{status}[/]",
        )

    console.print()
    console.print(table)

    # Evaluation comparison table (if available)
    eval_runs = [r for r in runs if r["eval_metrics"]]
    if eval_runs:
        eval_table = Table(
            title="Benchmark Comparison: Evaluation",
            border_style="green",
            show_lines=True,
        )
        eval_table.add_column("Machine", style="bold white", width=10)
        eval_table.add_column("Mode", style="cyan", width=10)
        eval_table.add_column("Base ROUGE", width=12, justify="right")
        eval_table.add_column("FT ROUGE", width=12, justify="right")
        eval_table.add_column("Base BLEU", width=12, justify="right")
        eval_table.add_column("FT BLEU", width=12, justify="right")
        eval_table.add_column("Questions", width=10, justify="right")

        for run in eval_runs:
            m = run["metrics"]
            em = run["eval_metrics"]
            b = em.get("baseline", {}).get("overall", {})
            ft = em.get("finetuned", {}).get("overall", {})

            eval_table.add_row(
                m.get("machine_label", "?"),
                m.get("mode", "?"),
                f"{b.get('rouge_l', {}).get('mean', 0):.4f}",
                f"{ft.get('rouge_l', {}).get('mean', 0):.4f}",
                f"{b.get('bleu', {}).get('mean', 0):.4f}",
                f"{ft.get('bleu', {}).get('mean', 0):.4f}",
                str(em.get("baseline", {}).get("total_samples", 0)),
            )

        console.print()
        console.print(eval_table)

    # Export CSV
    csv_path = output_csv or os.path.join(results_dir, "all_runs_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "machine_label", "mode", "status", "total_time_sec", "avg_step_time",
            "median_step_time", "peak_gpu_memory_gb", "final_loss", "tokens_per_sec",
            "samples_per_sec", "steps_completed", "run_dir",
        ])
        for run in runs:
            m = run["metrics"]
            writer.writerow([
                m.get("machine_label", ""),
                m.get("mode", ""),
                m.get("status", ""),
                m.get("total_wall_clock_sec", 0),
                m.get("avg_step_time", 0),
                m.get("median_step_time", 0),
                m.get("peak_gpu_memory_gb", 0),
                m.get("final_loss", 0),
                m.get("tokens_per_sec", 0),
                m.get("samples_per_sec", 0),
                m.get("steps_completed", 0),
                run["dir"],
            ])

    console.print(f"\n[dim]CSV exported to: {csv_path}[/]")


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
