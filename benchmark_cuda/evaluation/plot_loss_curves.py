"""Plot training/validation loss curves, GPU memory, and step time across modes.

Reads per-step metrics from benchmark_metrics.csv and validation loss from
train.log for each fine-tuning mode, then generates publication-quality
dark-themed plots saved as PNG (300 dpi) and SVG.
"""

import csv
import os
import re
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from ..utils.logging_utils import console

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODE_LABELS = {
    "lora": "LoRA",
    "qlora": "QLoRA",
    "fullft": "Full Fine-Tune",
}

MODE_ORDER = ["lora", "qlora", "fullft"]

MODE_COLORS = {
    "lora": "#00d4ff",    # cyan
    "qlora": "#ff79c6",   # magenta
    "fullft": "#f1fa8c",  # yellow
}

BG_COLOR = "#0d1117"
FACE_COLOR = "#0d1117"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"

VAL_LOSS_PATTERN = re.compile(
    r"Validation @ step (\d+): val_loss=([\d.]+)"
)

# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def find_best_runs(results_dir: str) -> dict[str, str]:
    """Find the latest successful run directory for each mode.

    Returns a dict mapping mode key -> run directory path.
    Uses the same logic as cross_compare.py: looks for summary.txt
    containing ``Status:               success`` and picks the last
    (i.e. most recent by sort order) candidate per mode.
    """
    best: dict[str, str] = {}
    if not os.path.isdir(results_dir):
        return best

    for mode in MODE_ORDER:
        candidates = []
        for entry in sorted(os.listdir(results_dir)):
            if f"_{mode}_" not in entry:
                continue
            run_dir = os.path.join(results_dir, entry)
            summary = os.path.join(run_dir, "summary.txt")
            if os.path.isfile(summary):
                with open(summary) as f:
                    if "Status:               success" in f.read():
                        candidates.append(run_dir)
        if candidates:
            best[mode] = candidates[-1]

    return best


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_training_data(run_dir: str) -> dict:
    """Load per-step training metrics and validation loss for a run.

    Returns a dict with keys:
        steps, loss, learning_rate, step_time_sec, gpu_memory_mb,
        tokens_processed, is_warmup          -- lists from CSV
        val_steps, val_loss                   -- lists from train.log
    """
    data: dict = {
        "steps": [],
        "loss": [],
        "learning_rate": [],
        "step_time_sec": [],
        "gpu_memory_mb": [],
        "tokens_processed": [],
        "is_warmup": [],
        "val_steps": [],
        "val_loss": [],
    }

    # --- CSV metrics ---
    csv_path = os.path.join(run_dir, "benchmark_metrics.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data["steps"].append(int(row["step"]))
                data["loss"].append(float(row["loss"]))
                data["learning_rate"].append(float(row["learning_rate"]))
                data["step_time_sec"].append(float(row["step_time_sec"]))
                data["gpu_memory_mb"].append(float(row["gpu_memory_mb"]))
                data["tokens_processed"].append(int(row["tokens_processed"]))
                data["is_warmup"].append(row["is_warmup"].strip() == "True")

    # --- Validation loss from train.log ---
    log_path = os.path.join(run_dir, "train.log")
    if os.path.isfile(log_path):
        with open(log_path) as f:
            for line in f:
                m = VAL_LOSS_PATTERN.search(line)
                if m:
                    data["val_steps"].append(int(m.group(1)))
                    data["val_loss"].append(float(m.group(2)))

    return data


# ---------------------------------------------------------------------------
# Smoothing helper
# ---------------------------------------------------------------------------


def _smooth(values: list[float], window: int = 20) -> np.ndarray:
    """Simple rolling average with same-length output (edge-padded)."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    # Use 'valid' convolution then pad edges
    smoothed = np.convolve(arr, kernel, mode="same")
    # Fix edge effects by using partial windows
    for i in range(window // 2):
        smoothed[i] = np.mean(arr[: i + window // 2 + 1])
    for i in range(len(arr) - window // 2, len(arr)):
        smoothed[i] = np.mean(arr[i - window // 2:])
    return smoothed


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _apply_dark_theme(ax: plt.Axes) -> None:
    """Apply the project's dark theme to an axes object."""
    ax.set_facecolor(FACE_COLOR)
    ax.tick_params(colors=TEXT_COLOR, which="both")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)


def _save(fig: plt.Figure, output_dir: str, name: str) -> None:
    """Save a figure as both PNG (300 dpi) and SVG."""
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{name}.png")
    svg_path = os.path.join(output_dir, f"{name}.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    console.print(f"  [dim]Saved {png_path}[/dim]")
    console.print(f"  [dim]Saved {svg_path}[/dim]")


def plot_loss_curves(runs_data: dict[str, dict], output_dir: str) -> None:
    """Generate three plots from the loaded run data.

    1. Training & validation loss curves
    2. GPU memory over time
    3. Step time over training

    Args:
        runs_data: dict mapping mode key -> output of load_training_data()
        output_dir: directory to save plots
    """
    # ── Plot 1: Loss curves ──────────────────────────────────────────────
    console.print("[bold cyan]Generating loss curve plot...[/bold cyan]")
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.set_facecolor(BG_COLOR)
    _apply_dark_theme(ax)

    for mode in MODE_ORDER:
        if mode not in runs_data:
            continue
        d = runs_data[mode]
        color = MODE_COLORS[mode]
        label = MODE_LABELS[mode]

        if d["steps"] and d["loss"]:
            steps = np.array(d["steps"])
            loss = np.array(d["loss"])

            # Raw loss as faint background
            ax.plot(steps, loss, color=color, alpha=0.15, linewidth=0.8)

            # Smoothed training loss (solid line)
            smoothed = _smooth(d["loss"], window=20)
            ax.plot(steps, smoothed, color=color, linewidth=2.0,
                    label=f"{label} (train)")

        if d["val_steps"] and d["val_loss"]:
            # Validation loss (dotted with markers)
            ax.plot(d["val_steps"], d["val_loss"], color=color,
                    linestyle=":", linewidth=1.8, marker="o", markersize=5,
                    label=f"{label} (val)")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss by Fine-Tuning Mode", fontsize=14,
                 fontweight="bold")
    legend = ax.legend(facecolor="#161b22", edgecolor=GRID_COLOR,
                       fontsize=10, loc="upper right")
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    _save(fig, output_dir, "loss_curves")
    plt.close(fig)

    # ── Plot 2: GPU memory ───────────────────────────────────────────────
    console.print("[bold cyan]Generating GPU memory plot...[/bold cyan]")
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.set_facecolor(BG_COLOR)
    _apply_dark_theme(ax)

    for mode in MODE_ORDER:
        if mode not in runs_data:
            continue
        d = runs_data[mode]
        color = MODE_COLORS[mode]
        label = MODE_LABELS[mode]

        if d["steps"] and d["gpu_memory_mb"]:
            mem_gb = np.array(d["gpu_memory_mb"]) / 1024.0
            ax.plot(d["steps"], mem_gb, color=color, linewidth=1.5,
                    label=label, alpha=0.9)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("GPU Memory (GB)", fontsize=12)
    ax.set_title("GPU Memory Usage Over Training", fontsize=14,
                 fontweight="bold")
    legend = ax.legend(facecolor="#161b22", edgecolor=GRID_COLOR,
                       fontsize=10, loc="upper right")
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    _save(fig, output_dir, "gpu_memory")
    plt.close(fig)

    # ── Plot 3: Step time ────────────────────────────────────────────────
    console.print("[bold cyan]Generating step time plot...[/bold cyan]")
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.set_facecolor(BG_COLOR)
    _apply_dark_theme(ax)

    for mode in MODE_ORDER:
        if mode not in runs_data:
            continue
        d = runs_data[mode]
        color = MODE_COLORS[mode]
        label = MODE_LABELS[mode]

        if d["steps"] and d["step_time_sec"]:
            steps = np.array(d["steps"])
            times = np.array(d["step_time_sec"])

            # Raw as faint
            ax.plot(steps, times, color=color, alpha=0.2, linewidth=0.6)

            # Smoothed
            smoothed = _smooth(d["step_time_sec"], window=10)
            ax.plot(steps, smoothed, color=color, linewidth=1.5,
                    label=label, alpha=0.9)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Step Time (seconds)", fontsize=12)
    ax.set_title("Per-Step Training Time", fontsize=14, fontweight="bold")
    legend = ax.legend(facecolor="#161b22", edgecolor=GRID_COLOR,
                       fontsize=10, loc="upper right")
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    _save(fig, output_dir, "step_time")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_all_plots(
    results_dir: str,
    output_dir: Optional[str] = None,
) -> None:
    """Discover runs, load data, and generate all plots.

    Args:
        results_dir: path to the results/ directory containing run folders.
        output_dir: where to save plots. Defaults to
                    ``<results_dir>/cross_comparison/``.
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "cross_comparison")

    console.print(
        "\n[bold green]===  Generating training plots  ===[/bold green]\n"
    )

    best_runs = find_best_runs(results_dir)

    if not best_runs:
        console.print("[yellow]No successful runs found. Nothing to plot.[/yellow]")
        return

    runs_data: dict[str, dict] = {}
    for mode, run_dir in best_runs.items():
        console.print(
            f"  Loading [bold]{MODE_LABELS[mode]}[/bold] from "
            f"[dim]{os.path.basename(run_dir)}[/dim]"
        )
        runs_data[mode] = load_training_data(run_dir)

    plot_loss_curves(runs_data, output_dir)

    console.print(
        f"\n[bold green]All plots saved to {output_dir}[/bold green]\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    results = sys.argv[1] if len(sys.argv) > 1 else "results"
    out = sys.argv[2] if len(sys.argv) > 2 else None
    generate_all_plots(results, out)
