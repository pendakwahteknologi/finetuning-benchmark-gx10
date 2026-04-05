"""Cross-mode comparison: Base vs LoRA vs QLoRA vs Full Fine-Tune.

Reads evaluation data from all completed runs, computes per-question metrics
across modes, and produces:
  - Rich terminal tables (designed for video recording)
  - HTML report (interactive, dark-themed)
  - Markdown summary (for README / class report)
"""

import csv
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Optional

from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..utils.logging_utils import console
from .eval_metrics import compute_rouge_l, compute_bleu, normalize_text


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

MODE_LABELS = {
    "lora": "LoRA",
    "qlora": "QLoRA",
    "fullft": "Full Fine-Tune",
}

MODE_ORDER = ["lora", "qlora", "fullft"]


def _find_best_run(results_dir: str, mode: str) -> Optional[str]:
    """Find the most recent successful run for a given mode."""
    candidates = []
    for entry in sorted(os.listdir(results_dir)):
        if f"_{mode}_" not in entry:
            continue
        run_dir = os.path.join(results_dir, entry)
        summary = os.path.join(run_dir, "summary.txt")
        eval_dir = os.path.join(run_dir, "evaluation")
        if os.path.isfile(summary) and os.path.isdir(eval_dir):
            with open(summary) as f:
                if "Status:               success" in f.read():
                    candidates.append(run_dir)
    return candidates[-1] if candidates else None


def _load_predictions(jsonl_path: str) -> dict:
    """Load predictions keyed by question id."""
    preds = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            preds[rec["id"]] = rec
    return preds


def _load_training_metrics(run_dir: str) -> dict:
    """Load benchmark_metrics.json for a run."""
    path = os.path.join(run_dir, "benchmark_metrics.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_all_runs(results_dir: str) -> dict:
    """Load baseline + finetuned predictions and training metrics for each mode."""
    runs = {}
    for mode in MODE_ORDER:
        run_dir = _find_best_run(results_dir, mode)
        if run_dir is None:
            continue
        eval_dir = os.path.join(run_dir, "evaluation")
        baseline_path = os.path.join(eval_dir, "baseline_predictions.jsonl")
        finetuned_path = os.path.join(eval_dir, "finetuned_predictions.jsonl")
        if not os.path.isfile(baseline_path) or not os.path.isfile(finetuned_path):
            continue
        runs[mode] = {
            "run_dir": run_dir,
            "baseline": _load_predictions(baseline_path),
            "finetuned": _load_predictions(finetuned_path),
            "training": _load_training_metrics(run_dir),
        }
    return runs


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_cross_metrics(runs: dict) -> dict:
    """Compute per-question and aggregated metrics across all modes.

    Returns:
        {
            "questions": [
                {
                    "id": 0, "category": "...", "instruction": "...", "input": "...",
                    "reference": "...",
                    "baseline": {"prediction": "...", "rouge_l": ..., "bleu": ...},
                    "lora":     {"prediction": "...", "rouge_l": ..., "bleu": ...},
                    "qlora":    {"prediction": "...", "rouge_l": ..., "bleu": ...},
                    "fullft":   {"prediction": "...", "rouge_l": ..., "bleu": ...},
                    "best_mode": "fullft",
                },
                ...
            ],
            "aggregated": {
                "overall": { mode: {"rouge_l": ..., "bleu": ...}, ... },
                "by_category": { cat: { mode: {"rouge_l": ..., "bleu": ...}, ... }, ... },
                "win_counts": { mode: N, ... },
                "category_winners": { cat: mode, ... },
            },
            "training_summary": { mode: { ... }, ... },
        }
    """
    # Use baseline from any run (they're identical across modes)
    any_mode = next(iter(runs))
    baseline_preds = runs[any_mode]["baseline"]
    question_ids = sorted(baseline_preds.keys())

    questions = []
    mode_scores = {m: {"rouge_l": [], "bleu": []} for m in runs}
    cat_scores = defaultdict(lambda: {m: {"rouge_l": [], "bleu": []} for m in runs})
    win_counts = defaultdict(int)

    for qid in question_ids:
        bp = baseline_preds[qid]
        reference = bp["reference_output"]

        q = {
            "id": qid,
            "category": bp["category"],
            "instruction": bp["instruction"],
            "input": bp.get("input", ""),
            "reference": reference,
            "baseline": {
                "prediction": bp["prediction"],
                "rouge_l": compute_rouge_l(bp["prediction"], reference),
                "bleu": compute_bleu(bp["prediction"], reference),
            },
        }

        best_rouge = -1
        best_mode = None

        for mode in MODE_ORDER:
            if mode not in runs:
                continue
            fp = runs[mode]["finetuned"].get(qid, {})
            pred = fp.get("prediction", "")
            rouge = compute_rouge_l(pred, reference)
            bleu = compute_bleu(pred, reference)

            q[mode] = {
                "prediction": pred,
                "rouge_l": rouge,
                "bleu": bleu,
            }

            mode_scores[mode]["rouge_l"].append(rouge)
            mode_scores[mode]["bleu"].append(bleu)
            cat_scores[bp["category"]][mode]["rouge_l"].append(rouge)
            cat_scores[bp["category"]][mode]["bleu"].append(bleu)

            if rouge > best_rouge:
                best_rouge = rouge
                best_mode = mode

        q["best_mode"] = best_mode
        if best_mode:
            win_counts[best_mode] += 1

        questions.append(q)

    # Aggregate
    overall = {}
    for mode in runs:
        overall[mode] = {
            "rouge_l_mean": statistics.mean(mode_scores[mode]["rouge_l"]),
            "rouge_l_median": statistics.median(mode_scores[mode]["rouge_l"]),
            "bleu_mean": statistics.mean(mode_scores[mode]["bleu"]),
            "bleu_median": statistics.median(mode_scores[mode]["bleu"]),
        }

    # Baseline aggregate
    base_rouges = [q["baseline"]["rouge_l"] for q in questions]
    base_bleus = [q["baseline"]["bleu"] for q in questions]
    overall["baseline"] = {
        "rouge_l_mean": statistics.mean(base_rouges),
        "rouge_l_median": statistics.median(base_rouges),
        "bleu_mean": statistics.mean(base_bleus),
        "bleu_median": statistics.median(base_bleus),
    }

    by_category = {}
    category_winners = {}
    for cat in sorted(cat_scores.keys()):
        by_category[cat] = {}
        best_cat_rouge = -1
        best_cat_mode = None
        for mode in runs:
            r = cat_scores[cat][mode]["rouge_l"]
            b = cat_scores[cat][mode]["bleu"]
            m_rouge = statistics.mean(r) if r else 0
            m_bleu = statistics.mean(b) if b else 0
            by_category[cat][mode] = {
                "rouge_l_mean": m_rouge,
                "bleu_mean": m_bleu,
                "count": len(r),
            }
            if m_rouge > best_cat_rouge:
                best_cat_rouge = m_rouge
                best_cat_mode = mode
        # Baseline per category
        cat_base_r = [q["baseline"]["rouge_l"] for q in questions if q["category"] == cat]
        cat_base_b = [q["baseline"]["bleu"] for q in questions if q["category"] == cat]
        by_category[cat]["baseline"] = {
            "rouge_l_mean": statistics.mean(cat_base_r) if cat_base_r else 0,
            "bleu_mean": statistics.mean(cat_base_b) if cat_base_b else 0,
            "count": len(cat_base_r),
        }
        category_winners[cat] = best_cat_mode

    # Training summary
    training_summary = {}
    for mode in runs:
        tm = runs[mode]["training"]
        training_summary[mode] = {
            "total_time_sec": tm.get("total_wall_clock_sec", 0),
            "peak_gpu_memory_gb": tm.get("peak_gpu_memory_gb", 0),
            "final_loss": tm.get("final_loss", 0),
            "tokens_per_sec": tm.get("tokens_per_sec", 0),
            "avg_step_time": tm.get("avg_step_time", 0),
        }

    return {
        "questions": questions,
        "aggregated": {
            "overall": overall,
            "by_category": by_category,
            "win_counts": dict(win_counts),
            "category_winners": category_winners,
        },
        "training_summary": training_summary,
    }


# ---------------------------------------------------------------------------
# Terminal output (Rich)
# ---------------------------------------------------------------------------

def _delta_str(ft_val: float, base_val: float) -> tuple:
    """Return (formatted delta string, style)."""
    delta = ft_val - base_val
    sign = "+" if delta >= 0 else ""
    style = "green" if delta > 0 else ("red" if delta < 0 else "dim")
    return f"{sign}{delta:.4f}", style


def _mode_color(mode: str) -> str:
    return {"lora": "cyan", "qlora": "magenta", "fullft": "yellow"}.get(mode, "white")


def print_cross_compare(data: dict):
    """Print full cross-comparison to terminal with Rich."""

    agg = data["aggregated"]
    overall = agg["overall"]
    training = data["training_summary"]
    modes_present = [m for m in MODE_ORDER if m in overall and m != "baseline"]

    # ── Banner ──
    console.print()
    console.print(Panel(
        "[bold white]CROSS-MODE COMPARISON[/]\n"
        "[dim]Base Model vs LoRA vs QLoRA vs Full Fine-Tune[/]",
        border_style="bright_cyan",
        padding=(1, 4),
    ))

    # ── Table 1: Training Performance ──
    console.print()
    t1 = Table(
        title="[bold]Training Performance Summary[/]",
        border_style="cyan",
        show_lines=True,
        title_style="bold cyan",
    )
    t1.add_column("Metric", style="bold white", width=22)
    for mode in modes_present:
        t1.add_column(MODE_LABELS.get(mode, mode), style=_mode_color(mode),
                       width=18, justify="right")

    def _fmt_time(sec):
        if sec <= 0:
            return "--"
        h, rem = divmod(int(sec), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

    rows = [
        ("Total Time", lambda m: _fmt_time(training.get(m, {}).get("total_time_sec", 0))),
        ("Avg Step Time", lambda m: f"{training.get(m, {}).get('avg_step_time', 0):.2f}s"),
        ("Peak GPU Memory", lambda m: f"{training.get(m, {}).get('peak_gpu_memory_gb', 0):.1f} GB"),
        ("Final Loss", lambda m: f"{training.get(m, {}).get('final_loss', 0):.4f}"),
        ("Tokens/sec", lambda m: f"{training.get(m, {}).get('tokens_per_sec', 0):,.1f}"),
    ]
    for label, fn in rows:
        t1.add_row(label, *[fn(m) for m in modes_present])
    console.print(t1)

    # ── Table 2: Overall Evaluation Metrics ──
    console.print()
    t2 = Table(
        title="[bold]Evaluation Metrics: Overall (80 Questions)[/]",
        border_style="green",
        show_lines=True,
        title_style="bold green",
    )
    t2.add_column("Metric", style="bold white", width=18)
    t2.add_column("Base Model", style="dim white", width=14, justify="right")
    for mode in modes_present:
        t2.add_column(
            f"{MODE_LABELS.get(mode, mode)}",
            style=_mode_color(mode), width=14, justify="right",
        )

    base = overall["baseline"]
    for metric_key, label in [
        ("rouge_l_mean", "ROUGE-L (mean)"),
        ("rouge_l_median", "ROUGE-L (median)"),
        ("bleu_mean", "BLEU (mean)"),
        ("bleu_median", "BLEU (median)"),
    ]:
        base_val = base[metric_key]
        cells = [f"{base_val:.4f}"]
        for mode in modes_present:
            ft_val = overall[mode][metric_key]
            delta_s, delta_style = _delta_str(ft_val, base_val)
            cells.append(f"{ft_val:.4f} [{delta_style}]({delta_s})[/]")
        t2.add_row(label, *cells)

    # Win counts row
    total_q = len(data["questions"])
    win_cells = ["--"]
    for mode in modes_present:
        wins = agg["win_counts"].get(mode, 0)
        pct = wins / total_q * 100
        is_top = wins == max(agg["win_counts"].values())
        style = "bold green" if is_top else "white"
        win_cells.append(f"[{style}]{wins} ({pct:.0f}%)[/]")
    t2.add_row("[bold]Best Answer Wins[/]", *win_cells)

    console.print(t2)

    # ── Table 3: Per-Category Breakdown ──
    console.print()
    t3 = Table(
        title="[bold]ROUGE-L by Category: Base vs Fine-Tuned Modes[/]",
        border_style="blue",
        show_lines=True,
        title_style="bold blue",
    )
    t3.add_column("Category", style="bold white", width=22)
    t3.add_column("Base", style="dim white", width=10, justify="right")
    for mode in modes_present:
        t3.add_column(MODE_LABELS.get(mode, mode), style=_mode_color(mode),
                       width=14, justify="right")
    t3.add_column("Winner", style="bold green", width=16)

    for cat in sorted(agg["by_category"].keys()):
        cat_data = agg["by_category"][cat]
        base_r = cat_data["baseline"]["rouge_l_mean"]
        cells = [cat, f"{base_r:.4f}"]
        for mode in modes_present:
            ft_r = cat_data.get(mode, {}).get("rouge_l_mean", 0)
            delta_s, delta_style = _delta_str(ft_r, base_r)
            cells.append(f"{ft_r:.4f} [{delta_style}]({delta_s})[/]")
        winner = agg["category_winners"].get(cat, "?")
        cells.append(MODE_LABELS.get(winner, winner))
        t3.add_row(*cells)

    console.print(t3)

    # ── Table 4: Side-by-side sample comparisons ──
    console.print()
    console.print(Panel(
        "[bold white]SIDE-BY-SIDE COMPARISONS[/]\n"
        "[dim]Showing select questions across all modes[/]",
        border_style="bright_cyan",
        padding=(0, 4),
    ))

    # Pick 2 questions from each category (best improvement + most interesting)
    shown = _pick_showcase_questions(data["questions"], modes_present)
    for q in shown:
        _print_question_panel(q, modes_present)

    # ── Final verdict ──
    _print_verdict(data, modes_present)


def _pick_showcase_questions(questions: list, modes: list, per_category: int = 1) -> list:
    """Pick showcase questions: one per category with biggest improvement over baseline."""
    by_cat = defaultdict(list)
    for q in questions:
        by_cat[q["category"]].append(q)

    selected = []
    for cat in sorted(by_cat.keys()):
        cat_qs = by_cat[cat]
        # Sort by max improvement over baseline across any mode
        def max_improvement(q):
            base_r = q["baseline"]["rouge_l"]
            return max(
                (q.get(m, {}).get("rouge_l", 0) - base_r) for m in modes
            )
        cat_qs.sort(key=max_improvement, reverse=True)
        selected.extend(cat_qs[:per_category])

    return selected


def _print_question_panel(q: dict, modes: list):
    """Print a single question with all mode responses."""
    instruction = q["instruction"]
    if q.get("input"):
        instruction += f"\n[dim]Context: {q['input'][:200]}{'...' if len(q.get('input','')) > 200 else ''}[/]"

    lines = [f"[bold cyan]Q:[/] {instruction[:300]}"]
    lines.append(f"[dim]Reference:[/] {q['reference'][:250]}")
    lines.append("")

    base_pred = q["baseline"]["prediction"]
    lines.append(f"[dim white]BASE MODEL[/] [dim](ROUGE-L: {q['baseline']['rouge_l']:.3f})[/]")
    lines.append(f"  {base_pred[:300]}")
    lines.append("")

    for mode in modes:
        if mode not in q:
            continue
        mdata = q[mode]
        color = _mode_color(mode)
        label = MODE_LABELS.get(mode, mode)
        delta = mdata["rouge_l"] - q["baseline"]["rouge_l"]
        sign = "+" if delta >= 0 else ""
        delta_style = "green" if delta > 0 else ("red" if delta < 0 else "dim")
        is_best = (q.get("best_mode") == mode)
        star = " [bold green]*BEST*[/]" if is_best else ""

        lines.append(
            f"[{color}]{label}[/] [dim](ROUGE-L: {mdata['rouge_l']:.3f}, "
            f"[{delta_style}]{sign}{delta:.3f}[/])[/]{star}"
        )
        lines.append(f"  {mdata['prediction'][:300]}")
        lines.append("")

    console.print(Panel(
        "\n".join(lines),
        title=f"[bold]{q['category'].upper()} — Question {q['id']}[/]",
        border_style="bright_blue",
        padding=(1, 2),
    ))


def _print_verdict(data: dict, modes: list):
    """Print final verdict panel."""
    agg = data["aggregated"]
    overall = agg["overall"]
    training = data["training_summary"]

    # Determine winners
    best_rouge_mode = max(modes, key=lambda m: overall[m]["rouge_l_mean"])
    best_bleu_mode = max(modes, key=lambda m: overall[m]["bleu_mean"])
    most_wins_mode = max(modes, key=lambda m: agg["win_counts"].get(m, 0))
    fastest_mode = min(modes, key=lambda m: training.get(m, {}).get("total_time_sec", float("inf")))
    lowest_mem_mode = min(modes, key=lambda m: training.get(m, {}).get("peak_gpu_memory_gb", float("inf")))
    lowest_loss_mode = min(modes, key=lambda m: training.get(m, {}).get("final_loss", float("inf")))

    base_rouge = overall["baseline"]["rouge_l_mean"]

    lines = []
    lines.append("[bold white]CATEGORY RANKINGS[/]")
    lines.append("")
    for mode in modes:
        cats_won = [c for c, w in agg["category_winners"].items() if w == mode]
        wins = agg["win_counts"].get(mode, 0)
        color = _mode_color(mode)
        label = MODE_LABELS.get(mode, mode)
        rouge_delta = overall[mode]["rouge_l_mean"] - base_rouge
        sign = "+" if rouge_delta >= 0 else ""
        lines.append(
            f"  [{color}]{label:16s}[/]  "
            f"ROUGE-L: {overall[mode]['rouge_l_mean']:.4f} ({sign}{rouge_delta:.4f} vs base)  "
            f"Wins: {wins}/80  "
            f"Categories: {', '.join(cats_won) if cats_won else 'none'}"
        )
    lines.append("")

    lines.append("[bold white]AWARDS[/]")
    lines.append("")
    awards = [
        ("Best Quality (ROUGE-L)", best_rouge_mode),
        ("Best Quality (BLEU)", best_bleu_mode),
        ("Most Per-Question Wins", most_wins_mode),
        ("Fastest Training", fastest_mode),
        ("Lowest Memory Usage", lowest_mem_mode),
        ("Lowest Final Loss", lowest_loss_mode),
    ]
    for award, mode in awards:
        color = _mode_color(mode)
        label = MODE_LABELS.get(mode, mode)
        lines.append(f"  {award:30s}  [{color}]{label}[/]")

    console.print()
    console.print(Panel(
        "\n".join(lines),
        title="[bold bright_cyan]FINAL VERDICT[/]",
        border_style="bright_cyan",
        padding=(1, 3),
    ))


# ---------------------------------------------------------------------------
# File outputs
# ---------------------------------------------------------------------------

def save_markdown(data: dict, output_path: str):
    """Save comprehensive markdown report."""
    agg = data["aggregated"]
    overall = agg["overall"]
    training = data["training_summary"]
    modes = [m for m in MODE_ORDER if m in overall and m != "baseline"]
    base = overall["baseline"]

    lines = [
        "# Cross-Mode Comparison: Base vs LoRA vs QLoRA vs Full Fine-Tune",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Training Performance",
        "",
        "| Metric | " + " | ".join(MODE_LABELS.get(m, m) for m in modes) + " |",
        "|--------|" + "|".join("-------:" for _ in modes) + "|",
    ]

    def _fmt_time(sec):
        if sec <= 0:
            return "--"
        h, rem = divmod(int(sec), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"

    train_rows = [
        ("Total Time", lambda m: _fmt_time(training.get(m, {}).get("total_time_sec", 0))),
        ("Avg Step Time", lambda m: f"{training.get(m, {}).get('avg_step_time', 0):.2f}s"),
        ("Peak GPU Memory", lambda m: f"{training.get(m, {}).get('peak_gpu_memory_gb', 0):.1f} GB"),
        ("Final Loss", lambda m: f"{training.get(m, {}).get('final_loss', 0):.4f}"),
        ("Tokens/sec", lambda m: f"{training.get(m, {}).get('tokens_per_sec', 0):,.1f}"),
    ]
    for label, fn in train_rows:
        lines.append(f"| {label} | " + " | ".join(fn(m) for m in modes) + " |")

    lines += [
        "",
        "## Evaluation Metrics (80 Curated Questions)",
        "",
        "| Metric | Base Model | " + " | ".join(MODE_LABELS.get(m, m) for m in modes) + " |",
        "|--------|-----------|" + "|".join("-----------:" for _ in modes) + "|",
    ]

    for metric_key, label in [
        ("rouge_l_mean", "ROUGE-L (mean)"),
        ("rouge_l_median", "ROUGE-L (median)"),
        ("bleu_mean", "BLEU (mean)"),
        ("bleu_median", "BLEU (median)"),
    ]:
        base_val = base[metric_key]
        cells = [f"{base_val:.4f}"]
        for mode in modes:
            ft_val = overall[mode][metric_key]
            delta = ft_val - base_val
            sign = "+" if delta >= 0 else ""
            cells.append(f"{ft_val:.4f} ({sign}{delta:.4f})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # Win counts
    total_q = len(data["questions"])
    win_cells = ["--"]
    for mode in modes:
        wins = agg["win_counts"].get(mode, 0)
        pct = wins / total_q * 100
        win_cells.append(f"{wins} ({pct:.0f}%)")
    lines.append(f"| **Best Answer Wins** | " + " | ".join(win_cells) + " |")

    # Per-category
    lines += [
        "",
        "## ROUGE-L by Category",
        "",
        "| Category | Base | " + " | ".join(MODE_LABELS.get(m, m) for m in modes) + " | Winner |",
        "|----------|------|" + "|".join("------:" for _ in modes) + "|--------|",
    ]
    for cat in sorted(agg["by_category"].keys()):
        cat_data = agg["by_category"][cat]
        base_r = cat_data["baseline"]["rouge_l_mean"]
        cells = [cat, f"{base_r:.4f}"]
        for mode in modes:
            ft_r = cat_data.get(mode, {}).get("rouge_l_mean", 0)
            delta = ft_r - base_r
            sign = "+" if delta >= 0 else ""
            cells.append(f"{ft_r:.4f} ({sign}{delta:.4f})")
        winner = agg["category_winners"].get(cat, "?")
        cells.append(f"**{MODE_LABELS.get(winner, winner)}**")
        lines.append("| " + " | ".join(cells) + " |")

    # Verdict
    best_rouge_mode = max(modes, key=lambda m: overall[m]["rouge_l_mean"])
    most_wins_mode = max(modes, key=lambda m: agg["win_counts"].get(m, 0))
    fastest_mode = min(modes, key=lambda m: training.get(m, {}).get("total_time_sec", float("inf")))
    lowest_mem_mode = min(modes, key=lambda m: training.get(m, {}).get("peak_gpu_memory_gb", float("inf")))

    lines += [
        "",
        "## Verdict",
        "",
        "| Award | Winner |",
        "|-------|--------|",
        f"| Best Quality (ROUGE-L) | **{MODE_LABELS.get(best_rouge_mode, best_rouge_mode)}** |",
        f"| Most Per-Question Wins | **{MODE_LABELS.get(most_wins_mode, most_wins_mode)}** |",
        f"| Fastest Training | **{MODE_LABELS.get(fastest_mode, fastest_mode)}** |",
        f"| Lowest Memory | **{MODE_LABELS.get(lowest_mem_mode, lowest_mem_mode)}** |",
    ]

    # Side-by-side samples (all questions)
    lines += [
        "",
        "## Side-by-Side Comparisons (All 80 Questions)",
        "",
    ]
    for q in data["questions"]:
        instruction = q["instruction"][:200]
        lines.append(f"### Q{q['id']}: [{q['category']}] {instruction}")
        lines.append("")
        if q.get("input"):
            lines.append(f"> **Context:** {q['input'][:300]}")
            lines.append("")
        lines.append(f"**Reference:** {q['reference'][:500]}")
        lines.append("")
        lines.append(f"**Base Model** (ROUGE-L: {q['baseline']['rouge_l']:.3f}):")
        lines.append(f"> {q['baseline']['prediction'][:500]}")
        lines.append("")
        for mode in modes:
            if mode not in q:
                continue
            mdata = q[mode]
            delta = mdata['rouge_l'] - q['baseline']['rouge_l']
            sign = "+" if delta >= 0 else ""
            best_mark = " **[BEST]**" if q.get("best_mode") == mode else ""
            lines.append(
                f"**{MODE_LABELS.get(mode, mode)}** "
                f"(ROUGE-L: {mdata['rouge_l']:.3f}, {sign}{delta:.3f}){best_mark}:"
            )
            lines.append(f"> {mdata['prediction'][:500]}")
            lines.append("")
        lines.append("---")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def save_html(data: dict, output_path: str):
    """Save interactive HTML comparison report."""
    agg = data["aggregated"]
    overall = agg["overall"]
    training = data["training_summary"]
    modes = [m for m in MODE_ORDER if m in overall and m != "baseline"]
    base = overall["baseline"]
    questions = data["questions"]

    mode_colors = {"lora": "#00d4ff", "qlora": "#ff79c6", "fullft": "#f1fa8c"}

    def _e(text):
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _fmt_time(sec):
        if sec <= 0:
            return "--"
        h, rem = divmod(int(sec), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

    def _delta_html(ft_val, base_val):
        delta = ft_val - base_val
        sign = "+" if delta >= 0 else ""
        color = "#50fa7b" if delta > 0 else ("#ff5555" if delta < 0 else "#6272a4")
        return f'<span style="color:{color}">{sign}{delta:.4f}</span>'

    # Build training rows
    train_html_rows = ""
    train_metrics = [
        ("Total Time", lambda m: _fmt_time(training.get(m, {}).get("total_time_sec", 0))),
        ("Avg Step Time", lambda m: f"{training.get(m, {}).get('avg_step_time', 0):.2f}s"),
        ("Peak GPU Memory", lambda m: f"{training.get(m, {}).get('peak_gpu_memory_gb', 0):.1f} GB"),
        ("Final Loss", lambda m: f"{training.get(m, {}).get('final_loss', 0):.4f}"),
        ("Tokens/sec", lambda m: f"{training.get(m, {}).get('tokens_per_sec', 0):,.1f}"),
    ]
    for label, fn in train_metrics:
        train_html_rows += f"<tr><td>{label}</td>"
        for mode in modes:
            train_html_rows += f"<td>{fn(mode)}</td>"
        train_html_rows += "</tr>\n"

    # Build eval metric rows
    eval_html_rows = ""
    for metric_key, label in [
        ("rouge_l_mean", "ROUGE-L (mean)"),
        ("rouge_l_median", "ROUGE-L (median)"),
        ("bleu_mean", "BLEU (mean)"),
        ("bleu_median", "BLEU (median)"),
    ]:
        base_val = base[metric_key]
        eval_html_rows += f"<tr><td>{label}</td><td>{base_val:.4f}</td>"
        for mode in modes:
            ft_val = overall[mode][metric_key]
            eval_html_rows += f"<td>{ft_val:.4f} {_delta_html(ft_val, base_val)}</td>"
        eval_html_rows += "</tr>\n"

    # Win counts
    total_q = len(questions)
    eval_html_rows += "<tr class='highlight'><td><strong>Best Answer Wins</strong></td><td>--</td>"
    max_wins = max(agg["win_counts"].values()) if agg["win_counts"] else 0
    for mode in modes:
        wins = agg["win_counts"].get(mode, 0)
        pct = wins / total_q * 100
        style = "color:#50fa7b;font-weight:bold" if wins == max_wins else ""
        eval_html_rows += f'<td style="{style}">{wins} ({pct:.0f}%)</td>'
    eval_html_rows += "</tr>\n"

    # Category rows
    cat_html_rows = ""
    for cat in sorted(agg["by_category"].keys()):
        cat_data = agg["by_category"][cat]
        base_r = cat_data["baseline"]["rouge_l_mean"]
        winner = agg["category_winners"].get(cat, "?")
        cat_html_rows += f"<tr><td>{cat}</td><td>{base_r:.4f}</td>"
        for mode in modes:
            ft_r = cat_data.get(mode, {}).get("rouge_l_mean", 0)
            cat_html_rows += f"<td>{ft_r:.4f} {_delta_html(ft_r, base_r)}</td>"
        cat_html_rows += f"<td><strong>{MODE_LABELS.get(winner, winner)}</strong></td></tr>\n"

    # Question cards
    question_cards = ""
    categories = sorted(set(q["category"] for q in questions))
    cat_options = "".join(f'<option value="{c}">{c}</option>' for c in categories)

    for q in questions:
        best = q.get("best_mode", "")
        mode_blocks = ""
        for mode in modes:
            if mode not in q:
                continue
            mdata = q[mode]
            delta = mdata["rouge_l"] - q["baseline"]["rouge_l"]
            sign = "+" if delta >= 0 else ""
            d_color = "#50fa7b" if delta > 0 else ("#ff5555" if delta < 0 else "#6272a4")
            is_best = (best == mode)
            best_badge = ' <span class="badge">BEST</span>' if is_best else ""
            color = mode_colors.get(mode, "#fff")
            mode_blocks += f"""
            <div class="mode-block" style="border-left: 3px solid {color}">
                <div class="mode-label" style="color:{color}">{MODE_LABELS.get(mode, mode)}
                    <span class="score">ROUGE-L: {mdata['rouge_l']:.3f}
                    <span style="color:{d_color}">({sign}{delta:.3f})</span></span>{best_badge}
                </div>
                <div class="prediction">{_e(mdata['prediction'][:600])}</div>
            </div>"""

        context_html = ""
        if q.get("input"):
            context_html = f'<div class="context"><strong>Context:</strong> {_e(q["input"][:400])}</div>'

        question_cards += f"""
        <div class="question-card" data-category="{q['category']}" data-id="{q['id']}">
            <div class="q-header">
                <span class="q-num">Q{q['id']}</span>
                <span class="q-cat">{q['category']}</span>
            </div>
            <div class="instruction">{_e(q['instruction'][:400])}</div>
            {context_html}
            <div class="reference"><strong>Reference:</strong> {_e(q['reference'][:400])}</div>
            <div class="mode-block baseline" style="border-left: 3px solid #6272a4">
                <div class="mode-label" style="color:#6272a4">Base Model
                    <span class="score">ROUGE-L: {q['baseline']['rouge_l']:.3f}</span>
                </div>
                <div class="prediction">{_e(q['baseline']['prediction'][:600])}</div>
            </div>
            {mode_blocks}
        </div>"""

    # Verdict
    best_rouge_mode = max(modes, key=lambda m: overall[m]["rouge_l_mean"])
    most_wins_mode = max(modes, key=lambda m: agg["win_counts"].get(m, 0))
    fastest_mode = min(modes, key=lambda m: training.get(m, {}).get("total_time_sec", float("inf")))
    lowest_mem_mode = min(modes, key=lambda m: training.get(m, {}).get("peak_gpu_memory_gb", float("inf")))

    mode_headers = "".join(
        f'<th style="color:{mode_colors.get(m, "#fff")}">{MODE_LABELS.get(m, m)}</th>'
        for m in modes
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cross-Mode Comparison — Fine-Tuning Benchmark</title>
<style>
:root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --dim: #8b949e; --accent: #58a6ff;
    --green: #50fa7b; --red: #ff5555; --cyan: #00d4ff;
    --magenta: #ff79c6; --yellow: #f1fa8c;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace; background: var(--bg); color: var(--text); line-height: 1.6; padding: 2rem; }}
h1 {{ color: var(--cyan); font-size: 1.8rem; margin-bottom: 0.5rem; }}
h2 {{ color: var(--accent); font-size: 1.3rem; margin: 2rem 0 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
.subtitle {{ color: var(--dim); margin-bottom: 2rem; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid var(--border); padding: 10px 14px; text-align: right; font-size: 0.85rem; }}
th {{ background: var(--surface); color: var(--accent); text-align: center; font-weight: 600; }}
td:first-child {{ text-align: left; font-weight: 500; }}
tr:hover {{ background: rgba(88, 166, 255, 0.05); }}
tr.highlight {{ background: var(--surface); }}

.verdict-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin: 1rem 0; }}
.verdict-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem; }}
.verdict-card .award {{ color: var(--dim); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}
.verdict-card .winner {{ font-size: 1.2rem; font-weight: bold; margin-top: 0.3rem; }}

.filter-bar {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; margin: 1rem 0; display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }}
.filter-bar label {{ color: var(--dim); font-size: 0.85rem; }}
.filter-bar select {{ background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 6px 10px; font-family: inherit; font-size: 0.85rem; }}

.question-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }}
.question-card.hidden {{ display: none; }}
.q-header {{ display: flex; gap: 1rem; align-items: center; margin-bottom: 0.8rem; }}
.q-num {{ color: var(--cyan); font-weight: bold; font-size: 1.1rem; }}
.q-cat {{ background: var(--bg); color: var(--dim); padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; }}
.instruction {{ color: var(--text); margin-bottom: 0.8rem; font-size: 0.95rem; }}
.context {{ color: var(--dim); font-size: 0.85rem; margin-bottom: 0.8rem; padding: 0.5rem; background: var(--bg); border-radius: 4px; }}
.reference {{ color: var(--dim); font-size: 0.85rem; margin-bottom: 1rem; padding: 0.5rem; background: var(--bg); border-radius: 4px; }}
.mode-block {{ padding: 0.8rem 1rem; margin: 0.5rem 0; background: var(--bg); border-radius: 4px; }}
.mode-label {{ font-weight: bold; font-size: 0.9rem; margin-bottom: 0.3rem; }}
.mode-label .score {{ font-weight: normal; color: var(--dim); font-size: 0.8rem; margin-left: 0.5rem; }}
.prediction {{ font-size: 0.85rem; color: var(--text); white-space: pre-wrap; word-break: break-word; }}
.badge {{ background: var(--green); color: var(--bg); padding: 1px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: bold; margin-left: 0.5rem; }}

footer {{ margin-top: 3rem; color: var(--dim); font-size: 0.8rem; border-top: 1px solid var(--border); padding-top: 1rem; }}
</style>
</head>
<body>

<h1>Cross-Mode Comparison</h1>
<p class="subtitle">Base Model vs LoRA vs QLoRA vs Full Fine-Tune — 80 Curated Questions</p>

<h2>Training Performance</h2>
<table>
<tr><th>Metric</th>{mode_headers}</tr>
{train_html_rows}
</table>

<h2>Evaluation Metrics</h2>
<table>
<tr><th>Metric</th><th style="color:var(--dim)">Base Model</th>{mode_headers}</tr>
{eval_html_rows}
</table>

<h2>ROUGE-L by Category</h2>
<table>
<tr><th>Category</th><th style="color:var(--dim)">Base</th>{mode_headers}<th>Winner</th></tr>
{cat_html_rows}
</table>

<h2>Verdict</h2>
<div class="verdict-grid">
    <div class="verdict-card">
        <div class="award">Best Quality (ROUGE-L)</div>
        <div class="winner" style="color:{mode_colors.get(best_rouge_mode, '#fff')}">{MODE_LABELS.get(best_rouge_mode, best_rouge_mode)}</div>
    </div>
    <div class="verdict-card">
        <div class="award">Most Per-Question Wins</div>
        <div class="winner" style="color:{mode_colors.get(most_wins_mode, '#fff')}">{MODE_LABELS.get(most_wins_mode, most_wins_mode)}</div>
    </div>
    <div class="verdict-card">
        <div class="award">Fastest Training</div>
        <div class="winner" style="color:{mode_colors.get(fastest_mode, '#fff')}">{MODE_LABELS.get(fastest_mode, fastest_mode)}</div>
    </div>
    <div class="verdict-card">
        <div class="award">Lowest Memory Usage</div>
        <div class="winner" style="color:{mode_colors.get(lowest_mem_mode, '#fff')}">{MODE_LABELS.get(lowest_mem_mode, lowest_mem_mode)}</div>
    </div>
</div>

<h2>Side-by-Side Comparisons</h2>
<div class="filter-bar">
    <label>Filter by category:</label>
    <select id="catFilter" onchange="filterQuestions()">
        <option value="all">All Categories</option>
        {cat_options}
    </select>
    <label>Show:</label>
    <select id="showFilter" onchange="filterQuestions()">
        <option value="all">All Questions</option>
        <option value="improved">Improved over Base</option>
        <option value="degraded">Degraded from Base</option>
    </select>
</div>

{question_cards}

<footer>
    Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Fine-Tuning Benchmark for ASUS Ascent GX10
</footer>

<script>
function filterQuestions() {{
    const cat = document.getElementById('catFilter').value;
    const show = document.getElementById('showFilter').value;
    document.querySelectorAll('.question-card').forEach(card => {{
        let visible = true;
        if (cat !== 'all' && card.dataset.category !== cat) visible = false;
        if (show === 'improved') {{
            const badge = card.querySelector('.badge');
            if (!badge) visible = false;
        }}
        if (show === 'degraded') {{
            const badge = card.querySelector('.badge');
            if (badge) visible = false;
        }}
        card.classList.toggle('hidden', !visible);
    }});
}}
</script>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)


def save_json(data: dict, output_path: str):
    """Save full comparison data as JSON."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_csv(data: dict, output_path: str):
    """Save per-question comparison CSV."""
    questions = data["questions"]
    modes = [m for m in MODE_ORDER if m in data["aggregated"]["overall"] and m != "baseline"]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["id", "category", "instruction", "reference",
                  "baseline_prediction", "baseline_rouge_l", "baseline_bleu"]
        for mode in modes:
            label = MODE_LABELS.get(mode, mode).lower().replace(" ", "_")
            header.extend([f"{label}_prediction", f"{label}_rouge_l", f"{label}_bleu"])
        header.append("best_mode")
        writer.writerow(header)

        for q in questions:
            row = [
                q["id"], q["category"], q["instruction"][:200], q["reference"][:200],
                q["baseline"]["prediction"][:200], f"{q['baseline']['rouge_l']:.4f}",
                f"{q['baseline']['bleu']:.4f}",
            ]
            for mode in modes:
                mdata = q.get(mode, {})
                row.extend([
                    mdata.get("prediction", "")[:200],
                    f"{mdata.get('rouge_l', 0):.4f}",
                    f"{mdata.get('bleu', 0):.4f}",
                ])
            row.append(MODE_LABELS.get(q.get("best_mode", ""), ""))
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cross_compare(results_dir: str, output_dir: Optional[str] = None):
    """Run full cross-mode comparison."""

    console.print("[bold cyan]Loading evaluation data from all runs...[/]")
    runs = load_all_runs(results_dir)

    if len(runs) < 2:
        console.print(f"[yellow]Found only {len(runs)} completed run(s) with evaluation data. "
                       f"Need at least 2 modes to compare.[/]")
        return

    console.print(f"  Found modes: {', '.join(MODE_LABELS.get(m, m) for m in runs)}")
    console.print()

    console.print("[bold cyan]Computing cross-mode metrics...[/]")
    data = compute_cross_metrics(runs)

    # Terminal output
    print_cross_compare(data)

    # File outputs
    out_dir = output_dir or os.path.join(results_dir, "cross_comparison")
    os.makedirs(out_dir, exist_ok=True)

    save_json(data, os.path.join(out_dir, "cross_comparison.json"))
    save_csv(data, os.path.join(out_dir, "cross_comparison.csv"))
    save_markdown(data, os.path.join(out_dir, "cross_comparison.md"))
    save_html(data, os.path.join(out_dir, "cross_comparison.html"))

    console.print()
    console.print(Panel(
        f"[bold green]Reports saved to:[/] {out_dir}\n\n"
        f"  cross_comparison.json   — Full data (machine-readable)\n"
        f"  cross_comparison.csv    — Per-question metrics (spreadsheet)\n"
        f"  cross_comparison.md     — Markdown report (README / class)\n"
        f"  cross_comparison.html   — Interactive HTML report (browser / video)",
        border_style="green",
        padding=(1, 2),
    ))
