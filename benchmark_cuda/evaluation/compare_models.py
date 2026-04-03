"""Generate side-by-side comparison between baseline and finetuned models."""

import csv
import json
import os
from typing import Optional

from rich.table import Table
from rich.panel import Panel

from ..utils.logging_utils import console


def generate_comparison(
    baseline_results: dict,
    finetuned_results: dict,
    config,
    eval_dir: str,
) -> dict:
    """Generate side-by-side comparison files and CLI output."""

    os.makedirs(eval_dir, exist_ok=True)

    baseline_preds = {p["id"]: p for p in baseline_results["predictions"]}
    finetuned_preds = {p["id"]: p for p in finetuned_results["predictions"]}

    # Build side-by-side records
    side_by_side = []
    for qid in sorted(baseline_preds.keys()):
        bp = baseline_preds[qid]
        fp = finetuned_preds.get(qid, {})
        side_by_side.append({
            "id": qid,
            "category": bp["category"],
            "instruction": bp["instruction"],
            "input": bp.get("input", ""),
            "reference_output": bp["reference_output"],
            "baseline_prediction": bp["prediction"],
            "finetuned_prediction": fp.get("prediction", ""),
            "baseline_rouge_l": bp.get("rouge_l", 0),
            "finetuned_rouge_l": fp.get("rouge_l", 0),
            "baseline_bleu": bp.get("bleu", 0),
            "finetuned_bleu": fp.get("bleu", 0),
            "baseline_latency": bp.get("latency_sec", 0),
            "finetuned_latency": fp.get("latency_sec", 0),
        })

    # Save side-by-side JSONL
    sbs_path = os.path.join(eval_dir, "side_by_side_comparison.jsonl")
    with open(sbs_path, "w") as f:
        for rec in side_by_side:
            f.write(json.dumps(rec) + "\n")

    # Metrics comparison
    b_agg = baseline_results["aggregated"]
    f_agg = finetuned_results["aggregated"]

    metrics_comparison = {
        "config": {
            "machine_label": config.machine_label,
            "mode": config.mode,
            "model": config.model_name,
            "dataset": config.dataset_name,
            "num_questions": len(side_by_side),
            "temperature": config.eval_temperature,
            "max_new_tokens": config.eval_max_new_tokens,
            "do_sample": config.eval_do_sample,
        },
        "baseline": b_agg,
        "finetuned": f_agg,
    }

    # Save metrics JSON
    with open(os.path.join(eval_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_comparison, f, indent=2)

    # Save metrics CSV
    _save_metrics_csv(eval_dir, b_agg, f_agg)

    # Save markdown summary
    _save_markdown_summary(eval_dir, metrics_comparison, side_by_side)

    # Save HTML table
    _save_html_table(eval_dir, side_by_side)

    # Print CLI summary
    _print_cli_summary(metrics_comparison, side_by_side)

    return metrics_comparison


def _save_metrics_csv(eval_dir: str, baseline: dict, finetuned: dict):
    path = os.path.join(eval_dir, "evaluation_metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "baseline", "finetuned", "delta"])

        for key in ["rouge_l", "bleu", "exact_match", "normalized_match",
                     "prediction_length", "latency_sec"]:
            b_val = baseline.get("overall", {}).get(key, {}).get("mean", 0)
            f_val = finetuned.get("overall", {}).get(key, {}).get("mean", 0)
            delta = f_val - b_val
            writer.writerow([key, f"{b_val:.4f}", f"{f_val:.4f}", f"{delta:+.4f}"])


def _save_markdown_summary(eval_dir: str, comparison: dict, side_by_side: list):
    cfg = comparison["config"]
    b = comparison["baseline"]
    ft = comparison["finetuned"]

    lines = [
        "# Evaluation Summary",
        "",
        f"- **Machine**: {cfg['machine_label']}",
        f"- **Mode**: {cfg['mode']}",
        f"- **Model**: {cfg['model']}",
        f"- **Questions**: {cfg['num_questions']}",
        f"- **Temperature**: {cfg['temperature']}",
        f"- **Max new tokens**: {cfg['max_new_tokens']}",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Baseline | Fine-tuned | Delta |",
        "|--------|----------|------------|-------|",
    ]

    for key in ["rouge_l", "bleu", "exact_match", "normalized_match",
                 "prediction_length", "latency_sec"]:
        b_val = b.get("overall", {}).get(key, {}).get("mean", 0)
        f_val = ft.get("overall", {}).get(key, {}).get("mean", 0)
        delta = f_val - b_val
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {key} | {b_val:.4f} | {f_val:.4f} | {sign}{delta:.4f} |")

    # Per-category
    lines += ["", "## Category Metrics", ""]
    for cat in sorted(ft.get("by_category", {}).keys()):
        lines.append(f"### {cat}")
        lines.append("")
        lines.append("| Metric | Baseline | Fine-tuned |")
        lines.append("|--------|----------|------------|")
        b_cat = b.get("by_category", {}).get(cat, {})
        f_cat = ft.get("by_category", {}).get(cat, {})
        for key in ["rouge_l", "bleu"]:
            bv = b_cat.get(key, {}).get("mean", 0)
            fv = f_cat.get(key, {}).get("mean", 0)
            lines.append(f"| {key} | {bv:.4f} | {fv:.4f} |")
        lines.append("")

    # Sample comparisons
    lines += ["## Sample Comparisons", ""]
    for rec in side_by_side[:5]:
        lines.append(f"**[{rec['category']}]** {rec['instruction'][:100]}")
        lines.append(f"- Reference: {rec['reference_output'][:150]}")
        lines.append(f"- Baseline: {rec['baseline_prediction'][:150]}")
        lines.append(f"- Fine-tuned: {rec['finetuned_prediction'][:150]}")
        lines.append("")

    path = os.path.join(eval_dir, "evaluation_summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _save_html_table(eval_dir: str, side_by_side: list):
    rows = []
    for rec in side_by_side:
        rows.append(
            f"<tr>"
            f"<td>{rec['id']}</td>"
            f"<td>{rec['category']}</td>"
            f"<td>{_html_escape(rec['instruction'][:80])}</td>"
            f"<td>{_html_escape(rec['reference_output'][:100])}</td>"
            f"<td>{_html_escape(rec['baseline_prediction'][:100])}</td>"
            f"<td>{_html_escape(rec['finetuned_prediction'][:100])}</td>"
            f"<td>{rec['baseline_rouge_l']:.3f}</td>"
            f"<td>{rec['finetuned_rouge_l']:.3f}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Evaluation Comparison</title>
<style>
body {{ font-family: monospace; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #444; padding: 8px; text-align: left; font-size: 12px; }}
th {{ background: #16213e; color: #00d4ff; }}
tr:nth-child(even) {{ background: #0f3460; }}
</style></head>
<body>
<h1>Side-by-Side Evaluation</h1>
<table>
<tr><th>ID</th><th>Category</th><th>Instruction</th><th>Reference</th>
<th>Baseline</th><th>Fine-tuned</th><th>Base ROUGE-L</th><th>FT ROUGE-L</th></tr>
{"".join(rows)}
</table></body></html>"""

    path = os.path.join(eval_dir, "evaluation_table.html")
    with open(path, "w") as f:
        f.write(html)


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _print_cli_summary(comparison: dict, side_by_side: list):
    cfg = comparison["config"]
    b = comparison["baseline"]
    ft = comparison["finetuned"]

    # Metrics table
    table = Table(
        title="Evaluation: Baseline vs Fine-Tuned",
        border_style="cyan",
    )
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Baseline", style="white", width=12, justify="right")
    table.add_column("Fine-tuned", style="white", width=12, justify="right")
    table.add_column("Delta", width=12, justify="right")

    for key in ["rouge_l", "bleu", "exact_match", "normalized_match",
                 "prediction_length", "latency_sec"]:
        b_val = b.get("overall", {}).get(key, {}).get("mean", 0)
        f_val = ft.get("overall", {}).get(key, {}).get("mean", 0)
        delta = f_val - b_val
        delta_style = "green" if delta > 0 else ("red" if delta < 0 else "white")
        if key == "latency_sec":
            delta_style = "green" if delta < 0 else "red"
        sign = "+" if delta >= 0 else ""
        table.add_row(
            key,
            f"{b_val:.4f}",
            f"{f_val:.4f}",
            f"[{delta_style}]{sign}{delta:.4f}[/]",
        )

    console.print()
    console.print(table)

    # Sample previews
    console.print()
    console.print("[bold cyan]Sample Comparisons:[/]")
    console.print()

    for rec in side_by_side[:5]:
        console.print(Panel(
            f"[cyan]Category:[/] {rec['category']}\n"
            f"[cyan]Question:[/] {rec['instruction'][:120]}\n"
            f"[cyan]Reference:[/] {rec['reference_output'][:150]}\n"
            f"[yellow]Baseline:[/] {rec['baseline_prediction'][:150]}\n"
            f"[green]Fine-tuned:[/] {rec['finetuned_prediction'][:150]}",
            title=f"Sample {rec['id']:02d}",
            border_style="dim",
            padding=(0, 1),
        ))
