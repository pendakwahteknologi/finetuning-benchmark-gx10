#!/usr/bin/env python3
"""Cross-model comparison: Instruct vs Base model LoRA fine-tuning.

Reads evaluation data from two LoRA runs (different base models) and produces:
  - HTML report (interactive, dark-themed, 80 questions side-by-side)
  - Markdown summary
  - JSON data
  - CSV summary
"""

import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

RUNS = {
    "instruct": {
        "label": "Llama-3.1-8B-Instruct",
        "short": "Instruct",
        "dir": RESULTS_DIR / "gx10_lora_20260403_140352",
        "color": "#58a6ff",  # blue
        "color_name": "blue",
    },
    "base": {
        "label": "Llama-3.1-8B",
        "short": "Base",
        "dir": RESULTS_DIR / "gx10_lora_20260405_213220",
        "color": "#ff79c6",  # magenta
        "color_name": "magenta",
    },
}

OUTPUT_DIR = RESULTS_DIR / "cross_model_comparison"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(jsonl_path):
    preds = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            preds[rec["id"]] = rec
    return preds


def load_training_metrics(run_dir):
    path = run_dir / "benchmark_metrics.json"
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {}


def load_summary(run_dir):
    path = run_dir / "summary.txt"
    if path.is_file():
        with open(path) as f:
            return f.read()
    return ""


def load_perplexity(run_dir):
    path = run_dir / "evaluation" / "perplexity.json"
    if path.is_file():
        with open(path) as f:
            return json.load(f)
    return {}


def load_side_by_side(jsonl_path):
    """Load side-by-side comparison which has pre-computed ROUGE-L/BLEU."""
    records = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            records[rec["id"]] = rec
    return records


def load_run(key):
    cfg = RUNS[key]
    run_dir = cfg["dir"]
    eval_dir = run_dir / "evaluation"
    return {
        "key": key,
        "label": cfg["label"],
        "short": cfg["short"],
        "color": cfg["color"],
        "baseline": load_predictions(eval_dir / "baseline_predictions.jsonl"),
        "finetuned": load_predictions(eval_dir / "finetuned_predictions.jsonl"),
        "side_by_side": load_side_by_side(eval_dir / "side_by_side_comparison.jsonl"),
        "training": load_training_metrics(run_dir),
        "perplexity": load_perplexity(run_dir),
        "summary": load_summary(run_dir),
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def rouge_l(prediction, reference):
    """Simple ROUGE-L based on LCS."""
    if not prediction or not reference:
        return 0.0
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / n if n else 0
    rec = lcs / m if m else 0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def fmt_time(sec):
    if sec <= 0:
        return "--"
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def delta_str(val, ref, fmt=".4f"):
    d = val - ref
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:{fmt}}"


def delta_html(val, ref, fmt=".4f"):
    d = val - ref
    sign = "+" if d >= 0 else ""
    color = "#50fa7b" if d > 0 else ("#ff5555" if d < 0 else "#8b949e")
    return f'<span style="color:{color}">{sign}{d:{fmt}}</span>'


# ---------------------------------------------------------------------------
# Compute comparison
# ---------------------------------------------------------------------------

def compute_comparison(runs_data):
    """Build question-level and aggregate comparison data."""
    instruct = runs_data["instruct"]
    base = runs_data["base"]

    # Use the same 80 question IDs
    question_ids = sorted(set(instruct["baseline"].keys()) & set(base["baseline"].keys()))

    questions = []
    scores = {k: {"rouge_l": [], "bleu": []} for k in
              ["instruct_baseline", "instruct_ft", "base_baseline", "base_ft"]}
    cat_scores = defaultdict(lambda: {k: {"rouge_l": [], "bleu": []} for k in
                                       ["instruct_baseline", "instruct_ft", "base_baseline", "base_ft"]})
    win_counts = {"instruct_ft": 0, "base_ft": 0, "tie": 0}

    for qid in question_ids:
        ib = instruct["baseline"][qid]
        ift = instruct["finetuned"].get(qid, {})
        bb = base["baseline"][qid]
        bft = base["finetuned"].get(qid, {})

        # Get pre-computed scores from side-by-side comparison files
        isbs = instruct["side_by_side"].get(qid, {})
        bsbs = base["side_by_side"].get(qid, {})

        cat = ib["category"]
        reference = ib["reference_output"]

        ib_rouge = isbs.get("baseline_rouge_l", 0)
        ib_bleu = isbs.get("baseline_bleu", 0)
        ift_rouge = isbs.get("finetuned_rouge_l", 0)
        ift_bleu = isbs.get("finetuned_bleu", 0)
        bb_rouge = bsbs.get("baseline_rouge_l", 0)
        bb_bleu = bsbs.get("baseline_bleu", 0)
        bft_rouge = bsbs.get("finetuned_rouge_l", 0)
        bft_bleu = bsbs.get("finetuned_bleu", 0)

        # Determine winner (finetuned vs finetuned)
        if ift_rouge > bft_rouge:
            winner = "instruct_ft"
        elif bft_rouge > ift_rouge:
            winner = "base_ft"
        else:
            winner = "tie"
        win_counts[winner] += 1

        q = {
            "id": qid,
            "category": cat,
            "instruction": ib["instruction"],
            "input": ib.get("input", ""),
            "reference": reference,
            "instruct_baseline": {
                "prediction": ib["prediction"],
                "rouge_l": ib_rouge, "bleu": ib_bleu,
            },
            "instruct_ft": {
                "prediction": ift.get("prediction", ""),
                "rouge_l": ift_rouge, "bleu": ift_bleu,
            },
            "base_baseline": {
                "prediction": bb["prediction"],
                "rouge_l": bb_rouge, "bleu": bb_bleu,
            },
            "base_ft": {
                "prediction": bft.get("prediction", ""),
                "rouge_l": bft_rouge, "bleu": bft_bleu,
            },
            "winner": winner,
        }
        questions.append(q)

        for sk in scores:
            scores[sk]["rouge_l"].append(q[sk]["rouge_l"])
            scores[sk]["bleu"].append(q[sk]["bleu"])
            cat_scores[cat][sk]["rouge_l"].append(q[sk]["rouge_l"])
            cat_scores[cat][sk]["bleu"].append(q[sk]["bleu"])

    # Aggregated
    agg = {}
    for sk in scores:
        agg[sk] = {
            "rouge_l_mean": statistics.mean(scores[sk]["rouge_l"]),
            "rouge_l_median": statistics.median(scores[sk]["rouge_l"]),
            "bleu_mean": statistics.mean(scores[sk]["bleu"]),
            "bleu_median": statistics.median(scores[sk]["bleu"]),
        }

    by_category = {}
    cat_winners = {}
    for cat in sorted(cat_scores.keys()):
        by_category[cat] = {}
        for sk in scores:
            by_category[cat][sk] = {
                "rouge_l_mean": statistics.mean(cat_scores[cat][sk]["rouge_l"]),
                "bleu_mean": statistics.mean(cat_scores[cat][sk]["bleu"]),
            }
        # Winner = which finetuned model has higher ROUGE-L
        if by_category[cat]["instruct_ft"]["rouge_l_mean"] > by_category[cat]["base_ft"]["rouge_l_mean"]:
            cat_winners[cat] = "instruct_ft"
        else:
            cat_winners[cat] = "base_ft"

    # Training
    training = {}
    for key in ["instruct", "base"]:
        tm = runs_data[key]["training"]
        training[key] = {
            "total_time_sec": tm.get("total_wall_clock_sec", 0),
            "peak_gpu_memory_gb": tm.get("peak_gpu_memory_gb", 0),
            "final_loss": tm.get("final_loss", 0),
            "tokens_per_sec": tm.get("tokens_per_sec", 0),
            "avg_step_time": tm.get("avg_step_time", 0),
        }

    return {
        "questions": questions,
        "aggregated": agg,
        "by_category": by_category,
        "category_winners": cat_winners,
        "win_counts": win_counts,
        "training": training,
        "perplexity": {
            key: runs_data[key]["perplexity"] for key in ["instruct", "base"]
        },
    }


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def save_markdown(data, output_path):
    agg = data["aggregated"]
    training = data["training"]
    wins = data["win_counts"]
    perp = data["perplexity"]

    lines = [
        "# Cross-Model Comparison: Instruct vs Base (LoRA Fine-Tuning)",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "Comparing **Llama-3.1-8B-Instruct** vs **Llama-3.1-8B** (base), both fine-tuned with LoRA on Dolly-15K for 500 steps.",
        "",
        "## Training Performance",
        "",
        "| Metric | Instruct (8B-Instruct) | Base (8B) |",
        "|--------|-------:|-------:|",
    ]

    train_rows = [
        ("Total Time", lambda k: fmt_time(training[k]["total_time_sec"])),
        ("Avg Step Time", lambda k: f"{training[k]['avg_step_time']:.2f}s"),
        ("Peak GPU Memory", lambda k: f"{training[k]['peak_gpu_memory_gb']:.1f} GB"),
        ("Final Loss", lambda k: f"{training[k]['final_loss']:.4f}"),
        ("Tokens/sec", lambda k: f"{training[k]['tokens_per_sec']:,.1f}"),
    ]
    for label, fn in train_rows:
        lines.append(f"| {label} | {fn('instruct')} | {fn('base')} |")

    # Perplexity
    if perp.get("base"):
        lines += [
            "",
            "## Perplexity",
            "",
            "| Metric | Instruct | Base |",
            "|--------|-------:|-------:|",
        ]
        ip = perp.get("instruct", {})
        bp = perp.get("base", {})
        if ip:
            lines.append(f"| Baseline Perplexity | {ip.get('baseline_perplexity', 'n/a')} | {bp.get('baseline_perplexity', 'n/a')} |")
            lines.append(f"| Finetuned Perplexity | {ip.get('finetuned_perplexity', 'n/a')} | {bp.get('finetuned_perplexity', 'n/a')} |")
            lines.append(f"| Improvement | {ip.get('improvement_pct', 'n/a')}% | {bp.get('improvement_pct', 'n/a')}% |")
        else:
            lines.append(f"| Baseline Perplexity | n/a | {bp.get('baseline_perplexity', 'n/a')} |")
            lines.append(f"| Finetuned Perplexity | n/a | {bp.get('finetuned_perplexity', 'n/a')} |")
            lines.append(f"| Improvement | n/a | {bp.get('improvement_pct', 'n/a')}% |")

    # Evaluation — baseline vs finetuned for each model
    lines += [
        "",
        "## Baseline Model Performance (Pre-Training)",
        "",
        "| Metric | Instruct Baseline | Base Baseline |",
        "|--------|-------:|-------:|",
        f"| ROUGE-L (mean) | {agg['instruct_baseline']['rouge_l_mean']:.4f} | {agg['base_baseline']['rouge_l_mean']:.4f} |",
        f"| BLEU (mean) | {agg['instruct_baseline']['bleu_mean']:.4f} | {agg['base_baseline']['bleu_mean']:.4f} |",
        "",
        "## Finetuned Model Performance (Post-Training)",
        "",
        "| Metric | Instruct LoRA | Base LoRA | Delta |",
        "|--------|-------:|-------:|-------:|",
        f"| ROUGE-L (mean) | {agg['instruct_ft']['rouge_l_mean']:.4f} | {agg['base_ft']['rouge_l_mean']:.4f} | {delta_str(agg['base_ft']['rouge_l_mean'], agg['instruct_ft']['rouge_l_mean'])} |",
        f"| ROUGE-L (median) | {agg['instruct_ft']['rouge_l_median']:.4f} | {agg['base_ft']['rouge_l_median']:.4f} | {delta_str(agg['base_ft']['rouge_l_median'], agg['instruct_ft']['rouge_l_median'])} |",
        f"| BLEU (mean) | {agg['instruct_ft']['bleu_mean']:.4f} | {agg['base_ft']['bleu_mean']:.4f} | {delta_str(agg['base_ft']['bleu_mean'], agg['instruct_ft']['bleu_mean'])} |",
        f"| BLEU (median) | {agg['instruct_ft']['bleu_median']:.4f} | {agg['base_ft']['bleu_median']:.4f} | {delta_str(agg['base_ft']['bleu_median'], agg['instruct_ft']['bleu_median'])} |",
        f"| **Best Answer Wins** | **{wins['instruct_ft']} ({wins['instruct_ft']/80*100:.0f}%)** | **{wins['base_ft']} ({wins['base_ft']/80*100:.0f}%)** | Ties: {wins['tie']} |",
    ]

    # Training improvement (delta from own baseline)
    lines += [
        "",
        "## Training Improvement (Delta from Own Baseline)",
        "",
        "| Metric | Instruct (FT - Base) | Base (FT - Base) |",
        "|--------|-------:|-------:|",
        f"| ROUGE-L | {delta_str(agg['instruct_ft']['rouge_l_mean'], agg['instruct_baseline']['rouge_l_mean'])} | {delta_str(agg['base_ft']['rouge_l_mean'], agg['base_baseline']['rouge_l_mean'])} |",
        f"| BLEU | {delta_str(agg['instruct_ft']['bleu_mean'], agg['instruct_baseline']['bleu_mean'])} | {delta_str(agg['base_ft']['bleu_mean'], agg['base_baseline']['bleu_mean'])} |",
    ]

    # ROUGE-L by category (finetuned)
    lines += [
        "",
        "## ROUGE-L by Category (Finetuned)",
        "",
        "| Category | Instruct LoRA | Base LoRA | Winner |",
        "|----------|-------:|-------:|--------|",
    ]
    for cat in sorted(data["by_category"].keys()):
        cd = data["by_category"][cat]
        ir = cd["instruct_ft"]["rouge_l_mean"]
        br = cd["base_ft"]["rouge_l_mean"]
        winner = "Instruct" if data["category_winners"][cat] == "instruct_ft" else "Base"
        lines.append(f"| {cat} | {ir:.4f} | {br:.4f} | **{winner}** |")

    # Verdict
    lines += [
        "",
        "## Verdict",
        "",
        "| Award | Winner |",
        "|-------|--------|",
    ]

    best_rouge = "Instruct" if agg["instruct_ft"]["rouge_l_mean"] > agg["base_ft"]["rouge_l_mean"] else "Base"
    best_bleu = "Instruct" if agg["instruct_ft"]["bleu_mean"] > agg["base_ft"]["bleu_mean"] else "Base"
    most_wins = "Instruct" if wins["instruct_ft"] > wins["base_ft"] else ("Base" if wins["base_ft"] > wins["instruct_ft"] else "Tie")
    lower_loss = "Instruct" if training["instruct"]["final_loss"] < training["base"]["final_loss"] else "Base"
    faster = "Instruct" if training["instruct"]["total_time_sec"] < training["base"]["total_time_sec"] else "Base"

    lines.append(f"| Best Quality (ROUGE-L) | **{best_rouge}** |")
    lines.append(f"| Best Quality (BLEU) | **{best_bleu}** |")
    lines.append(f"| Most Per-Question Wins | **{most_wins}** |")
    lines.append(f"| Lower Final Loss | **{lower_loss}** |")
    lines.append(f"| Faster Training | **{faster}** |")

    # Side-by-side (all 80 questions)
    lines += [
        "",
        "## Side-by-Side Comparisons (All 80 Questions)",
        "",
    ]

    for q in data["questions"]:
        winner_label = {"instruct_ft": "Instruct", "base_ft": "Base", "tie": "Tie"}[q["winner"]]
        lines.append(f"### Q{q['id']}: [{q['category']}] {q['instruction'][:120]}")
        lines.append("")
        if q.get("input"):
            lines.append(f"**Context:** {q['input'][:200]}")
            lines.append("")
        lines.append(f"**Reference:** {q['reference'][:300]}")
        lines.append("")

        for key, ft_key, bl_key in [("Instruct", "instruct_ft", "instruct_baseline"),
                                     ("Base", "base_ft", "base_baseline")]:
            ft = q[ft_key]
            bl = q[bl_key]
            is_winner = (q["winner"] == ft_key)
            badge = " **[BEST]**" if is_winner else ""
            lines.append(
                f"**{key} Baseline** (ROUGE-L: {bl['rouge_l']:.3f}):"
            )
            lines.append(f"> {bl['prediction'][:300]}")
            lines.append("")
            lines.append(
                f"**{key} LoRA** (ROUGE-L: {ft['rouge_l']:.3f}, "
                f"{delta_str(ft['rouge_l'], bl['rouge_l'], '.3f')} vs baseline){badge}:"
            )
            lines.append(f"> {ft['prediction'][:300]}")
            lines.append("")

        lines.append("---")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

def save_html(data, output_path):
    agg = data["aggregated"]
    training = data["training"]
    wins = data["win_counts"]
    perp = data["perplexity"]
    questions = data["questions"]

    ic = RUNS["instruct"]["color"]
    bc = RUNS["base"]["color"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cross-Model Comparison — Instruct vs Base LoRA</title>
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
.q-header {{ display: flex; gap: 1rem; align-items: center; margin-bottom: 0.8rem; flex-wrap: wrap; }}
.q-num {{ color: var(--cyan); font-weight: bold; font-size: 1.1rem; }}
.q-cat {{ background: var(--bg); color: var(--dim); padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; }}
.instruction {{ color: var(--text); margin-bottom: 0.8rem; font-size: 0.95rem; }}
.context {{ color: var(--dim); font-size: 0.85rem; margin-bottom: 0.8rem; padding: 0.5rem; background: var(--bg); border-radius: 4px; }}
.reference {{ color: var(--dim); font-size: 0.85rem; margin-bottom: 1rem; padding: 0.5rem; background: var(--bg); border-radius: 4px; }}
.model-block {{ padding: 0.8rem 1rem; margin: 0.5rem 0; background: var(--bg); border-radius: 4px; }}
.model-label {{ font-weight: bold; font-size: 0.9rem; margin-bottom: 0.3rem; }}
.model-label .score {{ font-weight: normal; color: var(--dim); font-size: 0.8rem; margin-left: 0.5rem; }}
.prediction {{ font-size: 0.85rem; color: var(--text); white-space: pre-wrap; word-break: break-word; }}
.badge {{ background: var(--green); color: var(--bg); padding: 1px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: bold; margin-left: 0.5rem; }}

footer {{ margin-top: 3rem; color: var(--dim); font-size: 0.8rem; border-top: 1px solid var(--border); padding-top: 1rem; }}
</style>
</head>
<body>

<h1>Cross-Model Comparison</h1>
<p class="subtitle">Llama-3.1-8B-Instruct vs Llama-3.1-8B (Base) — LoRA Fine-Tuning on Dolly-15K — 80 Questions</p>

<h2>Training Performance</h2>
<table>
<tr><th>Metric</th><th style="color:{ic}">Instruct (8B-Instruct)</th><th style="color:{bc}">Base (8B)</th></tr>
<tr><td>Total Time</td><td>{fmt_time(training['instruct']['total_time_sec'])}</td><td>{fmt_time(training['base']['total_time_sec'])}</td></tr>
<tr><td>Avg Step Time</td><td>{training['instruct']['avg_step_time']:.2f}s</td><td>{training['base']['avg_step_time']:.2f}s</td></tr>
<tr><td>Peak GPU Memory</td><td>{training['instruct']['peak_gpu_memory_gb']:.1f} GB</td><td>{training['base']['peak_gpu_memory_gb']:.1f} GB</td></tr>
<tr><td>Final Loss</td><td>{training['instruct']['final_loss']:.4f}</td><td>{training['base']['final_loss']:.4f}</td></tr>
<tr><td>Tokens/sec</td><td>{training['instruct']['tokens_per_sec']:,.1f}</td><td>{training['base']['tokens_per_sec']:,.1f}</td></tr>
</table>
"""

    # Perplexity
    bp = perp.get("base", {})
    ip = perp.get("instruct", {})
    if bp:
        html += """
<h2>Perplexity</h2>
<table>
<tr><th>Metric</th><th style="color:{ic}">Instruct</th><th style="color:{bc}">Base</th></tr>
<tr><td>Baseline Perplexity</td><td>{ib_ppl}</td><td>{bb_ppl}</td></tr>
<tr><td>Finetuned Perplexity</td><td>{ift_ppl}</td><td>{bft_ppl}</td></tr>
<tr><td>Improvement</td><td>{i_imp}</td><td>{b_imp}</td></tr>
</table>
""".format(
            ic=ic, bc=bc,
            ib_ppl=f"{ip['baseline_perplexity']:.4f}" if ip else "n/a",
            bb_ppl=f"{bp['baseline_perplexity']:.4f}" if bp else "n/a",
            ift_ppl=f"{ip['finetuned_perplexity']:.4f}" if ip else "n/a",
            bft_ppl=f"{bp['finetuned_perplexity']:.4f}" if bp else "n/a",
            i_imp=f"{ip['improvement_pct']:.1f}%" if ip else "n/a",
            b_imp=f"{bp['improvement_pct']:.1f}%" if bp else "n/a",
        )

    # Evaluation metrics
    html += f"""
<h2>Evaluation Metrics (Finetuned)</h2>
<table>
<tr><th>Metric</th><th style="color:{ic}">Instruct LoRA</th><th style="color:{bc}">Base LoRA</th></tr>
<tr><td>ROUGE-L (mean)</td><td>{agg['instruct_ft']['rouge_l_mean']:.4f}</td><td>{agg['base_ft']['rouge_l_mean']:.4f} {delta_html(agg['base_ft']['rouge_l_mean'], agg['instruct_ft']['rouge_l_mean'])}</td></tr>
<tr><td>ROUGE-L (median)</td><td>{agg['instruct_ft']['rouge_l_median']:.4f}</td><td>{agg['base_ft']['rouge_l_median']:.4f} {delta_html(agg['base_ft']['rouge_l_median'], agg['instruct_ft']['rouge_l_median'])}</td></tr>
<tr><td>BLEU (mean)</td><td>{agg['instruct_ft']['bleu_mean']:.4f}</td><td>{agg['base_ft']['bleu_mean']:.4f} {delta_html(agg['base_ft']['bleu_mean'], agg['instruct_ft']['bleu_mean'])}</td></tr>
<tr><td>BLEU (median)</td><td>{agg['instruct_ft']['bleu_median']:.4f}</td><td>{agg['base_ft']['bleu_median']:.4f} {delta_html(agg['base_ft']['bleu_median'], agg['instruct_ft']['bleu_median'])}</td></tr>
<tr class="highlight"><td><strong>Best Answer Wins</strong></td>
<td style="color:{ic};font-weight:bold">{wins['instruct_ft']} ({wins['instruct_ft']/80*100:.0f}%)</td>
<td style="color:{bc};font-weight:bold">{wins['base_ft']} ({wins['base_ft']/80*100:.0f}%)</td></tr>
</table>
"""

    # Improvement from own baseline
    html += f"""
<h2>Training Improvement (Delta from Own Baseline)</h2>
<table>
<tr><th>Metric</th><th style="color:var(--dim)">Instruct Baseline</th><th style="color:{ic}">Instruct FT</th><th style="color:var(--dim)">Base Baseline</th><th style="color:{bc}">Base FT</th></tr>
<tr><td>ROUGE-L (mean)</td>
<td>{agg['instruct_baseline']['rouge_l_mean']:.4f}</td>
<td>{agg['instruct_ft']['rouge_l_mean']:.4f} {delta_html(agg['instruct_ft']['rouge_l_mean'], agg['instruct_baseline']['rouge_l_mean'])}</td>
<td>{agg['base_baseline']['rouge_l_mean']:.4f}</td>
<td>{agg['base_ft']['rouge_l_mean']:.4f} {delta_html(agg['base_ft']['rouge_l_mean'], agg['base_baseline']['rouge_l_mean'])}</td></tr>
<tr><td>BLEU (mean)</td>
<td>{agg['instruct_baseline']['bleu_mean']:.4f}</td>
<td>{agg['instruct_ft']['bleu_mean']:.4f} {delta_html(agg['instruct_ft']['bleu_mean'], agg['instruct_baseline']['bleu_mean'])}</td>
<td>{agg['base_baseline']['bleu_mean']:.4f}</td>
<td>{agg['base_ft']['bleu_mean']:.4f} {delta_html(agg['base_ft']['bleu_mean'], agg['base_baseline']['bleu_mean'])}</td></tr>
</table>
"""

    # Category breakdown
    html += f"""
<h2>ROUGE-L by Category (Finetuned)</h2>
<table>
<tr><th>Category</th><th style="color:{ic}">Instruct LoRA</th><th style="color:{bc}">Base LoRA</th><th>Winner</th></tr>
"""
    for cat in sorted(data["by_category"].keys()):
        cd = data["by_category"][cat]
        ir = cd["instruct_ft"]["rouge_l_mean"]
        br = cd["base_ft"]["rouge_l_mean"]
        winner = "Instruct" if data["category_winners"][cat] == "instruct_ft" else "Base"
        wcolor = ic if winner == "Instruct" else bc
        html += f'<tr><td>{cat}</td><td>{ir:.4f}</td><td>{br:.4f}</td><td style="color:{wcolor};font-weight:bold">{winner}</td></tr>\n'
    html += "</table>\n"

    # Verdict
    best_rouge = "Instruct" if agg["instruct_ft"]["rouge_l_mean"] > agg["base_ft"]["rouge_l_mean"] else "Base"
    best_bleu = "Instruct" if agg["instruct_ft"]["bleu_mean"] > agg["base_ft"]["bleu_mean"] else "Base"
    most_wins_label = "Instruct" if wins["instruct_ft"] > wins["base_ft"] else ("Base" if wins["base_ft"] > wins["instruct_ft"] else "Tie")
    lower_loss = "Instruct" if training["instruct"]["final_loss"] < training["base"]["final_loss"] else "Base"

    html += """
<h2>Verdict</h2>
<div class="verdict-grid">
"""
    for award, winner in [
        ("Best Quality (ROUGE-L)", best_rouge),
        ("Best Quality (BLEU)", best_bleu),
        ("Most Per-Question Wins", most_wins_label),
        ("Lower Final Loss", lower_loss),
    ]:
        wcolor = ic if winner == "Instruct" else (bc if winner == "Base" else "#f1fa8c")
        html += f'<div class="verdict-card"><div class="award">{award}</div><div class="winner" style="color:{wcolor}">{winner}</div></div>\n'
    html += "</div>\n"

    # Filter bar
    categories = sorted(set(q["category"] for q in questions))
    cat_options = "".join(f'<option value="{c}">{c}</option>' for c in categories)

    html += f"""
<h2>Side-by-Side Comparisons (All 80 Questions)</h2>

<div class="filter-bar">
  <label>Filter by category:</label>
  <select id="catFilter" onchange="filterQuestions()">
    <option value="all">All Categories</option>
    {cat_options}
  </select>
  <label>Filter by winner:</label>
  <select id="winFilter" onchange="filterQuestions()">
    <option value="all">All</option>
    <option value="instruct_ft">Instruct Wins</option>
    <option value="base_ft">Base Wins</option>
    <option value="tie">Ties</option>
  </select>
</div>
"""

    # Question cards
    for q in questions:
        winner_label = {"instruct_ft": "Instruct", "base_ft": "Base", "tie": "Tie"}[q["winner"]]
        wcolor = ic if q["winner"] == "instruct_ft" else (bc if q["winner"] == "base_ft" else "#f1fa8c")

        html += f'<div class="question-card" data-cat="{q["category"]}" data-winner="{q["winner"]}">\n'
        html += f'<div class="q-header"><span class="q-num">Q{q["id"]}</span><span class="q-cat">{q["category"]}</span>'
        html += f'<span style="color:{wcolor};font-size:0.8rem;font-weight:bold">Winner: {winner_label}</span></div>\n'
        html += f'<div class="instruction">{_escape(q["instruction"])}</div>\n'
        if q.get("input"):
            html += f'<div class="context">Context: {_escape(q["input"][:300])}</div>\n'
        html += f'<div class="reference">Reference: {_escape(q["reference"][:400])}</div>\n'

        # Four blocks: instruct baseline, instruct FT, base baseline, base FT
        for model_key, ft_key, bl_key, label, color in [
            ("instruct", "instruct_ft", "instruct_baseline", "Instruct", ic),
            ("base", "base_ft", "base_baseline", "Base", bc),
        ]:
            bl = q[bl_key]
            ft = q[ft_key]
            is_winner = (q["winner"] == ft_key)
            badge = '<span class="badge">BEST</span>' if is_winner else ""

            html += f'<div class="model-block">\n'
            html += f'<div class="model-label" style="color:{color}">{label} Baseline'
            html += f'<span class="score">ROUGE-L: {bl["rouge_l"]:.3f}</span></div>\n'
            html += f'<div class="prediction">{_escape(bl["prediction"][:500])}</div>\n'
            html += f'</div>\n'

            d = ft["rouge_l"] - bl["rouge_l"]
            dsign = "+" if d >= 0 else ""
            dcolor = "#50fa7b" if d > 0 else ("#ff5555" if d < 0 else "#8b949e")

            html += f'<div class="model-block">\n'
            html += f'<div class="model-label" style="color:{color}">{label} LoRA'
            html += f'<span class="score">ROUGE-L: {ft["rouge_l"]:.3f} (<span style="color:{dcolor}">{dsign}{d:.3f}</span> vs baseline)</span>{badge}</div>\n'
            html += f'<div class="prediction">{_escape(ft["prediction"][:500])}</div>\n'
            html += f'</div>\n'

        html += '</div>\n'

    # JS filter
    html += """
<script>
function filterQuestions() {
    var cat = document.getElementById('catFilter').value;
    var win = document.getElementById('winFilter').value;
    var cards = document.querySelectorAll('.question-card');
    cards.forEach(function(card) {
        var matchCat = (cat === 'all' || card.dataset.cat === cat);
        var matchWin = (win === 'all' || card.dataset.winner === win);
        card.classList.toggle('hidden', !(matchCat && matchWin));
    });
}
</script>

<footer>
Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """ | Fine-Tuning Benchmark GX10
</footer>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"  Saved: {output_path}")


def _escape(text):
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# JSON / CSV output
# ---------------------------------------------------------------------------

def save_json(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {output_path}")


def save_csv(data, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "category", "instruction",
            "instruct_baseline_rouge_l", "instruct_ft_rouge_l",
            "base_baseline_rouge_l", "base_ft_rouge_l",
            "instruct_baseline_bleu", "instruct_ft_bleu",
            "base_baseline_bleu", "base_ft_bleu",
            "winner",
        ])
        for q in data["questions"]:
            writer.writerow([
                q["id"], q["category"], q["instruction"][:120],
                f"{q['instruct_baseline']['rouge_l']:.4f}",
                f"{q['instruct_ft']['rouge_l']:.4f}",
                f"{q['base_baseline']['rouge_l']:.4f}",
                f"{q['base_ft']['rouge_l']:.4f}",
                f"{q['instruct_baseline']['bleu']:.4f}",
                f"{q['instruct_ft']['bleu']:.4f}",
                f"{q['base_baseline']['bleu']:.4f}",
                f"{q['base_ft']['bleu']:.4f}",
                {"instruct_ft": "instruct", "base_ft": "base", "tie": "tie"}[q["winner"]],
            ])
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Cross-Model Comparison: Instruct vs Base LoRA")
    print("=" * 50)

    # Load data
    print("\nLoading runs...")
    runs_data = {}
    for key in RUNS:
        print(f"  Loading {RUNS[key]['label']}...")
        runs_data[key] = load_run(key)

    # Compute comparison
    print("\nComputing comparison...")
    data = compute_comparison(runs_data)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {OUTPUT_DIR}/")

    save_markdown(data, OUTPUT_DIR / "cross_model_comparison.md")
    save_html(data, OUTPUT_DIR / "cross_model_comparison.html")
    save_json(data, OUTPUT_DIR / "cross_model_comparison.json")
    save_csv(data, OUTPUT_DIR / "cross_model_comparison.csv")

    # Print summary to terminal
    agg = data["aggregated"]
    wins = data["win_counts"]
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Instruct LoRA — ROUGE-L: {agg['instruct_ft']['rouge_l_mean']:.4f}, BLEU: {agg['instruct_ft']['bleu_mean']:.4f}, Wins: {wins['instruct_ft']}/80")
    print(f"  Base LoRA     — ROUGE-L: {agg['base_ft']['rouge_l_mean']:.4f}, BLEU: {agg['base_ft']['bleu_mean']:.4f}, Wins: {wins['base_ft']}/80")
    print(f"  Ties: {wins['tie']}")
    print()


if __name__ == "__main__":
    main()
