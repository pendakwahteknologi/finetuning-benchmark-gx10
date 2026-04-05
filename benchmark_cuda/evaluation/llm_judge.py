"""LLM-as-Judge evaluation using Claude API.

Sends model outputs to Claude for qualitative scoring on helpfulness,
accuracy, completeness, and conciseness.  Also provides a cross-compare
mode where all model variants are ranked head-to-head per question.
"""

import json
import os
import time
from typing import Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..utils.logging_utils import console

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # handled at call sites

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODE_LABELS = {
    "lora": "LoRA",
    "qlora": "QLoRA",
    "fullft": "Full Fine-Tune",
}

MODE_ORDER = ["lora", "qlora", "fullft"]

JUDGE_MODEL = "claude-sonnet-4-20250514"

# Retry / rate-limit
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0   # seconds, doubles each attempt
CALL_SLEEP = 0.5          # seconds between API calls

# ---------------------------------------------------------------------------
# Helpers (borrowed from cross_compare to avoid circular imports)
# ---------------------------------------------------------------------------

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


def _load_predictions(jsonl_path: str) -> list[dict]:
    """Load predictions as a list sorted by id."""
    preds = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    preds.sort(key=lambda r: r["id"])
    return preds


def _load_predictions_dict(jsonl_path: str) -> dict[int, dict]:
    """Load predictions keyed by question id."""
    preds = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                preds[rec["id"]] = rec
    return preds


def _ensure_client() -> "Anthropic":
    if Anthropic is None:
        console.print(
            "[error]The 'anthropic' package is not installed.[/error]\n"
            "Install it with:  pip install anthropic"
        )
        raise SystemExit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[error]ANTHROPIC_API_KEY environment variable is not set.[/error]\n"
            "Export it with:  export ANTHROPIC_API_KEY=sk-ant-..."
        )
        raise SystemExit(1)
    return Anthropic()


# ---------------------------------------------------------------------------
# 1. Judge a single answer
# ---------------------------------------------------------------------------

SINGLE_JUDGE_SYSTEM = """\
You are an expert evaluator for language model outputs. You will be given:
- An instruction (the question or task)
- Optional input context
- A reference answer (gold standard written by a human)
- A model prediction to evaluate
- The category of the question

Your job is to rate the model prediction on four dimensions, each scored 1-5:

**Helpfulness** (1-5): Does the prediction actually answer the question or \
complete the task? A score of 1 means it is off-topic or refuses to answer; \
5 means it fully addresses the user's intent.

**Accuracy** (1-5): Is the information in the prediction factually correct \
when compared to the reference answer? A score of 1 means mostly wrong; \
5 means all claims are accurate and consistent with the reference.

**Completeness** (1-5): Does the prediction cover all the key points \
present in the reference answer? A score of 1 means it misses almost \
everything; 5 means it covers all major points (and possibly more).

**Conciseness** (1-5): Is the prediction focused and free of unnecessary \
filler, repetition, or irrelevant tangents? A score of 1 means it is \
extremely verbose or padded; 5 means every sentence serves a purpose.

After scoring, compute an overall score as the average of the four scores \
(round to one decimal place).

Provide a brief reasoning (2-4 sentences) explaining your scores.

Respond with ONLY a JSON object in this exact format (no markdown fences):
{"helpfulness": <int>, "accuracy": <int>, "completeness": <int>, "conciseness": <int>, "overall": <float>, "reasoning": "<string>"}
"""


def judge_single_answer(
    instruction: str,
    input_context: str,
    reference: str,
    prediction: str,
    category: str,
    client: Optional["Anthropic"] = None,
) -> dict:
    """Call Claude to score a single model prediction.

    Returns dict with keys: helpfulness, accuracy, completeness,
    conciseness, overall, reasoning.
    """
    if client is None:
        client = _ensure_client()

    user_msg = (
        f"**Category:** {category}\n\n"
        f"**Instruction:** {instruction}\n\n"
    )
    if input_context:
        user_msg += f"**Input context:** {input_context}\n\n"
    user_msg += (
        f"**Reference answer:**\n{reference}\n\n"
        f"**Model prediction:**\n{prediction}"
    )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=512,
                system=SINGLE_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if Claude adds them despite instructions
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[: raw.rfind("```")]
                raw = raw.strip()
            result = json.loads(raw)
            # Validate keys
            for key in ("helpfulness", "accuracy", "completeness", "conciseness"):
                result[key] = int(result[key])
            result["overall"] = round(float(result["overall"]), 1)
            result["reasoning"] = str(result.get("reasoning", ""))
            return result

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                console.print(
                    f"  [warning]Retry {attempt + 1}/{MAX_RETRIES} "
                    f"after error: {e}. Waiting {delay:.0f}s...[/warning]"
                )
                time.sleep(delay)

    # All retries failed -- return a default with error info
    console.print(f"  [error]All {MAX_RETRIES} attempts failed: {last_error}[/error]")
    return {
        "helpfulness": 0,
        "accuracy": 0,
        "completeness": 0,
        "conciseness": 0,
        "overall": 0.0,
        "reasoning": f"JUDGE ERROR: {last_error}",
    }


# ---------------------------------------------------------------------------
# 2. Judge all predictions in a JSONL file
# ---------------------------------------------------------------------------

def judge_all_predictions(
    predictions_jsonl_path: str,
    max_questions: int = 80,
    client: Optional["Anthropic"] = None,
) -> list[dict]:
    """Load predictions from JSONL and judge each one.

    Returns list of dicts, each containing the question id, category,
    and all judge scores.
    """
    if client is None:
        client = _ensure_client()

    preds = _load_predictions(predictions_jsonl_path)[:max_questions]
    results = []

    console.print(
        f"  Judging {len(preds)} predictions from "
        f"[info]{os.path.basename(predictions_jsonl_path)}[/info]..."
    )

    for i, rec in enumerate(preds):
        scores = judge_single_answer(
            instruction=rec["instruction"],
            input_context=rec.get("input", ""),
            reference=rec["reference_output"],
            prediction=rec["prediction"],
            category=rec["category"],
            client=client,
        )
        results.append({
            "id": rec["id"],
            "category": rec["category"],
            "instruction": rec["instruction"],
            **scores,
        })

        # Progress update every 10 questions
        if (i + 1) % 10 == 0 or (i + 1) == len(preds):
            console.print(
                f"    [{i + 1}/{len(preds)}] "
                f"last overall={scores['overall']}"
            )

        # Rate-limit sleep
        time.sleep(CALL_SLEEP)

    return results


# ---------------------------------------------------------------------------
# 3. Full evaluation: judge baseline & finetuned per mode
# ---------------------------------------------------------------------------

def _aggregate_scores(judged: list[dict]) -> dict:
    """Compute mean scores overall and per category."""
    dimensions = ["helpfulness", "accuracy", "completeness", "conciseness", "overall"]

    # Overall means
    overall = {}
    for dim in dimensions:
        vals = [r[dim] for r in judged if r[dim] > 0]
        overall[dim] = round(sum(vals) / len(vals), 2) if vals else 0.0

    # Per category
    by_cat: dict[str, list[dict]] = {}
    for r in judged:
        by_cat.setdefault(r["category"], []).append(r)

    per_category = {}
    for cat, items in sorted(by_cat.items()):
        cat_scores = {}
        for dim in dimensions:
            vals = [r[dim] for r in items if r[dim] > 0]
            cat_scores[dim] = round(sum(vals) / len(vals), 2) if vals else 0.0
        per_category[cat] = cat_scores

    return {"overall": overall, "per_category": per_category}


def run_llm_judge_evaluation(results_dir: str) -> dict:
    """Judge baseline and finetuned predictions for every available mode.

    Optimization: the baseline model is the same across modes, so we only
    judge it once (from the first available mode) and reuse the scores.

    Saves results to results/cross_comparison/llm_judge_results.json.
    """
    client = _ensure_client()
    output = {"modes": {}, "baseline": None}

    baseline_judged = None

    for mode in MODE_ORDER:
        run_dir = _find_best_run(results_dir, mode)
        if run_dir is None:
            console.print(
                f"[warning]No successful run found for {MODE_LABELS[mode]}, "
                f"skipping.[/warning]"
            )
            continue

        eval_dir = os.path.join(run_dir, "evaluation")
        baseline_path = os.path.join(eval_dir, "baseline_predictions.jsonl")
        finetuned_path = os.path.join(eval_dir, "finetuned_predictions.jsonl")

        if not os.path.isfile(baseline_path) or not os.path.isfile(finetuned_path):
            console.print(
                f"[warning]Missing prediction files for {MODE_LABELS[mode]}, "
                f"skipping.[/warning]"
            )
            continue

        console.print(Panel(
            f"Evaluating [header]{MODE_LABELS[mode]}[/header] "
            f"from {os.path.basename(run_dir)}",
            style="info",
        ))

        # Judge baseline only once
        if baseline_judged is None:
            console.print("[info]Judging baseline (base model) predictions...[/info]")
            baseline_judged = judge_all_predictions(
                baseline_path, client=client
            )
            output["baseline"] = {
                "details": baseline_judged,
                "aggregated": _aggregate_scores(baseline_judged),
            }

        # Judge finetuned
        console.print(
            f"[info]Judging finetuned ({MODE_LABELS[mode]}) predictions...[/info]"
        )
        finetuned_judged = judge_all_predictions(
            finetuned_path, client=client
        )

        output["modes"][mode] = {
            "run_dir": run_dir,
            "details": finetuned_judged,
            "aggregated": _aggregate_scores(finetuned_judged),
        }

    # Save
    out_dir = os.path.join(results_dir, "cross_comparison")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "llm_judge_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\n[success]LLM judge results saved to {out_path}[/success]")

    print_judge_summary(output)
    return output


# ---------------------------------------------------------------------------
# 4. Cross-compare: head-to-head ranking per question
# ---------------------------------------------------------------------------

CROSS_COMPARE_SYSTEM = """\
You are an expert evaluator comparing language model outputs. You will be \
given one question with its reference answer and multiple model responses \
(labeled by model variant). Your task is to:

1. Rank all model variants from best to worst for this question.
2. Pick the single best response and explain why in 2-3 sentences.
3. Note any significant quality differences between responses.

Respond with ONLY a JSON object in this exact format (no markdown fences):
{
  "ranking": ["<best_label>", "<second_label>", ...],
  "best": "<best_label>",
  "explanation": "<string>",
  "notes": "<string>"
}
"""


def run_llm_judge_cross_compare(results_dir: str) -> dict:
    """For each question, send all model responses to Claude in one call.

    Asks Claude to rank: base, LoRA, QLoRA, Full Fine-Tune.
    Saves to results/cross_comparison/llm_judge_cross_compare.json.
    """
    client = _ensure_client()

    # Load predictions from all modes
    all_preds: dict[str, dict[int, dict]] = {}  # label -> {id -> rec}
    mode_sources = {}

    for mode in MODE_ORDER:
        run_dir = _find_best_run(results_dir, mode)
        if run_dir is None:
            continue
        eval_dir = os.path.join(run_dir, "evaluation")
        finetuned_path = os.path.join(eval_dir, "finetuned_predictions.jsonl")
        if os.path.isfile(finetuned_path):
            all_preds[MODE_LABELS[mode]] = _load_predictions_dict(finetuned_path)
            mode_sources[mode] = run_dir

    # Load baseline from the first available mode
    baseline_loaded = False
    for mode in MODE_ORDER:
        run_dir = _find_best_run(results_dir, mode)
        if run_dir is None:
            continue
        baseline_path = os.path.join(run_dir, "evaluation", "baseline_predictions.jsonl")
        if os.path.isfile(baseline_path):
            all_preds["Base Model"] = _load_predictions_dict(baseline_path)
            baseline_loaded = True
            break

    if not baseline_loaded or len(all_preds) < 2:
        console.print("[error]Not enough data for cross-comparison.[/error]")
        return {}

    # Determine question ids (from baseline)
    question_ids = sorted(all_preds["Base Model"].keys())

    # Ordered labels for consistent presentation
    labels = ["Base Model"] + [
        MODE_LABELS[m] for m in MODE_ORDER if MODE_LABELS[m] in all_preds
    ]

    console.print(Panel(
        f"Cross-comparing {len(labels)} model variants across "
        f"{len(question_ids)} questions",
        style="header",
    ))

    comparisons = []
    win_counts: dict[str, int] = {label: 0 for label in labels}

    for i, qid in enumerate(question_ids):
        # Build the prompt with all responses
        base_rec = all_preds["Base Model"].get(qid)
        if base_rec is None:
            continue

        user_msg = (
            f"**Category:** {base_rec['category']}\n\n"
            f"**Instruction:** {base_rec['instruction']}\n\n"
        )
        if base_rec.get("input"):
            user_msg += f"**Input context:** {base_rec['input']}\n\n"
        user_msg += f"**Reference answer:**\n{base_rec['reference_output']}\n\n"
        user_msg += "---\n\n**Model responses to compare:**\n\n"

        available_labels = []
        for label in labels:
            rec = all_preds.get(label, {}).get(qid)
            if rec is not None:
                available_labels.append(label)
                user_msg += f"**[{label}]:**\n{rec['prediction']}\n\n"

        if len(available_labels) < 2:
            continue

        # Call Claude
        last_error = None
        result = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.messages.create(
                    model=JUDGE_MODEL,
                    max_tokens=512,
                    system=CROSS_COMPARE_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = response.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1]
                    if raw.endswith("```"):
                        raw = raw[: raw.rfind("```")]
                    raw = raw.strip()
                result = json.loads(raw)
                break
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)

        if result is None:
            console.print(
                f"  [error]Q{qid}: all retries failed: {last_error}[/error]"
            )
            comparisons.append({
                "id": qid,
                "category": base_rec["category"],
                "error": str(last_error),
            })
        else:
            best = result.get("best", "")
            if best in win_counts:
                win_counts[best] += 1
            comparisons.append({
                "id": qid,
                "category": base_rec["category"],
                "ranking": result.get("ranking", []),
                "best": best,
                "explanation": result.get("explanation", ""),
                "notes": result.get("notes", ""),
            })

        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == len(question_ids):
            console.print(f"  [{i + 1}/{len(question_ids)}] completed")

        time.sleep(CALL_SLEEP)

    # Summary
    output = {
        "labels": labels,
        "win_counts": win_counts,
        "comparisons": comparisons,
        "total_questions": len(question_ids),
    }

    out_dir = os.path.join(results_dir, "cross_comparison")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "llm_judge_cross_compare.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\n[success]Cross-compare results saved to {out_path}[/success]")

    # Print win summary
    _print_cross_compare_summary(output)
    return output


# ---------------------------------------------------------------------------
# 5. Rich terminal summaries
# ---------------------------------------------------------------------------

def _score_style(score: float) -> str:
    """Return a Rich style string based on score value."""
    if score >= 4.5:
        return "bold green"
    elif score >= 3.5:
        return "green"
    elif score >= 2.5:
        return "yellow"
    elif score >= 1.5:
        return "red"
    else:
        return "bold red"


def print_judge_summary(results: dict) -> None:
    """Print a Rich table summarizing LLM judge scores per mode."""
    if not results or not results.get("modes"):
        console.print("[warning]No judge results to display.[/warning]")
        return

    dimensions = ["helpfulness", "accuracy", "completeness", "conciseness", "overall"]

    # --- Overall scores table ---
    table = Table(
        title="LLM Judge Scores (Claude Evaluation)",
        show_header=True,
        header_style="bold cyan",
        title_style="bold magenta",
        padding=(0, 1),
    )
    table.add_column("Model", style="bold")
    for dim in dimensions:
        table.add_column(dim.capitalize(), justify="center")

    # Baseline row
    if results.get("baseline"):
        agg = results["baseline"]["aggregated"]["overall"]
        row = ["Base Model"]
        for dim in dimensions:
            val = agg.get(dim, 0.0)
            row.append(Text(f"{val:.1f}", style=_score_style(val)))
        table.add_row(*row)

    # Mode rows
    for mode in MODE_ORDER:
        if mode not in results["modes"]:
            continue
        agg = results["modes"][mode]["aggregated"]["overall"]
        row = [MODE_LABELS[mode]]
        for dim in dimensions:
            val = agg.get(dim, 0.0)
            row.append(Text(f"{val:.1f}", style=_score_style(val)))
        table.add_row(*row)

    console.print()
    console.print(table)

    # --- Per-category breakdown ---
    categories = set()
    if results.get("baseline"):
        categories.update(results["baseline"]["aggregated"]["per_category"].keys())
    for mode in MODE_ORDER:
        if mode in results["modes"]:
            categories.update(
                results["modes"][mode]["aggregated"]["per_category"].keys()
            )

    if categories:
        cat_table = Table(
            title="LLM Judge - Overall Score by Category",
            show_header=True,
            header_style="bold cyan",
            title_style="bold magenta",
            padding=(0, 1),
        )
        cat_table.add_column("Category", style="bold")
        cat_table.add_column("Base", justify="center")
        for mode in MODE_ORDER:
            if mode in results["modes"]:
                cat_table.add_column(MODE_LABELS[mode], justify="center")

        for cat in sorted(categories):
            row = [cat]
            # Baseline
            if results.get("baseline"):
                val = results["baseline"]["aggregated"]["per_category"].get(
                    cat, {}
                ).get("overall", 0.0)
                row.append(Text(f"{val:.1f}", style=_score_style(val)))
            else:
                row.append("-")
            # Modes
            for mode in MODE_ORDER:
                if mode not in results["modes"]:
                    continue
                val = results["modes"][mode]["aggregated"]["per_category"].get(
                    cat, {}
                ).get("overall", 0.0)
                row.append(Text(f"{val:.1f}", style=_score_style(val)))
            cat_table.add_row(*row)

        console.print()
        console.print(cat_table)


def _print_cross_compare_summary(results: dict) -> None:
    """Print head-to-head win counts."""
    if not results or not results.get("win_counts"):
        return

    table = Table(
        title="LLM Judge - Head-to-Head Wins",
        show_header=True,
        header_style="bold cyan",
        title_style="bold magenta",
        padding=(0, 1),
    )
    table.add_column("Model", style="bold")
    table.add_column("Wins", justify="center")
    table.add_column("Win %", justify="center")

    total = results.get("total_questions", 1)
    # Sort by wins descending
    sorted_wins = sorted(
        results["win_counts"].items(), key=lambda x: x[1], reverse=True
    )
    for label, wins in sorted_wins:
        pct = (wins / total * 100) if total > 0 else 0
        style = _score_style(5.0 if wins == sorted_wins[0][1] else 2.5)
        table.add_row(
            label,
            Text(str(wins), style=style),
            Text(f"{pct:.1f}%", style=style),
        )

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# CLI entry point (for standalone usage)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM judge evaluation")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to results directory",
    )
    parser.add_argument(
        "--mode",
        choices=["evaluate", "cross-compare", "both"],
        default="both",
        help="Which evaluation to run",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)

    if args.mode in ("evaluate", "both"):
        console.print(Panel(
            "Running LLM Judge Evaluation",
            style="header",
        ))
        run_llm_judge_evaluation(results_dir)

    if args.mode in ("cross-compare", "both"):
        console.print(Panel(
            "Running LLM Judge Cross-Compare",
            style="header",
        ))
        run_llm_judge_cross_compare(results_dir)
