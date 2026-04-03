"""Evaluation metrics: ROUGE-L, BLEU, exact match, per-category."""

import re
from collections import defaultdict
from typing import Optional


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def normalized_string_match(prediction: str, reference: str) -> float:
    """Fuzzy normalized containment check."""
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not ref:
        return 1.0 if not pred else 0.0
    if ref in pred or pred in ref:
        return 1.0
    # Token overlap
    pred_tokens = set(pred.split())
    ref_tokens = set(ref.split())
    if not ref_tokens:
        return 0.0
    overlap = len(pred_tokens & ref_tokens)
    return overlap / len(ref_tokens)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores["rougeL"].fmeasure
    except ImportError:
        # Fallback: LCS-based ROUGE-L
        return _lcs_rouge_l(prediction, reference)


def _lcs_rouge_l(prediction: str, reference: str) -> float:
    """Fallback LCS-based ROUGE-L."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not ref_tokens or not pred_tokens:
        return 0.0

    # LCS length
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_bleu(prediction: str, reference: str) -> float:
    """Compute sentence-level BLEU score."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = normalize_text(reference).split()
        pred_tokens = normalize_text(prediction).split()
        if not ref_tokens or not pred_tokens:
            return 0.0
        smooth = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
    except ImportError:
        return 0.0


def compute_all_metrics(prediction: str, reference: str) -> dict:
    """Compute all metrics for a single prediction-reference pair."""
    return {
        "rouge_l": compute_rouge_l(prediction, reference),
        "bleu": compute_bleu(prediction, reference),
        "exact_match": exact_match(prediction, reference),
        "normalized_match": normalized_string_match(prediction, reference),
        "prediction_length": len(prediction.split()),
        "reference_length": len(reference.split()),
    }


def aggregate_metrics(results: list, group_by: str = "category") -> dict:
    """Aggregate metrics overall and per-group."""
    overall = defaultdict(list)
    grouped = defaultdict(lambda: defaultdict(list))

    metric_keys = ["rouge_l", "bleu", "exact_match", "normalized_match",
                    "prediction_length", "latency_sec"]

    for r in results:
        for k in metric_keys:
            if k in r and r[k] is not None:
                overall[k].append(r[k])
                if group_by in r:
                    grouped[r[group_by]][k].append(r[k])

    def _summarize(d: dict) -> dict:
        summary = {}
        for k, vals in d.items():
            if vals:
                summary[k] = {
                    "mean": sum(vals) / len(vals),
                    "count": len(vals),
                }
        return summary

    return {
        "overall": _summarize(overall),
        "total_samples": len(results),
        "by_category": {cat: _summarize(metrics) for cat, metrics in grouped.items()},
    }
