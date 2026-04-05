#!/usr/bin/env bash
#
# run_comparison.sh — Generate all comparison reports from completed training runs.
#
# This script takes the already-trained models' evaluation data and produces
# the complete comparison: training metrics table + cross-mode evaluation
# (Base Model vs LoRA vs QLoRA vs Full Fine-Tune) with all report formats.
#
# Usage:
#   ./run_comparison.sh                           # Default (./results)
#   ./run_comparison.sh --results-dir ./results   # Custom results dir
#
set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────

RESULTS_DIR="./results"

# ─── Parse arguments ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_comparison.sh [OPTIONS]"
            echo ""
            echo "Generates all comparison reports from completed training runs."
            echo "No retraining or re-inference needed — uses existing evaluation data."
            echo ""
            echo "Options:"
            echo "  --results-dir DIR   Results directory (default: ./results)"
            echo "  --help              Show this help"
            echo ""
            echo "Outputs:"
            echo "  Terminal            Rich tables (training + evaluation + verdict)"
            echo "  HTML report         Interactive dark-themed report for browser/video"
            echo "  Markdown report     For README, class reports, documentation"
            echo "  CSV                 Per-question metrics for spreadsheets"
            echo "  JSON                Full machine-readable data"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run ./run_comparison.sh --help for usage."
            exit 1
            ;;
    esac
done

# ─── Setup ───────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "================================================================"
echo "  FINE-TUNING BENCHMARK — COMPARISON REPORT GENERATOR"
echo "================================================================"
echo ""
echo "  Results dir: $RESULTS_DIR"
echo ""

# ─── Check results exist ────────────────────────────────────────────────────

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Count completed runs
RUNS=$(find "$RESULTS_DIR" -maxdepth 2 -name "summary.txt" 2>/dev/null | wc -l)
if [ "$RUNS" -eq 0 ]; then
    echo "ERROR: No completed benchmark runs found in $RESULTS_DIR"
    exit 1
fi

echo "  Found $RUNS completed run(s):"
for summary in $(find "$RESULTS_DIR" -maxdepth 2 -name "summary.txt" -exec grep -l "success" {} \; 2>/dev/null | sort); do
    run_dir=$(dirname "$summary")
    mode=$(basename "$run_dir" | sed 's/.*_\(lora\|qlora\|fullft\)_.*/\1/')
    echo "    - $(basename "$run_dir") ($mode)"
done
echo ""

# ─── Step 1: Training comparison ────────────────────────────────────────────

echo "================================================================"
echo "  STEP 1: TRAINING PERFORMANCE COMPARISON"
echo "================================================================"
echo ""

python3 -m benchmark_cuda compare --results-dir "$RESULTS_DIR"

echo ""

# ─── Step 2: Cross-mode evaluation ──────────────────────────────────────────

echo "================================================================"
echo "  STEP 2: CROSS-MODE EVALUATION (Base vs LoRA vs QLoRA vs Full FT)"
echo "================================================================"
echo ""

python3 -m benchmark_cuda cross-compare --results-dir "$RESULTS_DIR"

echo ""

# ─── Step 3: Training plots ─────────────────────────────────────────────────

echo "================================================================"
echo "  STEP 3: TRAINING PLOTS (Loss Curves, GPU Memory, Step Time)"
echo "================================================================"
echo ""

python3 -m benchmark_cuda plot --results-dir "$RESULTS_DIR"

echo ""

# ─── Step 4: LLM-as-Judge (optional) ────────────────────────────────────────

echo "================================================================"
echo "  STEP 4: LLM-AS-JUDGE EVALUATION (Claude API)"
echo "================================================================"
echo ""

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "  Anthropic API key found. Running LLM judge evaluation..."
    echo "  (This will make API calls — may take a few minutes)"
    echo ""
    python3 -m benchmark_cuda judge --results-dir "$RESULTS_DIR" || {
        echo ""
        echo "  WARNING: LLM judge evaluation failed (non-fatal)."
        echo "  Check your ANTHROPIC_API_KEY or API quota."
    }
else
    echo "  Skipped: ANTHROPIC_API_KEY not set."
    echo "  To enable: export ANTHROPIC_API_KEY=your-key-here"
    echo "  Then re-run: ./run_comparison.sh"
fi

echo ""
echo "================================================================"
echo "  DONE"
echo "================================================================"
echo ""
echo "  All reports saved to: $RESULTS_DIR/cross_comparison/"
echo ""
echo "  Reports:"
echo "    $RESULTS_DIR/cross_comparison/cross_comparison.html   (interactive HTML)"
echo "    $RESULTS_DIR/cross_comparison/cross_comparison.md     (markdown)"
echo "    $RESULTS_DIR/cross_comparison/cross_comparison.csv    (spreadsheet)"
echo ""
echo "  Plots:"
echo "    $RESULTS_DIR/cross_comparison/loss_curves.png"
echo "    $RESULTS_DIR/cross_comparison/gpu_memory.png"
echo "    $RESULTS_DIR/cross_comparison/step_time.png"
echo ""
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
echo "  LLM Judge:"
echo "    $RESULTS_DIR/cross_comparison/llm_judge_cross_compare.json"
fi
echo ""
