#!/usr/bin/env bash
#
# run_benchmark.sh — Complete fine-tuning benchmark: train + evaluate + report.
#
# This is the single script that does everything from zero to final report:
#
#   Phase 1: Pre-flight checks (Python, CUDA, HuggingFace, dependencies)
#   Phase 2: Data preparation (dataset splits + 80 evaluation questions)
#   Phase 3: Training (LoRA → QLoRA → Full Fine-Tune)
#            Each mode includes: training + baseline vs fine-tuned evaluation + perplexity
#   Phase 4: Training performance comparison table
#   Phase 5: Cross-mode evaluation (Base vs LoRA vs QLoRA vs Full FT, 80 questions)
#   Phase 6: Training plots (loss curves, GPU memory, step time)
#   Phase 7: LLM-as-Judge evaluation via Claude API (if ANTHROPIC_API_KEY is set)
#   Phase 8: Final report summary with all output locations
#
# Usage:
#   ./run_benchmark.sh                          # Full run (500 steps, all modes)
#   ./run_benchmark.sh --epochs 1               # Train for 1 full epoch
#   ./run_benchmark.sh --dry-run                # Quick sanity check (5 steps)
#   ./run_benchmark.sh --steps 100              # Custom step count
#   ./run_benchmark.sh --machine-label rtx5090  # Different machine
#   ./run_benchmark.sh --modes "lora fullft"    # Only specific modes
#   ./run_benchmark.sh --skip-training          # Skip training, run reports only
#
# Background (safe to close terminal / SSH):
#   nohup ./run_benchmark.sh > benchmark.log 2>&1 &
#   tail -f benchmark.log
#
set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────

MACHINE_LABEL="gx10"
MAX_STEPS=500
EPOCHS=""
MODES="lora qlora fullft"
DRY_RUN=false
SKIP_TRAINING=false
SKIP_JUDGE=false
RESULTS_DIR="./results"
EXTRA_ARGS=""

# ─── Parse arguments ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --machine-label)
            MACHINE_LABEL="$2"
            shift 2
            ;;
        --modes)
            MODES="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-judge)
            SKIP_JUDGE=true
            shift
            ;;
        --help|-h)
            cat <<'HELPTEXT'
Usage: ./run_benchmark.sh [OPTIONS]

Complete fine-tuning benchmark: train all modes, evaluate, and generate reports.

Training options:
  --steps N               Number of training steps (default: 500)
  --epochs N              Train for N epochs (alternative to --steps)
  --machine-label LABEL   Machine identifier (default: gx10)
  --modes "m1 m2 ..."     Modes to run (default: "lora qlora fullft")
  --dry-run               Quick test (5 steps, skip evaluation)

Report options:
  --skip-training         Skip training, only generate reports from existing results
  --skip-judge            Skip LLM-as-judge evaluation (saves API costs)
  --results-dir DIR       Results directory (default: ./results)

Examples:
  ./run_benchmark.sh                                  # Full benchmark (500 steps)
  ./run_benchmark.sh --epochs 1                       # Train for 1 full epoch
  ./run_benchmark.sh --dry-run                        # Quick sanity check
  ./run_benchmark.sh --steps 1000 --modes "lora"      # Custom run
  ./run_benchmark.sh --skip-training                  # Reports only (no training)
  ./run_benchmark.sh --skip-training --skip-judge     # Reports without API calls
  nohup ./run_benchmark.sh > benchmark.log 2>&1 &     # Background execution

Phases:
  1. Pre-flight checks     — Python, CUDA, HuggingFace auth, dependencies
  2. Data preparation       — Dataset splits + 80 curated evaluation questions
  3. Training               — LoRA → QLoRA → Full Fine-Tune (each with evaluation)
  4. Training comparison    — Performance metrics table across all modes
  5. Cross-mode evaluation  — Base vs LoRA vs QLoRA vs Full FT (80 questions)
  6. Training plots         — Loss curves, GPU memory, step time (PNG + SVG)
  7. LLM-as-Judge           — Claude API rates every answer (if API key set)
  8. Final report           — Summary with all output file locations

Output:
  results/<machine>_<mode>_<timestamp>/     — Per-run training + evaluation data
  results/cross_comparison/                  — All comparison reports
    cross_comparison.html                    — Interactive HTML report
    cross_comparison.md                      — Markdown report
    cross_comparison.csv                     — Per-question metrics
    loss_curves.png                          — Training loss plot
    gpu_memory.png                           — GPU memory usage plot
    step_time.png                            — Step time plot
    llm_judge_cross_compare.json             — LLM judge results
HELPTEXT
            exit 0
            ;;
        --model|--micro-batch-size|--grad-accum|--learning-rate|--lr|--dtype|--lora-r|--lora-alpha|--seed|--logging-steps|--eval-steps|--seq-len|--warmup-steps)
            EXTRA_ARGS="$EXTRA_ARGS $1 $2"
            shift 2
            ;;
        --gradient-checkpointing|--skip-eval)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        *)
            echo "Unknown option: $1 (use --help for usage)"
            exit 1
            ;;
    esac
done

# ─── Setup ───────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/benchmark_${TIMESTAMP}.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$MASTER_LOG"
}

separator() {
    echo "" | tee -a "$MASTER_LOG"
    echo "================================================================" | tee -a "$MASTER_LOG"
    echo "  $1" | tee -a "$MASTER_LOG"
    echo "================================================================" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"
}

phase_time() {
    local start=$1
    local end=$(date +%s)
    local dur=$(( end - start ))
    local m=$(( dur / 60 ))
    local s=$(( dur % 60 ))
    echo "${m}m ${s}s"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

separator "PHASE 1: PRE-FLIGHT CHECKS"

log "Machine label:  $MACHINE_LABEL"
if [ -n "$EPOCHS" ]; then
    log "Epochs:         $EPOCHS"
else
    log "Max steps:      $MAX_STEPS"
fi
log "Modes:          $MODES"
log "Results dir:    $RESULTS_DIR"
log "Skip training:  $SKIP_TRAINING"
log "Skip judge:     $SKIP_JUDGE"
log "Dry run:        $DRY_RUN"
log "Master log:     $MASTER_LOG"
log ""

# Python
if ! command -v python3 &>/dev/null; then
    log "ERROR: python3 not found"
    exit 1
fi
log "Python:         $(python3 --version 2>&1)"

# CUDA
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    GPU_MEM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB')" 2>/dev/null)
    log "CUDA:           $CUDA_VER"
    log "GPU:            $GPU_NAME ($GPU_MEM)"
else
    log "ERROR: CUDA is not available"
    exit 1
fi

# HuggingFace
if python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" &>/dev/null; then
    HF_USER=$(python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null)
    log "HuggingFace:    $HF_USER"
else
    log "WARNING: HuggingFace not authenticated (run: huggingface-cli login)"
fi

# Dependencies
if ! python3 -c "import transformers, peft, datasets, rich, typer, matplotlib" 2>/dev/null; then
    log "ERROR: Missing dependencies. Run: pip install -r benchmark_cuda/requirements.txt"
    exit 1
fi
log "Dependencies:   OK"

# Anthropic API (for LLM judge)
if [ -n "${ANTHROPIC_API_KEY:-}" ] && [ "$SKIP_JUDGE" = false ]; then
    log "Anthropic API:  available (LLM judge enabled)"
    HAS_JUDGE=true
else
    if [ "$SKIP_JUDGE" = true ]; then
        log "Anthropic API:  skipped (--skip-judge)"
    else
        log "Anthropic API:  not set (LLM judge will be skipped)"
    fi
    HAS_JUDGE=false
fi

log ""
log "All checks passed."

TOTAL_START=$(date +%s)

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

if [ "$SKIP_TRAINING" = false ]; then
    separator "PHASE 2: DATA PREPARATION"
    PHASE_START=$(date +%s)

    log "Preparing dataset and evaluation questions..."
    python3 -m benchmark_cuda prepare 2>&1 | tee -a "$MASTER_LOG"

    log "Data preparation complete. ($(phase_time $PHASE_START))"
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

FAILED_MODES=""
COMPLETED_MODES=""

if [ "$SKIP_TRAINING" = false ]; then
    for MODE in $MODES; do
        separator "PHASE 3: TRAINING — ${MODE^^}"

        MODE_LOG="$LOG_DIR/${MODE}_${TIMESTAMP}.log"
        MODE_START=$(date +%s)

        log "Starting $MODE training..."

        # Build command
        CMD="python3 -m benchmark_cuda run --mode $MODE --machine-label $MACHINE_LABEL --output-dir $RESULTS_DIR"
        if [ -n "$EPOCHS" ]; then
            CMD="$CMD --epochs $EPOCHS"
        else
            CMD="$CMD --max-steps $MAX_STEPS"
        fi
        if [ "$DRY_RUN" = true ]; then
            CMD="$CMD --dry-run"
        fi
        if [ -n "$EXTRA_ARGS" ]; then
            CMD="$CMD $EXTRA_ARGS"
        fi

        log "Command: $CMD"
        echo ""

        if $CMD 2>&1 | tee -a "$MODE_LOG" "$MASTER_LOG"; then
            log "$MODE completed successfully. ($(phase_time $MODE_START))"
            COMPLETED_MODES="$COMPLETED_MODES $MODE"
        else
            log "ERROR: $MODE failed after $(phase_time $MODE_START)"
            FAILED_MODES="$FAILED_MODES $MODE"
        fi

        echo "" | tee -a "$MASTER_LOG"
    done
else
    separator "PHASE 3: TRAINING (SKIPPED)"
    log "Using existing results from $RESULTS_DIR"

    # Detect completed runs
    for MODE in lora qlora fullft; do
        if find "$RESULTS_DIR" -maxdepth 2 -name "summary.txt" -path "*_${MODE}_*" -exec grep -l "success" {} \; 2>/dev/null | grep -q .; then
            COMPLETED_MODES="$COMPLETED_MODES $MODE"
        fi
    done
    log "Found completed modes:$COMPLETED_MODES"
fi

COMPLETED_COUNT=$(echo "$COMPLETED_MODES" | wc -w)

if [ "$COMPLETED_COUNT" -eq 0 ]; then
    log ""
    log "ERROR: No completed runs found. Cannot generate reports."
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: TRAINING COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

separator "PHASE 4: TRAINING PERFORMANCE COMPARISON"
PHASE_START=$(date +%s)

python3 -m benchmark_cuda compare --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$MASTER_LOG"

log "Training comparison complete. ($(phase_time $PHASE_START))"

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: CROSS-MODE EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

if [ "$COMPLETED_COUNT" -ge 2 ] && [ "$DRY_RUN" = false ]; then
    separator "PHASE 5: CROSS-MODE EVALUATION (Base vs LoRA vs QLoRA vs Full FT)"
    PHASE_START=$(date +%s)

    python3 -m benchmark_cuda cross-compare --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$MASTER_LOG"

    log "Cross-mode evaluation complete. ($(phase_time $PHASE_START))"
else
    separator "PHASE 5: CROSS-MODE EVALUATION (SKIPPED)"
    if [ "$DRY_RUN" = true ]; then
        log "Skipped: dry run mode (no evaluation data)."
    else
        log "Skipped: need at least 2 completed runs."
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: TRAINING PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

if [ "$DRY_RUN" = false ]; then
    separator "PHASE 6: TRAINING PLOTS"
    PHASE_START=$(date +%s)

    python3 -m benchmark_cuda plot --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$MASTER_LOG"

    log "Plots generated. ($(phase_time $PHASE_START))"
else
    separator "PHASE 6: TRAINING PLOTS (SKIPPED)"
    log "Skipped: dry run mode."
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 7: LLM-AS-JUDGE
# ═══════════════════════════════════════════════════════════════════════════════

if [ "$HAS_JUDGE" = true ] && [ "$COMPLETED_COUNT" -ge 2 ] && [ "$DRY_RUN" = false ]; then
    separator "PHASE 7: LLM-AS-JUDGE EVALUATION (Claude API)"
    PHASE_START=$(date +%s)

    log "Sending 80 questions to Claude for evaluation..."
    log "(This may take several minutes depending on API rate limits)"
    echo ""

    if python3 -m benchmark_cuda judge --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$MASTER_LOG"; then
        log "LLM judge evaluation complete. ($(phase_time $PHASE_START))"
    else
        log "WARNING: LLM judge evaluation failed (non-fatal). ($(phase_time $PHASE_START))"
    fi
else
    separator "PHASE 7: LLM-AS-JUDGE (SKIPPED)"
    if [ "$DRY_RUN" = true ]; then
        log "Skipped: dry run mode."
    elif [ "$HAS_JUDGE" = false ]; then
        log "Skipped: ANTHROPIC_API_KEY not set or --skip-judge used."
        log "To enable: export ANTHROPIC_API_KEY=your-key-here"
    else
        log "Skipped: need at least 2 completed runs."
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 8: FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
TOTAL_HOURS=$(( TOTAL_DURATION / 3600 ))
TOTAL_MINS=$(( (TOTAL_DURATION % 3600) / 60 ))
TOTAL_SECS=$(( TOTAL_DURATION % 60 ))

separator "PHASE 8: FINAL REPORT"

log "Total time: ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
log ""

# Status
log "TRAINING STATUS"
for MODE in $MODES; do
    if echo "$COMPLETED_MODES" | grep -qw "$MODE"; then
        log "  ${MODE^^}: SUCCESS"
    elif echo "$FAILED_MODES" | grep -qw "$MODE"; then
        log "  ${MODE^^}: FAILED"
    else
        log "  ${MODE^^}: SKIPPED"
    fi
done
log ""

# Output files
log "OUTPUT FILES"
log ""
log "  Per-run results:"
for MODE in $MODES; do
    LATEST=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "*_${MODE}_*" 2>/dev/null | sort | tail -1)
    if [ -n "$LATEST" ]; then
        log "    $(basename "$LATEST")/"
    fi
done
log ""

log "  Training comparison:"
log "    $RESULTS_DIR/all_runs_summary.csv"
log ""

if [ "$COMPLETED_COUNT" -ge 2 ] && [ "$DRY_RUN" = false ]; then
    log "  Cross-mode reports:"
    log "    $RESULTS_DIR/cross_comparison/cross_comparison.html   <- Open in browser"
    log "    $RESULTS_DIR/cross_comparison/cross_comparison.md     <- Class / README"
    log "    $RESULTS_DIR/cross_comparison/cross_comparison.csv    <- Spreadsheet"
    log "    $RESULTS_DIR/cross_comparison/cross_comparison.json   <- Raw data"
    log ""
    log "  Plots:"
    log "    $RESULTS_DIR/cross_comparison/loss_curves.png"
    log "    $RESULTS_DIR/cross_comparison/gpu_memory.png"
    log "    $RESULTS_DIR/cross_comparison/step_time.png"
    log ""

    if [ "$HAS_JUDGE" = true ]; then
        log "  LLM Judge:"
        log "    $RESULTS_DIR/cross_comparison/llm_judge_cross_compare.json"
        log ""
    fi
fi

log "  Master log:"
log "    $MASTER_LOG"
log ""

# Final status
if [ -n "$FAILED_MODES" ]; then
    log "WARNING: Some modes failed:$FAILED_MODES"
    log "Check individual logs in $LOG_DIR/ for details."
    exit 1
fi

log "BENCHMARK COMPLETE. All phases finished successfully."
