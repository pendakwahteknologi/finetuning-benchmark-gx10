#!/usr/bin/env bash
#
# run_all.sh — Run the complete fine-tuning benchmark from zero to final report.
#
# This script runs all three training modes (LoRA, QLoRA, Full Fine-Tune),
# then produces the training comparison and cross-mode evaluation report.
#
# Usage:
#   ./run_all.sh                          # Full run (500 steps, all modes)
#   ./run_all.sh --dry-run                # Quick test (5 steps, skip eval)
#   ./run_all.sh --steps 100              # Custom step count
#   ./run_all.sh --machine-label rtx5090  # Different machine label
#   ./run_all.sh --modes "lora fullft"    # Only specific modes
#
# Background usage (safe to close terminal):
#   nohup ./run_all.sh > benchmark_all.log 2>&1 &
#   tail -f benchmark_all.log             # Monitor progress
#
set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────

MACHINE_LABEL="gx10"
MAX_STEPS=500
EPOCHS=""
MODES="lora qlora fullft"
DRY_RUN=false
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
        --help|-h)
            echo "Usage: ./run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run               Quick test (5 steps, skip eval)"
            echo "  --steps N               Number of training steps (default: 500)"
            echo "  --epochs N              Train for N epochs (alternative to --steps)"
            echo "  --machine-label LABEL   Machine identifier (default: gx10)"
            echo "  --modes \"m1 m2 ...\"     Modes to run (default: \"lora qlora fullft\")"
            echo "  --results-dir DIR       Results directory (default: ./results)"
            echo "  --help                  Show this help"
            echo ""
            echo "Examples:"
            echo "  ./run_all.sh                                    # Full benchmark (500 steps)"
            echo "  ./run_all.sh --epochs 1                         # Train for 1 full epoch"
            echo "  ./run_all.sh --dry-run                          # Quick sanity check"
            echo "  ./run_all.sh --steps 100 --modes \"lora fullft\"  # Custom run"
            echo "  nohup ./run_all.sh > benchmark_all.log 2>&1 &   # Background"
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# ─── Setup ───────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/run_all_${TIMESTAMP}.log"

# Logging helper
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

# ─── Pre-flight checks ──────────────────────────────────────────────────────

separator "PRE-FLIGHT CHECKS"

log "Machine label: $MACHINE_LABEL"
if [ -n "$EPOCHS" ]; then
    log "Epochs:        $EPOCHS"
else
    log "Max steps:     $MAX_STEPS"
fi
log "Modes:         $MODES"
log "Results dir:   $RESULTS_DIR"
log "Dry run:       $DRY_RUN"
log "Master log:    $MASTER_LOG"

# Check Python
if ! command -v python3 &>/dev/null; then
    log "ERROR: python3 not found"
    exit 1
fi
log "Python:        $(python3 --version 2>&1)"

# Check CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA:          {torch.version.cuda}')" 2>&1 | tee -a "$MASTER_LOG"
if [ $? -ne 0 ]; then
    log "ERROR: CUDA is not available"
    exit 1
fi

# Check HuggingFace auth
python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" &>/dev/null
if [ $? -ne 0 ]; then
    log "WARNING: HuggingFace not authenticated. Run: huggingface-cli login"
    log "         You also need to accept the Llama model license on HuggingFace."
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check dependencies
python3 -c "import transformers, peft, datasets, rich, typer" 2>&1 | tee -a "$MASTER_LOG"
if [ $? -ne 0 ]; then
    log "ERROR: Missing dependencies. Run: pip install -r benchmark_cuda/requirements.txt"
    exit 1
fi

log "All checks passed."

# ─── Phase 1: Data preparation ──────────────────────────────────────────────

separator "PHASE 1: DATA PREPARATION"

log "Preparing dataset and evaluation questions..."
python3 -m benchmark_cuda prepare 2>&1 | tee -a "$MASTER_LOG"
log "Data preparation complete."

# ─── Phase 2: Training benchmarks ───────────────────────────────────────────

FAILED_MODES=""
COMPLETED_MODES=""
TOTAL_START=$(date +%s)

for MODE in $MODES; do
    separator "PHASE 2: TRAINING — ${MODE^^}"

    MODE_LOG="$LOG_DIR/${MODE}_${TIMESTAMP}.log"
    MODE_START=$(date +%s)

    log "Starting $MODE benchmark..."
    log "Mode log: $MODE_LOG"

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

    # Run training
    if $CMD 2>&1 | tee -a "$MODE_LOG" "$MASTER_LOG"; then
        MODE_END=$(date +%s)
        MODE_DURATION=$(( MODE_END - MODE_START ))
        MODE_MINS=$(( MODE_DURATION / 60 ))
        MODE_SECS=$(( MODE_DURATION % 60 ))
        log "$MODE completed successfully in ${MODE_MINS}m ${MODE_SECS}s"
        COMPLETED_MODES="$COMPLETED_MODES $MODE"
    else
        MODE_END=$(date +%s)
        MODE_DURATION=$(( MODE_END - MODE_START ))
        log "ERROR: $MODE failed after ${MODE_DURATION}s"
        FAILED_MODES="$FAILED_MODES $MODE"
    fi

    echo "" | tee -a "$MASTER_LOG"
done

# ─── Phase 3: Training comparison ───────────────────────────────────────────

separator "PHASE 3: TRAINING COMPARISON"

log "Generating training comparison table..."
python3 -m benchmark_cuda compare --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$MASTER_LOG"

# ─── Phase 4: Cross-mode evaluation comparison ──────────────────────────────

COMPLETED_COUNT=$(echo "$COMPLETED_MODES" | wc -w)
if [ "$COMPLETED_COUNT" -ge 2 ] && [ "$DRY_RUN" = false ]; then
    separator "PHASE 4: CROSS-MODE COMPARISON"

    log "Generating cross-mode evaluation report..."
    python3 -m benchmark_cuda cross-compare --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$MASTER_LOG"
    log "Cross-comparison reports saved to: $RESULTS_DIR/cross_comparison/"
else
    separator "PHASE 4: CROSS-MODE COMPARISON (SKIPPED)"
    if [ "$DRY_RUN" = true ]; then
        log "Skipped: dry run mode (no evaluation data)."
    else
        log "Skipped: need at least 2 completed runs with evaluation data."
    fi
fi

# ─── Final summary ──────────────────────────────────────────────────────────

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
TOTAL_HOURS=$(( TOTAL_DURATION / 3600 ))
TOTAL_MINS=$(( (TOTAL_DURATION % 3600) / 60 ))
TOTAL_SECS=$(( TOTAL_DURATION % 60 ))

separator "COMPLETE"

log "Total wall-clock time: ${TOTAL_HOURS}h ${TOTAL_MINS}m ${TOTAL_SECS}s"
log ""
log "Completed modes: ${COMPLETED_MODES:-none}"
if [ -n "$FAILED_MODES" ]; then
    log "Failed modes:    $FAILED_MODES"
fi
log ""
log "Results:           $RESULTS_DIR/"
log "Training compare:  $RESULTS_DIR/all_runs_summary.csv"
if [ "$COMPLETED_COUNT" -ge 2 ] && [ "$DRY_RUN" = false ]; then
    log "Cross-comparison:  $RESULTS_DIR/cross_comparison/"
    log "  - HTML report:   $RESULTS_DIR/cross_comparison/cross_comparison.html"
    log "  - Markdown:      $RESULTS_DIR/cross_comparison/cross_comparison.md"
    log "  - CSV data:      $RESULTS_DIR/cross_comparison/cross_comparison.csv"
    log "  - JSON data:     $RESULTS_DIR/cross_comparison/cross_comparison.json"
fi
log "Master log:        $MASTER_LOG"

# Exit with error if any mode failed
if [ -n "$FAILED_MODES" ]; then
    log ""
    log "WARNING: Some modes failed. Check logs for details."
    exit 1
fi

log ""
log "All benchmarks completed successfully!"
