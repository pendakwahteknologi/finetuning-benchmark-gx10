# Fine-Tuning Benchmark for ASUS Ascent GX10

A complete benchmarking suite for comparing LLM fine-tuning methods on real hardware. Trains **LoRA**, **QLoRA**, and **Full Fine-Tune** using the same model, dataset, and hyperparameters, then evaluates all three against the base model on 80 curated questions to determine which method produces the best results.

Built and tested on the **ASUS Ascent GX10** — an ARM-based desktop powered by the **NVIDIA GB10** (Blackwell architecture) with **128 GB unified memory**.

---

## Results at a Glance

All three fine-tuning methods trained **Llama 3.1 8B Instruct** for 500 steps on the Databricks Dolly 15k dataset, then answered 80 curated questions across 8 categories.

### Training Performance

| Metric | LoRA | QLoRA | Full Fine-Tune |
|--------|-----:|------:|---------------:|
| **Total Time** | 4h 48m | 9h 14m | 5h 06m |
| **Avg Step Time** | 33.93s | 66.21s | 36.45s |
| **Peak GPU Memory** | 87.4 GB | 12.4 GB | 93.6 GB |
| **Final Loss** | 1.5124 | 1.6082 | 1.2897 |
| **Tokens/sec** | 161.9 | 83.0 | 150.7 |
| **Trainable Params** | 13.6M (0.17%) | 13.6M (0.17%) | 8.03B (100%) |

### Evaluation: Base Model vs Fine-Tuned (80 Questions)

| Metric | Base Model | LoRA | QLoRA | Full Fine-Tune |
|--------|--------:|-----:|------:|---------------:|
| **ROUGE-L** | 0.1554 | 0.1529 (-0.0025) | 0.1538 (-0.0016) | 0.1471 (-0.0083) |
| **BLEU** | 0.0393 | 0.0404 (+0.0011) | 0.0429 (+0.0036) | 0.0405 (+0.0012) |
| **Best Answer Wins** | -- | 32/80 (40%) | 24/80 (30%) | 24/80 (30%) |

### Category Winners (ROUGE-L)

| Category | Winner |
|----------|--------|
| Brainstorming | **LoRA** |
| Classification | **QLoRA** |
| Closed QA | **QLoRA** |
| Creative Writing | **LoRA** |
| General QA | **QLoRA** |
| Information Extraction | **LoRA** |
| Open QA | **LoRA** |
| Summarization | **LoRA** |

### Verdict

| Award | Winner |
|-------|--------|
| Best Quality (ROUGE-L) | **QLoRA** |
| Best Quality (BLEU) | **QLoRA** |
| Most Per-Question Wins | **LoRA** (32/80) |
| Fastest Training | **LoRA** (4h 48m) |
| Lowest Memory Usage | **QLoRA** (12.4 GB) |
| Lowest Final Loss | **Full Fine-Tune** (1.2897) |

---

## Key Findings

### LoRA: Best All-Rounder

LoRA won the most individual questions (32/80) and 5 out of 8 categories. It was also the fastest to train (4h 48m) with the highest throughput (161.9 tok/s). For most practical use cases on this hardware, LoRA offers the best balance of quality, speed, and simplicity.

### QLoRA: Best for Memory-Constrained Environments

QLoRA used only **12.4 GB** of GPU memory — 7x less than LoRA and Full FT. It achieved the highest overall ROUGE-L and BLEU scores despite being the slowest to train (9h 14m). The 2x slowdown comes from 4-bit dequantization overhead on ARM aarch64 with unified memory. On standard discrete GPUs, this gap would be smaller.

### Full Fine-Tune: Lowest Loss, Not Necessarily Best Output

Full FT achieved the lowest training loss (1.2897) by optimizing all 8 billion parameters, but this didn't translate to the best evaluation scores. It won zero categories and showed the largest ROUGE-L regression from baseline (-0.0083). This suggests 500 steps may not be enough for full parameter training to converge on output quality, or that the model may be overfitting to training patterns rather than generalizing well on the evaluation questions.

### Why Do Fine-Tuned Scores Look Close to (or Below) Baseline?

This is expected and worth explaining:

1. **The base model is already instruction-tuned.** We used `Llama-3.1-8B-Instruct`, which Meta already fine-tuned on high-quality instruction data. Further fine-tuning on Dolly 15k (a smaller, community-generated dataset) can cause slight regression on general instruction-following.

2. **500 steps is a short run.** This benchmark prioritizes fair, reproducible comparison across methods rather than training to convergence. A production fine-tune would typically run for 1-3 epochs (thousands of steps).

3. **ROUGE-L and BLEU measure surface-level similarity.** The fine-tuned models often produce more concise, focused answers that score lower on token overlap but may actually be more useful. The side-by-side comparisons in the HTML report show this clearly.

4. **The real value is in the comparison.** Even with modest absolute improvements, the relative differences between LoRA, QLoRA, and Full FT are meaningful and consistent.

---

## Challenges and Lessons Learned

### QLoRA Crashes on ARM aarch64 at Higher Batch Sizes

Running QLoRA with `micro_batch_size=8` caused the GX10 to **hard-reboot** after completing only 1 training step. No OOM-killer trace or GPU Xid error appeared in kernel logs, suggesting a hardware-level fault — likely a power spike or thermal event on the unified memory bus during 4-bit dequantization.

**Root cause:** The bitsandbytes library's 4-bit CUDA kernels behave differently on ARM aarch64 with NVIDIA's unified (C2C) memory architecture. The dequantize-compute-quantize cycle at batch size 8 appears to create memory access patterns that destabilize the system, even though total memory usage was only 27.8 GB (well within the 128 GB available).

**Workaround (applied automatically):** QLoRA mode now defaults to `micro_batch_size=2` with `gradient_accumulation_steps=16` (keeping effective batch size at 32) and enables gradient checkpointing. These settings trade training speed for stability.

**Takeaway for others:** If you're running QLoRA on ARM-based systems with unified memory (Grace Hopper, GB10, Jetson), start with small batch sizes and scale up carefully. The memory headroom numbers can be misleading.

### Full Fine-Tune Device Mismatch Error

Using `device_map="auto"` with the accelerate library caused a "tensors on different devices" error — accelerate placed some layers (embedding tables) on CPU while the training loop expected everything on CUDA.

**Fix:** Changed to `device_map="cuda"` in `benchmark_cuda/modes/fullft.py`. This is safe on the GX10 because the entire 8B model fits in GPU memory. On machines with less VRAM, you'd need to use `device_map="auto"` with proper handling for mixed-device tensors.

### Unified Memory: Not the Same as More VRAM

The GX10's 128 GB unified memory (shared via NVIDIA C2C) behaves differently from discrete GPU VRAM:

- **Bandwidth:** Unified memory has lower bandwidth than dedicated HBM. This is why LoRA at 87 GB and Full FT at 93 GB still run at similar speeds — they're not bandwidth-limited in the same way a discrete GPU would be.
- **QLoRA paradox:** Despite using only 12.4 GB (easily fits in any GPU's VRAM), QLoRA was the slowest mode. The 4-bit dequantization creates scattered memory access patterns that are particularly expensive on unified memory.
- **Advantage:** The GX10 can run Full Fine-Tune of an 8B model without gradient checkpointing, CPU offloading, or model sharding. On a typical 24 GB GPU, this would be impossible without these techniques.

### What Would We Do Differently?

1. **Use the base model (not Instruct)** for a more dramatic before/after comparison. Fine-tuning an already instruction-tuned model makes the improvements subtle.
2. **Train longer** (1000-2000 steps) to let Full Fine-Tune converge — 500 steps may underrepresent its potential.
3. **Add human evaluation** alongside automated metrics — ROUGE-L and BLEU don't capture answer usefulness well.
4. **Test multiple LoRA ranks** (4, 8, 16, 32, 64) to find the sweet spot for this model and dataset.

---

## Hardware Profile

| Component | Specification |
|-----------|--------------|
| **System** | ASUS Ascent GX10 |
| **GPU** | NVIDIA GB10 (Blackwell), 48 SMs, Compute 12.1 |
| **Memory** | 128 GB Unified (CPU + GPU via NVIDIA C2C) |
| **CPU** | ARM aarch64 — 10x Cortex-X925 + 10x Cortex-A725 |
| **OS** | Ubuntu 24.04.4 LTS |
| **CUDA** | 13.0, Driver 580.142 |
| **PyTorch** | 2.11.0+cu130 |

---

## What This Benchmarks

| Mode | Description | What Gets Trained |
|------|-------------|-------------------|
| **LoRA** | Lightweight adapter training — base model weights frozen, only small adapter matrices are trained | 13.6M params (0.17%) |
| **QLoRA** | 4-bit quantized base model with LoRA adapters — reduces memory footprint while training adapters | 13.6M params (0.17%) |
| **Full Fine-Tune** | All model parameters are trainable — true full parameter optimization | 8.03B params (100%) |

All three modes use the same base model, dataset, prompt format, sequence length, seed, and step count to ensure a fair comparison.

### Metrics Collected

**Training:**
- Total wall-clock time, average / median / P95 step time
- Peak GPU memory usage
- Tokens per second and samples per second
- Final training loss, validation loss at checkpoints

**Evaluation:**
- ROUGE-L and BLEU scores vs reference answers
- Per-category breakdown (8 categories, 10 questions each)
- Side-by-side generation comparison (base model vs fine-tuned)
- Cross-mode comparison (which fine-tuning method wins per question)

---

## Model and Dataset

| | Details |
|---|---|
| **Base Model** | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| **Dataset** | [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |
| **Split** | 90% train / 5% validation / 5% test (seed 42) |
| **Evaluation** | 80 preset questions (10 per category) from held-out test set |

**Categories evaluated:** brainstorming, classification, closed QA, creative writing, general QA, information extraction, open QA, summarization.

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/pendakwahteknologi/Fine-Tuning-Benchmark-GX10.git
cd Fine-Tuning-Benchmark-GX10
pip install -r benchmark_cuda/requirements.txt --break-system-packages
```

### 2. Set Up Hugging Face Access

The Llama 3.1 model is gated — you need a Hugging Face account with access granted.

1. Create a [Hugging Face account](https://huggingface.co)
2. Request access at [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (usually granted within minutes)
3. Create a [read token](https://huggingface.co/settings/tokens) and save it:

```bash
mkdir -p ~/.cache/huggingface
echo "hf_YOUR_TOKEN_HERE" > ~/.cache/huggingface/token
```

4. Verify: `python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"`

### 3. Run Everything

**Full benchmark (all three modes + comparison reports):**

```bash
./run_all.sh
```

**Background (safe to close terminal):**

```bash
nohup ./run_all.sh > benchmark_all.log 2>&1 &
tail -f benchmark_all.log
```

**Quick sanity check (5 steps, no evaluation):**

```bash
./run_all.sh --dry-run
```

**On a different machine:**

```bash
./run_all.sh --machine-label rtx5090
```

This runs: data preparation, LoRA training, QLoRA training, Full Fine-Tune training, training comparison, and cross-mode evaluation report — fully automated.

Expected time on GX10: ~19 hours total (LoRA ~5h + QLoRA ~9h + Full FT ~5h).

### 4. Generate Comparison Reports (After Training)

If training is already complete, generate all comparison reports without retraining:

```bash
./run_comparison.sh
```

This produces:
- Rich terminal tables with color-coded metrics and verdict
- `results/cross_comparison/cross_comparison.html` — Interactive dark-themed HTML report
- `results/cross_comparison/cross_comparison.md` — Markdown report for documentation
- `results/cross_comparison/cross_comparison.csv` — Per-question metrics for spreadsheets
- `results/cross_comparison/cross_comparison.json` — Full machine-readable data

---

## Usage

### Individual Commands

```bash
# Run a single mode
python3 -m benchmark_cuda run --mode lora --machine-label gx10 --max-steps 500
python3 -m benchmark_cuda run --mode qlora --machine-label gx10 --max-steps 500
python3 -m benchmark_cuda run --mode fullft --machine-label gx10 --max-steps 500

# Training metrics comparison table
python3 -m benchmark_cuda compare --results-dir ./results

# Cross-mode evaluation (Base vs LoRA vs QLoRA vs Full FT)
python3 -m benchmark_cuda cross-compare --results-dir ./results

# Inspect a single run
python3 -m benchmark_cuda inspect --run ./results/gx10_lora_20260403_140352

# Prepare dataset only (no training)
python3 -m benchmark_cuda prepare
```

### CLI Options

```
python3 -m benchmark_cuda run [OPTIONS]

Options:
  --machine-label, -m       Machine identifier              [default: gx10]
  --mode                    Training mode: lora, qlora, fullft
  --model                   HuggingFace model name          [default: meta-llama/Llama-3.1-8B]
  --max-steps               Number of optimizer steps        [default: 500]
  --warmup-steps            Steps excluded from timing       [default: 3]
  --seq-len                 Maximum sequence length          [default: 1024]
  --micro-batch-size        Batch size per step
  --grad-accum              Gradient accumulation steps
  --learning-rate, --lr     Learning rate
  --dtype                   Precision: bfloat16, float16     [default: bfloat16]
  --gradient-checkpointing  Enable gradient checkpointing
  --lora-r                  LoRA rank                        [default: 16]
  --lora-alpha              LoRA alpha                       [default: 32]
  --seed                    Random seed                      [default: 42]
  --logging-steps           Log every N steps                [default: 10]
  --eval-steps              Validate every N steps           [default: 100]
  --skip-eval               Skip post-training evaluation
  --dry-run                 Run 5 steps only, skip evaluation
  --output-dir              Output directory                 [default: ./results]
```

### Monitoring a Background Run

```bash
tail -f benchmark_all.log                          # Follow live output
grep -E "Mode:|Step " benchmark_all.log | tail -5  # Check current progress
ps aux | grep benchmark_cuda | grep -v grep        # Check if running
nvidia-smi                                         # Check GPU activity
```

---

## GX10-Optimized Defaults

The benchmark defaults are tuned for the GX10's 128 GB unified memory:

| Parameter | LoRA | QLoRA | Full FT |
|-----------|------|-------|---------|
| Micro batch size | 8 | 2 | 4 |
| Gradient accumulation | 4 | 16 | 8 |
| Effective batch size | 32 | 32 | 32 |
| Learning rate | 2e-4 | 2e-4 | 2e-5 |
| Precision | bf16 | bf16 | bf16 |
| Attention | SDPA | SDPA | SDPA |
| Gradient checkpointing | No | Yes | No |

QLoRA uses smaller batch sizes and gradient checkpointing due to stability issues on ARM aarch64 with unified memory (see [Challenges](#challenges-and-lessons-learned)).

---

## Output Structure

```
results/
  gx10_lora_20260403_140352/
    config.json                       # Full run configuration
    benchmark_metrics.json            # Aggregated training metrics
    benchmark_metrics.csv             # Per-step training metrics
    train.log                         # Full training log
    summary.txt                       # Human-readable summary
    system_info.json                  # System hardware details
    gpu_info.json                     # GPU details
    evaluation/
      preset_questions.jsonl          # The 80 evaluation questions
      baseline_predictions.jsonl      # Base model answers
      finetuned_predictions.jsonl     # Fine-tuned model answers
      side_by_side_comparison.jsonl   # Per-question comparison
      evaluation_metrics.json         # ROUGE-L, BLEU, per-category
      evaluation_metrics.csv          # Metrics in CSV format
      evaluation_summary.md           # Markdown evaluation report
      evaluation_table.html           # Visual HTML comparison

  cross_comparison/                   # Generated by cross-compare
    cross_comparison.html             # Interactive HTML report
    cross_comparison.md               # Markdown report
    cross_comparison.csv              # Per-question CSV
    cross_comparison.json             # Full machine-readable data
```

---

## Benchmark Design

### Fairness

All runs use identical:
- Base model and tokenizer
- Dataset splits and preprocessing
- Prompt template (Alpaca-style)
- Sequence length (1024), random seed (42), step count (500)
- Logging and evaluation frequency
- Evaluation inference settings (temperature=0, deterministic)

If any parameter differs due to hardware constraints (e.g., smaller batch size), it is explicitly logged in `config.json` under `fairness_notes`.

### Timing

Measures **training time only** — excludes model download, dataset download, tokenization, and environment setup. The first 3 steps are treated as warmup (CUDA kernel compilation) and excluded from timing statistics.

### Evaluation Pipeline

After training completes, the evaluation pipeline:
1. Loads the **base model** (unmodified) and runs inference on 80 curated questions
2. Runs the same 80 questions through the **fine-tuned model**
3. Computes ROUGE-L, BLEU, exact match, and normalized match scores
4. Generates side-by-side comparisons with per-category breakdowns

The `cross-compare` command then aggregates all three modes into a unified comparison, ranking which method produces the best output for each question and category.

---

## Cross-Machine Comparison

This tool is designed to compare hardware. Run the same benchmark on different machines:

```bash
# On machine 2
./run_all.sh --machine-label rtx5090

# Copy results to one directory and compare
python3 -m benchmark_cuda compare --results-dir ./results
```

---

## Cross-Model Comparison (Instruct vs Base)

A standalone script compares LoRA fine-tuning results between the Instruct and Base variants of Llama 3.1 8B:

```bash
python3 scripts/cross_model_compare.py
```

This reads evaluation data from two LoRA runs (one using `Llama-3.1-8B-Instruct`, the other using `Llama-3.1-8B`) and produces:

- `results/cross_model_comparison/cross_model_comparison.html` -- Interactive dark-themed HTML report with 80 questions side-by-side
- `results/cross_model_comparison/cross_model_comparison.md` -- Markdown summary
- `results/cross_model_comparison/cross_model_comparison.json` -- Full machine-readable data
- `results/cross_model_comparison/cross_model_comparison.csv` -- Per-question metrics

---

## Project Structure

```
benchmark_cuda/
  benchmark.py                # CLI entrypoint (run / compare / cross-compare / inspect / prepare)
  trainer.py                  # Shared training loop with timing and progress display
  modes/
    lora.py                   # LoRA: freeze base, attach PEFT adapters
    qlora.py                  # QLoRA: 4-bit quantized base + LoRA adapters
    fullft.py                 # Full FT: all parameters trainable
  data/
    prepare.py                # Load dataset, normalize, split, save JSONL
    prompt_format.py          # Alpaca-style prompt template
    preset_questions.py       # Sample evaluation questions from test split
  evaluation/
    evaluate.py               # Baseline vs fine-tuned inference pipeline
    eval_metrics.py           # ROUGE-L, BLEU, exact match, per-category
    compare_models.py         # Single-run side-by-side comparison
    cross_compare.py          # Cross-mode comparison (all modes at once)
  utils/
    config.py                 # Benchmark configuration dataclass
    logging_utils.py          # Rich terminal output + file logging
    metrics.py                # Step timing, memory tracking, aggregation
    system_info.py            # System and GPU information capture
    gpu_monitor.py            # Background GPU memory monitoring
  commands/
    compare.py                # Cross-run training comparison tables
    inspect.py                # Single-run detail viewer
run_all.sh                    # Full benchmark: train all modes + generate reports
run_benchmark.sh              # Single script for complete benchmark pipeline
run_comparison.sh             # Generate comparison reports from existing results
scripts/
  cross_model_compare.py      # Cross-model comparison (Instruct vs Base LoRA)
```

---

## Acknowledgments

- Base model: [Meta Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- Dataset: [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- Fine-tuning libraries: [Hugging Face Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

---

## License

This project is open source. The Llama 3.1 model is subject to the [Meta Llama 3.1 Community License Agreement](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/LICENSE).
