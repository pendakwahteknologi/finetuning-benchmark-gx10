# Fine-Tuning Benchmark for ASUS Ascent GX10

A CLI-first benchmarking suite for measuring and comparing LLM fine-tuning performance across GPU hardware. Built for the **ASUS Ascent GX10** powered by the **NVIDIA GB10** (Blackwell architecture) with 128 GB unified memory.

This tool benchmarks three fine-tuning modes — **LoRA**, **QLoRA**, and **Full Fine-Tune** — using the same model, dataset, and hyperparameters, producing fair, reproducible timing comparisons with rich terminal output designed for video recording.

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

| Mode | Description |
|------|-------------|
| **LoRA** | Lightweight adapter training — base model weights frozen, only small adapter matrices are trained |
| **QLoRA** | 4-bit quantized base model with LoRA adapters — reduces memory footprint while training adapters |
| **Full Fine-Tune** | All model parameters are trainable — true full parameter optimization |

All three modes use the same base model, dataset, prompt format, sequence length, seed, and step count to ensure a fair comparison.

### What Gets Measured

- Total training wall-clock time
- Average / median / P95 step time
- Peak GPU memory usage
- Tokens per second and samples per second
- Final training loss
- Before vs after evaluation (ROUGE-L, BLEU, side-by-side generation comparison)

---

## Model and Dataset

| | Details |
|---|---|
| **Base Model** | [`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B) (base, not Instruct) |
| **Dataset** | [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |
| **Split** | 90% train / 5% validation / 5% test (seed 42) |
| **Evaluation** | 80 preset questions (10 per category) from held-out test set |

We use the **base** model (not Instruct) so the before/after difference is dramatic — the base model cannot follow instructions, while the fine-tuned model produces clean, structured answers.

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pendakwahteknologi/finetuning-benchmark-gx10.git
cd finetuning-benchmark-gx10
```

### 2. Install Dependencies

```bash
pip install -r benchmark_cuda/requirements.txt --break-system-packages
```

The core dependencies are:
- `torch` (with CUDA support)
- `transformers`, `peft`, `datasets`, `accelerate`
- `bitsandbytes` (for QLoRA)
- `rich`, `typer` (for terminal UI)
- `rouge-score`, `nltk` (for evaluation metrics)

### 3. Set Up Hugging Face Access

The Llama 3.1 model is gated — you need a Hugging Face account with access granted.

**Step 1:** Create a Hugging Face account at [huggingface.co](https://huggingface.co)

**Step 2:** Go to the model page and request access:
- [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- Click **"Request access"** and accept the license agreement
- Access is usually granted within minutes

**Step 3:** Create an access token:
- Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Click **"Create new token"**
- Name it (e.g., `gx10-benchmark`)
- Set type to **Read**
- Copy the token (starts with `hf_`)

**Step 4:** Save the token on your machine:

```bash
mkdir -p ~/.cache/huggingface
echo "hf_YOUR_TOKEN_HERE" > ~/.cache/huggingface/token
```

**Step 5:** Verify access:

```bash
python3 -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami()['name'])"
```

If this prints your username, you're ready.

### 4. Pre-Download Model and Data

To avoid downloads during benchmarking, cache everything first:

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')

print('Downloading model...')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B', dtype=torch.bfloat16)
del model

print('Downloading dataset...')
load_dataset('databricks/databricks-dolly-15k', split='train')

print('All cached and ready.')
"
```

### 5. Prepare Dataset Splits

```bash
python3 -m benchmark_cuda prepare
```

This creates:
- `data/processed/train.jsonl` (13,509 samples)
- `data/processed/val.jsonl` (750 samples)
- `data/processed/test.jsonl` (752 samples)
- `data/eval/preset_questions.jsonl` (80 evaluation questions)

---

## Usage

### Quick Test (Dry Run)

Run 5 steps to verify everything works:

```bash
python3 -m benchmark_cuda run --mode lora --dry-run --machine-label gx10
```

### Run Benchmarks

```bash
# LoRA fine-tuning
python3 -m benchmark_cuda run --mode lora --machine-label gx10 --max-steps 500

# QLoRA fine-tuning
python3 -m benchmark_cuda run --mode qlora --machine-label gx10 --max-steps 500

# Full fine-tuning
python3 -m benchmark_cuda run --mode fullft --machine-label gx10 --max-steps 500
```

### Run All Three Sequentially (Background)

```bash
nohup bash -c '
python3 -m benchmark_cuda run --mode lora --machine-label gx10 --max-steps 500 2>&1 | tee lora_run.log
python3 -m benchmark_cuda run --mode qlora --machine-label gx10 --max-steps 500 2>&1 | tee qlora_run.log
python3 -m benchmark_cuda run --mode fullft --machine-label gx10 --max-steps 500 2>&1 | tee fullft_run.log
python3 -m benchmark_cuda compare --results-dir ./results 2>&1 | tee compare_run.log
' > benchmark_all.log 2>&1 &
```

Monitor progress:

```bash
tail -5 benchmark_all.log
```

### Compare Results

```bash
python3 -m benchmark_cuda compare --results-dir ./results
```

### Inspect a Single Run

```bash
python3 -m benchmark_cuda inspect --run ./results/gx10_lora_20260403_140000
```

---

## CLI Options

```
python3 -m benchmark_cuda run [OPTIONS]

Options:
  --machine-label, -m    Machine identifier             [default: gx10]
  --mode                 Training mode: lora, qlora, fullft  [default: lora]
  --model                HuggingFace model name         [default: meta-llama/Llama-3.1-8B]
  --max-steps            Number of optimizer steps       [default: 500]
  --warmup-steps         Steps excluded from timing      [default: 3]
  --seq-len              Maximum sequence length         [default: 1024]
  --micro-batch-size     Batch size per step
  --grad-accum           Gradient accumulation steps
  --learning-rate, --lr  Learning rate
  --dtype                Precision: bfloat16, float16    [default: bfloat16]
  --gradient-checkpointing  Enable gradient checkpointing
  --lora-r               LoRA rank                       [default: 16]
  --lora-alpha           LoRA alpha                      [default: 32]
  --seed                 Random seed                     [default: 42]
  --logging-steps        Log every N steps               [default: 10]
  --eval-steps           Validate every N steps          [default: 100]
  --skip-eval            Skip post-training evaluation
  --dry-run              Run 5 steps only, skip evaluation
  --output-dir           Output directory                [default: ./results]
```

---

## Output Structure

Each benchmark run produces a complete result folder:

```
results/
  gx10_lora_20260403_140000/
    config.json                 # Full run configuration
    benchmark_metrics.json      # Aggregated metrics
    benchmark_metrics.csv       # Per-step metrics
    train.log                   # Full training log
    summary.txt                 # Human-readable summary
    system_info.json            # System details
    gpu_info.json               # GPU details
    evaluation/
      preset_questions.jsonl    # The 80 evaluation questions
      baseline_predictions.jsonl    # Base model answers
      finetuned_predictions.jsonl   # Fine-tuned model answers
      side_by_side_comparison.jsonl # Side-by-side comparison
      evaluation_metrics.json   # ROUGE-L, BLEU, per-category scores
      evaluation_metrics.csv    # Metrics in CSV format
      evaluation_summary.md     # Markdown summary report
      evaluation_table.html     # Visual HTML comparison table
```

---

## Benchmark Design

### Fairness

All runs use identical:
- Base model and tokenizer
- Dataset splits and preprocessing
- Prompt template (Alpaca-style)
- Sequence length (1024)
- Random seed (42)
- Number of steps (500)
- Logging and evaluation frequency
- Evaluation inference settings (temperature=0, deterministic decoding)

If any parameter must differ due to hardware constraints (e.g., smaller batch size on a GPU with less memory), the difference is explicitly logged in `config.json` under `fairness_notes`.

### Timing

The benchmark measures **training time only**:
- Includes: forward pass, backward pass, optimizer steps, validation
- Excludes: model download, dataset download, tokenization, environment setup

The first 3 steps are treated as warmup (CUDA kernel compilation, JIT) and excluded from timing statistics.

### Failure Handling

If a run hits an out-of-memory error or other failure:
- A clear failure banner is printed
- The failure reason is saved in metrics
- Partial results are still written to disk
- Status is marked as `oom`, `failed`, or `interrupted`

---

## GX10-Optimized Defaults

The benchmark defaults are tuned for the GX10's 128 GB unified memory:

| Parameter | LoRA | QLoRA | Full FT |
|-----------|------|-------|---------|
| Micro batch size | 8 | 8 | 4 |
| Gradient accumulation | 4 | 4 | 8 |
| Effective batch size | 32 | 32 | 32 |
| Learning rate | 2e-4 | 2e-4 | 2e-5 |
| Precision | bf16 | bf16 | bf16 |
| Attention | SDPA | SDPA | SDPA |

The GX10 can run Full Fine-Tune of an 8B parameter model natively without offloading — a task that would require gradient checkpointing or CPU offload on GPUs with less memory.

---

## Cross-Machine Comparison

This tool is designed to compare the GX10 against other machines (e.g., RTX 5090). Run the same benchmarks on a different machine:

```bash
python3 -m benchmark_cuda run --mode lora --machine-label rtx5090 --max-steps 500
python3 -m benchmark_cuda run --mode qlora --machine-label rtx5090 --max-steps 500
python3 -m benchmark_cuda run --mode fullft --machine-label rtx5090 --max-steps 500
```

Then copy all result folders into one `results/` directory and compare:

```bash
python3 -m benchmark_cuda compare --results-dir ./results
```

---

## Project Structure

```
benchmark_cuda/
  benchmark.py              # CLI entrypoint (run / compare / inspect / prepare)
  trainer.py                # Shared training loop with timing and progress display
  modes/
    lora.py                 # LoRA mode: freeze base, attach PEFT adapters
    qlora.py                # QLoRA mode: 4-bit quantized base + LoRA adapters
    fullft.py               # Full FT mode: all parameters trainable
  data/
    prepare.py              # Load dataset, normalize, split, save JSONL
    prompt_format.py        # Alpaca-style prompt template
    preset_questions.py     # Sample evaluation questions from test split
  evaluation/
    evaluate.py             # Baseline vs fine-tuned inference pipeline
    eval_metrics.py         # ROUGE-L, BLEU, exact match, per-category
    compare_models.py       # Side-by-side comparison generation
  utils/
    config.py               # Benchmark configuration dataclass
    logging_utils.py        # Rich terminal output + file logging
    metrics.py              # Step timing, memory tracking, aggregation
    system_info.py          # System and GPU information capture
    gpu_monitor.py          # Background GPU memory monitoring
  commands/
    compare.py              # Cross-run comparison tables
    inspect.py              # Single-run detail viewer
  requirements.txt
```

---

## Acknowledgments

- Workflow inspired by [pendakwahteknologi/finetune-rocm](https://github.com/pendakwahteknologi/finetune-rocm)
- Base model: [Meta Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-8B)
- Dataset: [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

---

## License

This project is open source. The Llama 3.1 model is subject to the [Meta Llama 3.1 Community License Agreement](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/LICENSE).
