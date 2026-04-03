# CUDA Fine-Tuning Benchmark Script Specification

## Purpose

Create a **CLI-first benchmarking script suite** for comparing fine-tuning performance between:

1. **ASUS Ascent GX10**
2. **RTX 5090 powered machine**

The benchmark must measure and compare the **time taken to perform fine-tuning** using the **same base model** and **same dataset** across three training modes:

- **LoRA**
- **QLoRA**
- **Full fine-tune** (true full parameter training, not just a larger LoRA run)

This work must be based on the workflow and structure of this repository:

`https://github.com/pendakwahteknologi/finetune-rocm`

The new implementation must be adapted for **NVIDIA CUDA**, must be runnable from the **command line**, and must produce **very clear and visually informative CLI output** because the process will be **recorded for a YouTube video**.

Primary workflow reference repo:
`https://github.com/pendakwahteknologi/finetune-rocm`

---



## Relevant links

Use these official or primary references when building the solution:

### Repository and benchmark baseline
- ROCm reference repository: `https://github.com/pendakwahteknologi/finetune-rocm`

### Model and dataset
- Base model: `https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct`
- Dataset: `https://huggingface.co/datasets/databricks/databricks-dolly-15k`

### Core libraries and docs
- Transformers: `https://huggingface.co/docs/transformers/index`
- PEFT: `https://huggingface.co/docs/peft/index`
- Accelerate: `https://huggingface.co/docs/accelerate/index`
- Datasets: `https://huggingface.co/docs/datasets/index`
- bitsandbytes: `https://huggingface.co/docs/bitsandbytes/index`
- Rich: `https://rich.readthedocs.io/en/stable/`

### Notes on usage
- Use the ROCm repository only as the workflow and structure reference.
- The CUDA implementation should target NVIDIA GPUs and must not depend on ROCm.
- Use the model page for access requirements and tokenizer/model loading details.
- Use the dataset page for dataset schema and category information.
- Use the Hugging Face library documentation as the primary implementation references.


## Important clarification

The existing ROCm repository is the **starting point and workflow reference only**.

The CUDA version must preserve the spirit of the original flow, but it must explicitly support these three distinct and real tuning modes:

1. `lora`
2. `qlora`
3. `fullft`

Do **not** treat the repo's existing larger LoRA path as a true full fine-tune.  
The CUDA benchmark must implement a **real full fine-tuning mode** where all model weights are trainable.

---

## What you need to build

Create a complete CUDA-based benchmarkable training workflow that can be run from the CLI.

### Deliverables

Build the following:

1. A **main benchmark CLI entrypoint**
2. Supporting training scripts/modules
3. A configuration structure for model, dataset and benchmark settings
4. Clear benchmark logging and result export
5. Helpful terminal output suitable for recording on video
6. Documentation on how to run each benchmark mode

---

## High level goals

The script suite must:

- Use the **same base model** for all benchmark modes
- Use the **same dataset** for all benchmark modes
- Use the **same prompt formatting**
- Use the **same sequence length**
- Use the **same seed**
- Use the **same number of optimizer steps**
- Use the **same evaluation cadence**
- Measure only **training benchmark time**, not model download time
- Produce repeatable and fair comparison outputs between both machines

---

## Benchmark modes required

### 1. LoRA
- Standard PEFT LoRA fine-tuning on CUDA
- Base model weights frozen
- Adapter weights trainable

### 2. QLoRA
- 4-bit quantized base model
- PEFT adapters trainable
- CUDA implementation, preferably using `bitsandbytes`

### 3. Full fine-tune
- True full parameter fine-tuning
- All model parameters trainable
- Must be clearly separate from LoRA or QLoRA logic

---

## Benchmark philosophy

This benchmark is primarily about:

- **time taken**
- **stability**
- **memory usage**
- **comparability across systems**

This is **not** just a training script.  
It is a **benchmark tool** designed to produce a fair and visually understandable comparison.

---

## Output requirements

The CLI output must be **extremely clear, friendly and informative**.

The output should look polished enough to be shown directly in a YouTube video recording.

### The terminal output must include:

- Clear phase banners
- Step-by-step progress
- Current mode (`LoRA`, `QLoRA`, `Full FT`)
- Machine name / label
- Model name
- Dataset name
- Sequence length
- Batch size
- Gradient accumulation
- Precision mode
- Total planned steps
- Current step
- Elapsed time
- Average step time
- Estimated remaining time
- Current loss
- Learning rate
- GPU memory usage
- Tokens/sec if available
- Samples/sec if available
- Final summary block
- Path to saved logs/results

### CLI style expectations

The CLI should be:

- readable
- visually structured
- easy to follow on camera
- not noisy in a confusing way
- informative enough for viewers to understand what is happening

Use:

- section headers
- separators
- aligned labels
- concise status messages
- periodic summaries
- end-of-run recap

It is fine to use a rich terminal library such as:

- `rich`
- `typer`
- `click`
- `tqdm`

`rich` is strongly preferred for visually clean output.

---

## Very important UX requirement for YouTube recording

The output should make it obvious to a viewer:

1. what machine is being benchmarked
2. what tuning mode is running
3. which model is being used
4. how far the run has progressed
5. how long it has taken so far
6. whether the run succeeded, failed or was memory-constrained
7. what the final benchmark result is

At the end of every run, print a strong final summary like:

- benchmark mode
- machine label
- completed steps
- total elapsed time
- average step time
- peak GPU memory
- throughput
- output artifact paths
- final status

---

## Required benchmark metrics

At minimum, collect and save these metrics per run:

- machine label
- benchmark mode
- model name
- dataset name
- seed
- sequence length
- micro batch size
- gradient accumulation steps
- effective batch size
- optimizer steps completed
- training start timestamp
- training end timestamp
- total wall clock seconds
- average step time
- median step time if possible
- peak GPU memory used
- final loss
- tokens/sec
- samples/sec
- whether run succeeded
- whether run failed
- failure reason if applicable
- whether CPU offload or other fallback was used

---

## Output files required

Each run must produce its own result folder.

Example layout:

```text
results/
  gx10_lora_20260403_140000/
    config.json
    benchmark_metrics.json
    benchmark_metrics.csv
    train.log
    system_info.json
    gpu_info.json
    summary.txt
  gx10_qlora_20260403_150000/
  gx10_fullft_20260403_160000/
  rtx5090_lora_20260403_170000/
```

### Required saved artifacts per run

- `config.json`
- `benchmark_metrics.json`
- `benchmark_metrics.csv`
- `train.log`
- `summary.txt`
- `system_info.json`
- `gpu_info.json`

Also create an optional aggregated file such as:

- `results/all_runs_summary.csv`

---

## What must be timed

The benchmark timing must measure **training time only**.

### Include in timing
- training loop startup once training is actually beginning
- forward/backward/optimizer steps
- evaluation steps if evaluation is part of the benchmark design

### Do not include in timing
- model download
- tokenizer download
- dataset download
- one-time environment setup
- package installation
- Docker image build time

The benchmark should assume all dependencies are already installed and the model is already cached before the timed training run starts.

---

## Fairness rules

These fairness rules must be respected across both machines:

- same base model
- same tokenizer
- same dataset split
- same dataset preprocessing
- same seed
- same max steps
- same logging frequency
- same eval frequency
- same save frequency or disabled equally
- same prompt template
- same sequence length
- same optimizer if possible
- same learning rate schedule if possible

If a parameter must differ because of hardware constraints, the script must:

1. clearly log that difference
2. explain why it changed
3. save that explanation in the result files

---

## Preferred benchmark design

Use **fixed-step benchmarking** for the main benchmark.

Example:
- 500 steps
- or 1000 steps

This is preferred over "train until convergence" because it makes timing more comparable.

Optional:
Add a separate mode later for "time to target eval loss" but the initial implementation should focus on fixed-step benchmarking.

---

## Suggested CLI commands

Design the CLI so it is easy to use and easy to show on video.

Example command style:

```bash
python benchmark.py run \
  --machine-label gx10 \
  --mode lora \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset ./data/processed/train.jsonl \
  --eval-dataset ./data/processed/eval.jsonl \
  --max-steps 1000 \
  --seq-len 1024 \
  --micro-batch-size 1 \
  --grad-accum 16 \
  --output-dir ./results
```

Also support:

```bash
python benchmark.py run --machine-label gx10 --mode qlora ...
python benchmark.py run --machine-label gx10 --mode fullft ...
python benchmark.py run --machine-label rtx5090 --mode lora ...
```

### Nice to have subcommands

```bash
python benchmark.py run ...
python benchmark.py compare --results-dir ./results
python benchmark.py inspect --run ./results/gx10_lora_20260403_140000
```

---

## Integration with existing repository structure

The new CUDA implementation should be inspired by the existing ROCm repo structure, but it does not need to copy it blindly.

You may reuse or adapt the repository's staged workflow such as:

- prelim / cache
- prepare / preprocess
- train
- compare

However, the resulting CUDA benchmark tooling must clearly support the three required modes:

- LoRA
- QLoRA
- Full fine-tune

---

## Technical expectations

### Preferred stack
- Python
- PyTorch
- Hugging Face Transformers: `https://huggingface.co/docs/transformers/index`
- PEFT: `https://huggingface.co/docs/peft/index`
- Datasets: `https://huggingface.co/docs/datasets/index`
- Accelerate: `https://huggingface.co/docs/accelerate/index`
- bitsandbytes for QLoRA if needed: `https://huggingface.co/docs/bitsandbytes/index`
- rich for terminal output: `https://rich.readthedocs.io/en/stable/`

### CUDA expectations
The implementation must work on NVIDIA CUDA systems.

It should:
- detect CUDA availability
- print CUDA device information
- print GPU name
- print total VRAM
- print CUDA version if available
- log GPU memory usage during training

---

## Training implementation guidance

### LoRA mode
- load full precision or mixed precision base model
- freeze base weights
- attach LoRA adapters
- train adapters only

### QLoRA mode
- load quantized base model in 4-bit
- attach LoRA adapters
- train adapters only
- use CUDA-compatible quantization path

### Full FT mode
- load base model normally
- set all weights trainable
- run actual full parameter optimization

---

## Memory and failure handling

This script must handle memory failures gracefully.

If a run fails due to OOM or another hardware limitation:

- print a clear failure banner
- explain the likely cause
- save the failure reason in metrics/logs
- mark status as failed
- still write the partial benchmark record
- do not crash without explanation

Example statuses:
- `success`
- `oom`
- `failed`
- `interrupted`
- `fallback_used`

---

## Logging behavior

The script must log to both:

1. terminal
2. file

Everything important shown in the terminal should also be preserved in `train.log`.

Periodic progress updates should be easy to follow.

Example progress ideas:
- every N steps
- at evaluation points
- at save points
- at start/end of each major phase

---

## System information capture

Before each run, collect and save:

- hostname
- OS
- Python version
- PyTorch version
- CUDA version
- GPU name
- GPU count
- GPU memory size
- CPU model if possible
- RAM amount if possible

Store these in:
- `system_info.json`
- `gpu_info.json`

---

## Compare mode requirement

Add a comparison command or script that reads completed runs and outputs a simple comparison table.

Example comparison fields:
- machine label
- mode
- elapsed time
- avg step time
- peak memory
- final loss
- tokens/sec
- status

Also export a CSV summary.

---

## Code quality requirements

The implementation must be:

- clean
- modular
- easy to run from CLI
- easy to read
- easy to modify later
- robust against partial failures

Use clear file organization and comments where helpful.

---

## Suggested project structure

Example:

```text
benchmark_cuda/
  benchmark.py
  trainer.py
  modes/
    lora.py
    qlora.py
    fullft.py
  utils/
    logging_utils.py
    metrics.py
    system_info.py
    gpu_monitor.py
    config.py
  requirements.txt
  README.md
```

This is only a suggestion.  
A better structure is welcome if it improves clarity.

---

## What the final solution must include

The final output you generate must include:

1. The complete Python CLI benchmark script or script suite
2. Any helper modules needed
3. A `requirements.txt`
4. Clear setup instructions
5. Example commands for all three modes
6. Notes on how to adapt paths for my environment
7. Result file format details
8. Comparison usage instructions

---

## Benchmark outcome expectations

The completed tool must allow me to produce a benchmark result set like this:

- GX10 LoRA timing
- GX10 QLoRA timing
- GX10 Full FT timing
- RTX 5090 LoRA timing
- RTX 5090 QLoRA timing
- RTX 5090 Full FT timing

And then compare:
- total elapsed time
- average step time
- peak memory use
- throughput
- completion success/failure

---

## Strong preference on usability

I want the script to feel polished when run from the CLI.

Think of it as a benchmark demo tool, not just an internal training utility.

It should be easy for someone watching the terminal recording to understand:
- what command is being run
- what stage the system is in
- whether the benchmark is progressing well
- what the result means at the end

---

## What not to do

Do not:

- build a GUI
- focus on ROCm
- skip QLoRA
- fake full fine-tuning using LoRA
- produce bare minimum logs
- produce messy unreadable console output
- hide failures
- omit machine/system details
- include download/setup time in benchmark timing

---

## Final instruction

Please generate the full CUDA-based benchmark script suite and supporting files based on the requirements above.

The implementation must be practical, runnable from the CLI and designed for clear benchmark recording on video.

Where design choices are needed, prefer:
- clarity
- reproducibility
- benchmark fairness
- strong terminal UX
- robust logging
- modular code structure


---

## Additional required section: model choice, dataset choice and evaluation workflow

This section is mandatory and must be implemented in the final solution.

The benchmark must not stop at training time only.  
It must also include a clear **evaluation workflow** that compares:

- the **non-fine-tuned baseline model**
- the **fine-tuned model**

using a fixed set of **preset evaluation questions**.

The evaluation must be reproducible, saved to disk and easy to present in a YouTube recording.

---

## Required base model choice

Use this base model for the benchmark:

- `meta-llama/Llama-3.1-8B-Instruct`
- Model page: `https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct`

### Why this model
Use this model because:

1. it matches the existing repository direction
2. it is a strong instruction-tuned baseline
3. it is widely used in the Hugging Face ecosystem
4. it is suitable for LoRA and QLoRA workflows
5. it gives a realistic benchmark target for CUDA systems

### Important note
This is the **starting model before fine-tuning**.  
Do not call this "pre-training". This benchmark is for **fine-tuning**, not pre-training.

### Access note
The script and docs should clearly mention that this model may require:
- Hugging Face access approval
- a valid Hugging Face token

---

## Required dataset choice

Use this dataset for the benchmark:

- `databricks/databricks-dolly-15k`
- Dataset page: `https://huggingface.co/datasets/databricks/databricks-dolly-15k`

### Why this dataset
Use this dataset because:

1. it is openly available
2. it is instruction-tuning friendly
3. it is a manageable size for repeatable benchmarking
4. it includes useful task categories such as:
   - brainstorming
   - classification
   - closed QA
   - open QA
   - generation
   - information extraction
   - summarization
5. it is well suited for both timing benchmarks and simple qualitative evaluation

### Dataset fields
The implementation should expect fields equivalent to:
- instruction
- context or input
- response or output
- category

If the exact column names differ in the source dataset loader, normalize them into a standard internal format.

---

## Dataset split requirements

The final solution must create reproducible benchmark splits from the chosen dataset.

Use a fixed random seed and generate:

- **train split**: 90%
- **validation split**: 5%
- **test split**: 5%

### Split rules
- use a fixed seed such as `42`
- save the split manifest to disk
- save the processed train, validation and test JSONL files
- ensure the same split is used on both machines
- ensure the same split is used across LoRA, QLoRA and Full FT

The split process must be deterministic.

---

## Prompt formatting requirement

The training and evaluation prompt format must be fixed and reused consistently.

A recommended internal normalized structure is:

```text
### Instruction:
{instruction}

### Input:
{input_or_context_if_any}

### Response:
{target_response}
```

For records with empty context/input, omit the input section cleanly or leave it blank consistently.

The exact formatting used for training must also be used for evaluation prompts, except evaluation prompts must **not** include the target response.

---

## Required evaluation design

The benchmark must include both:

1. **automatic evaluation**
2. **side-by-side generation comparison**

### A. Automatic evaluation
Run the baseline model and the fine-tuned model on the same held-out test prompts.

At minimum compute:
- loss on validation or test set if feasible
- ROUGE-L
- BLEU
- exact match where applicable
- normalized string match where applicable

Because many instruction-following tasks are open-ended, also include:
- category-wise metrics
- average generated length
- inference latency per answer if practical

### B. Side-by-side generation comparison
Create a preset evaluation set of prompts and run both:

- baseline model
- fine-tuned model

on the same questions.

Save:
- prompt
- reference answer
- baseline answer
- fine-tuned answer
- category
- response latency
- optional simple score fields

This is required because it is much easier to show on video.

---

## Preset question requirement

The implementation must create a fixed evaluation set from the held-out test split.

### Required preset evaluation set
Create a reproducible preset set of **at least 50 questions** from the test split.

Preferred target:
- **70 questions total**
- **10 questions per category** if category balance allows

If exact category balance is not possible, sample as evenly as possible and log the actual category distribution.

### Rules for preset questions
- use held-out test split only
- do not use training examples
- keep the same preset questions across all runs
- save them to a file such as:
  - `data/eval/preset_questions.jsonl`

Each preset question record should include:
- id
- category
- instruction
- input/context
- reference output

---

## Evaluation comparison requirement

The final solution must evaluate **before and after fine-tuning**.

For every benchmark run:

1. load the baseline model
2. run inference on the preset questions
3. save baseline answers
4. load the fine-tuned model
5. run inference on the same preset questions
6. save fine-tuned answers
7. generate a side-by-side comparison report

This is a required part of the workflow.

---

## Evaluation artifacts required

Each benchmark run must save evaluation artifacts such as:

```text
results/
  gx10_lora_20260403_140000/
    evaluation/
      preset_questions.jsonl
      baseline_predictions.jsonl
      finetuned_predictions.jsonl
      side_by_side_comparison.jsonl
      evaluation_metrics.json
      evaluation_metrics.csv
      evaluation_summary.md
      evaluation_table.html
```

### Required evaluation files

- `preset_questions.jsonl`
- `baseline_predictions.jsonl`
- `finetuned_predictions.jsonl`
- `side_by_side_comparison.jsonl`
- `evaluation_metrics.json`
- `evaluation_metrics.csv`
- `evaluation_summary.md`

Optional but strongly recommended:
- `evaluation_table.html`

---

## Strong requirement for video-friendly comparison output

At the end of evaluation, print a clean CLI summary that shows:

- machine label
- benchmark mode
- model
- dataset
- number of preset questions
- baseline average metrics
- fine-tuned average metrics
- delta or improvement where applicable
- path to saved evaluation files

Also print a short side-by-side preview for a few samples, for example:

```text
[Sample 01] Category: closed_qa
Question: Who gave the UN the land in New York for its HQ?
Reference: John D. Rockefeller Jr.
Baseline: ...
Fine-tuned: ...
```

Keep this readable and camera-friendly.

---

## Preferred evaluation metrics design

Because the dataset includes multiple instruction categories, implement metrics in two layers.

### Layer 1: overall metrics
- overall ROUGE-L
- overall BLEU
- overall average latency
- overall average output length
- total evaluated samples

### Layer 2: category metrics
Compute the same metrics per category where practical:
- brainstorming
- classification
- closed QA
- generation
- information extraction
- open QA
- summarization

If a metric is not appropriate for a category, the script should note that clearly rather than failing silently.

---

## Recommended qualitative scoring extension

If practical, add a lightweight judge mode that scores:
- correctness
- completeness
- instruction following
- verbosity

This can be optional.

If implemented, it must be clearly labeled as:
- heuristic
- model-based
- non-authoritative

Do not confuse heuristic scoring with ground-truth metrics.

---

## Required inference settings for fair comparison

The baseline and fine-tuned evaluation runs must use the same inference settings, such as:
- same temperature
- same top_p
- same max_new_tokens
- same stop conditions

Recommended default for reproducibility:
- temperature = 0.0 or very low
- deterministic decoding preferred for benchmark comparison

These settings must be saved in the result config.

---

## Additional compare command requirement

The compare command should also compare evaluation outputs across runs.

For example, it should be able to summarize:

- GX10 LoRA baseline vs fine-tuned evaluation
- GX10 QLoRA baseline vs fine-tuned evaluation
- GX10 Full FT baseline vs fine-tuned evaluation
- RTX 5090 LoRA baseline vs fine-tuned evaluation
- RTX 5090 QLoRA baseline vs fine-tuned evaluation
- RTX 5090 Full FT baseline vs fine-tuned evaluation

and then produce a summary table with:
- training time
- avg step time
- peak memory
- evaluation score summary
- status

---

## Final instruction update

Please build the CUDA benchmark tool so that it includes:

1. training benchmark
2. baseline vs fine-tuned evaluation
3. preset held-out questions
4. proper model selection
5. proper dataset preparation
6. deterministic dataset split
7. evaluation artifact export
8. clean video-friendly CLI summaries

The final implementation must be practical, reproducible and easy to run entirely from the CLI.
