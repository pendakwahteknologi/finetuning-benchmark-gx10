"""Perplexity computation for baseline and fine-tuned models.

Computes perplexity (exp of average cross-entropy loss) on a held-out test set.
Can be called during training (model in memory) or after training (loads from disk).
"""

import json
import math
import os
from typing import Optional

import torch
from rich.table import Table

from ..data.prompt_format import format_train_prompt
from ..utils.logging_utils import console


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _load_test_samples(data_path: str, max_samples: int = 200) -> list[dict]:
    """Load samples from a JSONL file (instruction / input / output fields)."""
    samples = []
    with open(data_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            samples.append(rec)
            if len(samples) >= max_samples:
                break
    return samples


# ---------------------------------------------------------------------------
# Core perplexity computation
# ---------------------------------------------------------------------------

def compute_perplexity(
    model,
    tokenizer,
    data_path: str,
    seq_len: int = 1024,
    max_samples: int = 200,
    batch_size: int = 4,
) -> float:
    """Compute perplexity on a JSONL test set.

    Args:
        model: A HuggingFace causal-LM (already on the correct device).
        tokenizer: Matching tokenizer.
        data_path: Path to a JSONL file with instruction/input/output fields.
        seq_len: Maximum sequence length for tokenisation.
        max_samples: Cap on the number of test samples to use.
        batch_size: Mini-batch size for forward passes.

    Returns:
        Perplexity (float) = exp(mean cross-entropy loss).
    """
    samples = _load_test_samples(data_path, max_samples)
    if not samples:
        console.print("[warning]No test samples found -- returning inf perplexity[/warning]")
        return float("inf")

    # Build full-text strings using the training prompt format
    texts = [
        format_train_prompt(
            s.get("instruction", ""),
            s.get("input", ""),
            s.get("output", ""),
        )
        for s in samples
    ]

    device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    console.print(
        f"[info]Computing perplexity on {len(texts)} samples "
        f"(batch_size={batch_size}, seq_len={seq_len})...[/info]"
    )

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_len,
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        # Labels: same as input_ids but with padding tokens masked out
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            # outputs.loss is the mean cross-entropy over non-ignored tokens.
            # Weight by valid-token count for a correct global mean.
            valid_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * valid_tokens
            total_tokens += valid_tokens

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(texts))
            console.print(
                f"  [{done}/{len(texts)}] running avg loss = "
                f"{total_loss / max(total_tokens, 1):.4f}"
            )

    # Restore original training state
    if model_was_training:
        model.train()

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    console.print(
        f"[success]Perplexity: {perplexity:.4f}  "
        f"(avg CE loss: {avg_loss:.4f}, tokens: {total_tokens})[/success]"
    )
    return perplexity


# ---------------------------------------------------------------------------
# Post-training assessment (loads models from disk)
# ---------------------------------------------------------------------------

def _find_best_run(results_dir: str, mode: str) -> Optional[str]:
    """Find the most recent successful run directory for *mode*.

    Uses the same logic as cross_compare.py: scan results_dir for directories
    whose name contains ``_{mode}_`` and whose summary.txt includes "success".
    Returns the latest match (sorted alphabetically by dirname which embeds a
    timestamp).
    """
    candidates = []
    if not os.path.isdir(results_dir):
        return None
    for entry in sorted(os.listdir(results_dir)):
        if f"_{mode}_" not in entry:
            continue
        run_dir = os.path.join(results_dir, entry)
        summary = os.path.join(run_dir, "summary.txt")
        if os.path.isfile(summary):
            with open(summary) as f:
                if "Status:               success" in f.read():
                    candidates.append(run_dir)
    return candidates[-1] if candidates else None


def run_perplexity_assessment(
    results_dir: str,
    test_data_path: str = "./data/processed/test.jsonl",
) -> dict:
    """Run perplexity measurement for every successful mode found in *results_dir*.

    NOTE: This function attempts to load models from saved checkpoints.  Because
    ``save_checkpoints`` defaults to ``False`` in BenchmarkConfig, fine-tuned
    model weights are typically **not** persisted.  In that scenario only the
    baseline perplexity will be computed.  For a full baseline-vs-finetuned
    comparison, use :func:`compute_and_save_perplexity` *during* training while
    the model is still loaded on the GPU.

    Returns:
        dict mapping mode -> perplexity results dict.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODE_ORDER = ["lora", "qlora", "fullft"]
    all_results = {}

    for mode in MODE_ORDER:
        run_dir = _find_best_run(results_dir, mode)
        if run_dir is None:
            console.print(f"[warning]No successful {mode} run found -- skipping.[/warning]")
            continue

        config_path = os.path.join(run_dir, "config.json")
        if not os.path.isfile(config_path):
            console.print(f"[warning]No config.json in {run_dir} -- skipping.[/warning]")
            continue

        with open(config_path) as f:
            cfg = json.load(f)

        model_name = cfg.get("model_name", "meta-llama/Llama-3.1-8B")
        dtype_str = cfg.get("dtype", "bfloat16")
        torch_dtype = DTYPE_MAP.get(dtype_str, torch.bfloat16)
        seq_len = cfg.get("seq_len", 1024)

        console.print(f"\n[header]===  {mode.upper()}  ({os.path.basename(run_dir)})  ===[/header]")

        # --- Baseline model ---
        console.print(f"[info]Loading baseline model: {model_name}[/info]")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        baseline_ppl = compute_perplexity(base_model, tokenizer, test_data_path, seq_len=seq_len)

        # --- Fine-tuned model ---
        finetuned_ppl = None
        if mode in ("lora", "qlora"):
            # Look for a PEFT adapter directory
            adapter_dir = os.path.join(run_dir, "checkpoint")
            if not os.path.isdir(adapter_dir):
                for candidate in ["adapter", "peft_model", "final_model"]:
                    alt = os.path.join(run_dir, candidate)
                    if os.path.isdir(alt):
                        adapter_dir = alt
                        break

            if os.path.isdir(adapter_dir) and os.path.isfile(
                os.path.join(adapter_dir, "adapter_config.json")
            ):
                console.print(f"[info]Loading PEFT adapter from {adapter_dir}[/info]")
                from peft import PeftModel

                ft_model = PeftModel.from_pretrained(base_model, adapter_dir)
                ft_model.eval()
                finetuned_ppl = compute_perplexity(
                    ft_model, tokenizer, test_data_path, seq_len=seq_len
                )
                del ft_model
            else:
                console.print(
                    "[warning]No saved adapter found (save_checkpoints is likely False). "
                    "Use compute_and_save_perplexity() during training instead.[/warning]"
                )
        elif mode == "fullft":
            model_dir = os.path.join(run_dir, "checkpoint")
            if not os.path.isdir(model_dir):
                for candidate in ["final_model", "model"]:
                    alt = os.path.join(run_dir, candidate)
                    if os.path.isdir(alt):
                        model_dir = alt
                        break

            if os.path.isdir(model_dir) and os.path.isfile(
                os.path.join(model_dir, "config.json")
            ):
                console.print(f"[info]Loading full fine-tuned model from {model_dir}[/info]")
                ft_model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
                finetuned_ppl = compute_perplexity(
                    ft_model, tokenizer, test_data_path, seq_len=seq_len
                )
                del ft_model
            else:
                console.print(
                    "[warning]No saved full-FT model found (save_checkpoints is likely False). "
                    "Use compute_and_save_perplexity() during training instead.[/warning]"
                )

        # Free baseline model memory
        del base_model
        torch.cuda.empty_cache()

        result = {
            "baseline_perplexity": round(baseline_ppl, 4),
            "finetuned_perplexity": round(finetuned_ppl, 4) if finetuned_ppl is not None else None,
            "mode": mode,
            "model_name": model_name,
            "run_dir": run_dir,
        }
        if finetuned_ppl is not None:
            result["delta"] = round(finetuned_ppl - baseline_ppl, 4)
            result["improvement_pct"] = round(
                (finetuned_ppl - baseline_ppl) / baseline_ppl * 100, 2
            )

        all_results[mode] = result

    return all_results


# Keep the name the user specified as a public alias
run_perplexity_evaluation = run_perplexity_assessment


# ---------------------------------------------------------------------------
# In-training perplexity (model already in memory)
# ---------------------------------------------------------------------------

def compute_and_save_perplexity(
    model,
    tokenizer,
    config,
    run_dir: str,
    test_data_path: str = "./data/processed/test.jsonl",
) -> dict:
    """Compute perplexity for an in-memory model and its baseline, then save.

    This is the recommended entry-point: call it at the end of a training run
    while the fine-tuned model is still loaded on the GPU.

    Args:
        model: The fine-tuned model (already on device).
        tokenizer: Matching tokenizer.
        config: A :class:`BenchmarkConfig` instance (or anything with
                ``model_name``, ``dtype``, ``seq_len`` attributes/keys).
        run_dir: Directory for the current training run.
        test_data_path: Path to test JSONL file.

    Returns:
        dict with baseline_perplexity, finetuned_perplexity, delta, etc.
    """
    from transformers import AutoModelForCausalLM

    # Allow config to be a dataclass or a plain dict
    if hasattr(config, "model_name"):
        model_name = config.model_name
        dtype_str = config.dtype
        seq_len = config.seq_len
    else:
        model_name = config.get("model_name", "meta-llama/Llama-3.1-8B")
        dtype_str = config.get("dtype", "bfloat16")
        seq_len = config.get("seq_len", 1024)

    torch_dtype = DTYPE_MAP.get(dtype_str, torch.bfloat16)

    if not os.path.isfile(test_data_path):
        console.print(
            f"[warning]Test data not found at {test_data_path} -- skipping perplexity.[/warning]"
        )
        return {}

    # --- Fine-tuned model perplexity ---
    console.print("\n[header]Perplexity Measurement[/header]")
    console.print("[info]Computing fine-tuned model perplexity...[/info]")
    finetuned_ppl = compute_perplexity(model, tokenizer, test_data_path, seq_len=seq_len)

    # --- Baseline model perplexity ---
    console.print(f"[info]Loading baseline model ({model_name}) for comparison...[/info]")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    baseline_ppl = compute_perplexity(baseline_model, tokenizer, test_data_path, seq_len=seq_len)

    del baseline_model
    torch.cuda.empty_cache()

    # --- Compute deltas ---
    delta = finetuned_ppl - baseline_ppl
    improvement_pct = (delta / baseline_ppl * 100) if baseline_ppl > 0 else 0.0

    test_samples = len(_load_test_samples(test_data_path, max_samples=200))

    results = {
        "baseline_perplexity": round(baseline_ppl, 4),
        "finetuned_perplexity": round(finetuned_ppl, 4),
        "delta": round(delta, 4),
        "improvement_pct": round(improvement_pct, 2),
        "test_samples": test_samples,
        "test_data": os.path.basename(test_data_path),
    }

    # --- Save to disk ---
    output_dir = os.path.join(run_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "perplexity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[info]Perplexity results saved to {out_path}[/info]")

    # --- Pretty-print a Rich table ---
    table = Table(title="Perplexity Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Model", style="bold")
    table.add_column("Perplexity", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Change %", justify="right")

    table.add_row(
        "Baseline",
        f"{baseline_ppl:.4f}",
        "--",
        "--",
    )

    delta_style = "green" if delta < 0 else ("red" if delta > 0 else "white")
    table.add_row(
        "Fine-tuned",
        f"{finetuned_ppl:.4f}",
        f"[{delta_style}]{delta:+.4f}[/{delta_style}]",
        f"[{delta_style}]{improvement_pct:+.2f}%[/{delta_style}]",
    )

    console.print(table)

    return results
