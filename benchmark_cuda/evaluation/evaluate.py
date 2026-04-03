"""Run baseline vs fine-tuned inference on preset questions."""

import json
import logging
import os
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.config import BenchmarkConfig
from ..utils.logging_utils import print_phase, console
from ..data.prompt_format import format_eval_prompt
from .eval_metrics import compute_all_metrics, aggregate_metrics

logger = logging.getLogger("benchmark")


def load_preset_questions(path: str) -> list:
    questions = []
    with open(path) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def run_inference(
    model,
    tokenizer,
    questions: list,
    config: BenchmarkConfig,
    label: str = "model",
) -> list:
    """Run inference on preset questions. Returns list of prediction records."""
    model.train(False)  # set to inference mode

    results = []
    total = len(questions)

    with torch.no_grad():
        for i, q in enumerate(questions):
            prompt = format_eval_prompt(q["instruction"], q.get("input", ""))

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=config.seq_len,
                truncation=True,
            )
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            start_t = time.time()
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.eval_max_new_tokens,
                temperature=config.eval_temperature if config.eval_do_sample else None,
                top_p=config.eval_top_p if config.eval_do_sample else None,
                do_sample=config.eval_do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
            latency = time.time() - start_t

            # Decode only the generated part
            new_tokens = output_ids[0][input_ids.shape[1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            results.append({
                "id": q["id"],
                "category": q["category"],
                "instruction": q["instruction"],
                "input": q.get("input", ""),
                "reference_output": q.get("reference_output", ""),
                "prediction": answer,
                "latency_sec": round(latency, 3),
                "output_tokens": len(new_tokens),
            })

            if (i + 1) % 10 == 0 or i == 0:
                console.print(
                    f"  [{label}] {i + 1}/{total} "
                    f"({(i + 1) / total * 100:.0f}%) "
                    f"latency: {latency:.2f}s"
                )

    return results


def run_model_evaluation(
    model,
    tokenizer,
    config: BenchmarkConfig,
    run_dir: str,
    model_type: str = "finetuned",
) -> dict:
    """Full evaluation pipeline for one model (baseline or finetuned)."""

    eval_dir = os.path.join(run_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Load preset questions
    questions = load_preset_questions(config.eval_preset_path)
    console.print(f"  Loaded {len(questions)} preset questions")

    # Copy preset questions to run dir
    preset_copy = os.path.join(eval_dir, "preset_questions.jsonl")
    with open(preset_copy, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    # Run inference
    print_phase(f"Inference: {model_type}", f"{len(questions)} questions")
    predictions = run_inference(model, tokenizer, questions, config, label=model_type)

    # Save predictions
    pred_file = os.path.join(eval_dir, f"{model_type}_predictions.jsonl")
    with open(pred_file, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    # Compute metrics
    scored = []
    for p in predictions:
        metrics = compute_all_metrics(p["prediction"], p["reference_output"])
        scored.append({**p, **metrics})

    aggregated = aggregate_metrics(scored)

    return {
        "predictions": scored,
        "aggregated": aggregated,
        "model_type": model_type,
    }


def run_full_evaluation(
    finetuned_model,
    tokenizer,
    config: BenchmarkConfig,
    run_dir: str,
) -> dict:
    """Run both baseline and finetuned evaluation, produce comparison."""

    print_phase("Evaluation Pipeline", "Baseline vs Fine-Tuned")

    eval_dir = os.path.join(run_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # --- Baseline inference ---
    console.print("[cyan]Loading baseline model for comparison...[/]")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map.get(config.dtype, torch.bfloat16)

    baseline_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=model_dtype,
        attn_implementation=config.attn_implementation,
        device_map="auto",
    )

    baseline_results = run_model_evaluation(
        baseline_model, tokenizer, config, run_dir, model_type="baseline"
    )

    # Free baseline model
    del baseline_model
    torch.cuda.empty_cache()

    # --- Finetuned inference ---
    finetuned_results = run_model_evaluation(
        finetuned_model, tokenizer, config, run_dir, model_type="finetuned"
    )

    # --- Side-by-side comparison ---
    from .compare_models import generate_comparison
    comparison = generate_comparison(
        baseline_results, finetuned_results, config, eval_dir
    )

    return comparison
