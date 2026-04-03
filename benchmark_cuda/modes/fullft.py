"""Full fine-tuning mode: all parameters trainable."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.config import BenchmarkConfig
from ..utils.logging_utils import print_phase, console


def setup_fullft(config: BenchmarkConfig) -> tuple:
    """Load model with all parameters trainable. Returns (model, tokenizer)."""

    print_phase("Loading Model", f"Full Fine-Tune mode - {config.model_name}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map.get(config.dtype, torch.bfloat16)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model — no PEFT, no quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=model_dtype,
        attn_implementation=config.attn_implementation,
        device_map="auto",
    )

    # Ensure ALL parameters are trainable
    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    console.print(f"  Total parameters:     {total_params:,}")
    console.print(f"  Trainable parameters: {trainable_params:,} (100.00%)")
    console.print(f"  Dtype:                {model_dtype}")

    # Apply gradient checkpointing if requested
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        console.print(f"  Grad checkpointing:   [yellow]ENABLED[/] (fairness note recorded)")
        config.add_fairness_note("gradient_checkpointing enabled by user for Full FT")
    else:
        console.print(f"  Grad checkpointing:   disabled")

    console.print()

    return model, tokenizer
