"""LoRA fine-tuning mode: freeze base, attach PEFT adapters."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from ..utils.config import BenchmarkConfig
from ..utils.logging_utils import print_phase, console


def setup_lora(config: BenchmarkConfig) -> tuple:
    """Load model with LoRA adapters. Returns (model, tokenizer)."""

    print_phase("Loading Model", f"LoRA mode - {config.model_name}")

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

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=model_dtype,
        attn_implementation=config.attn_implementation,
        device_map="auto",
    )

    # Freeze base weights
    for param in model.parameters():
        param.requires_grad = False

    # Attach LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    pct = trainable / total * 100

    console.print(f"  Total parameters:     {total:,}")
    console.print(f"  Trainable parameters: {trainable:,} ({pct:.2f}%)")
    console.print(f"  LoRA rank:            {config.lora_r}")
    console.print(f"  LoRA alpha:           {config.lora_alpha}")
    console.print()

    return model, tokenizer
