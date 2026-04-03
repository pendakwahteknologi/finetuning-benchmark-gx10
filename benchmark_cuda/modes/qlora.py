"""QLoRA fine-tuning mode: 4-bit quantized base + LoRA adapters."""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.config import BenchmarkConfig
from ..utils.logging_utils import print_phase, print_failure_banner, console


def setup_qlora(config: BenchmarkConfig) -> tuple:
    """Load 4-bit quantized model with LoRA adapters. Returns (model, tokenizer)."""

    print_phase("Loading Model", f"QLoRA mode - {config.model_name}")

    # Check bitsandbytes availability
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    except ImportError as e:
        print_failure_banner(
            "qlora",
            f"Missing dependency: {e}",
            "Install bitsandbytes: pip install bitsandbytes",
        )
        raise RuntimeError(f"QLoRA dependency not available: {e}")

    # Test that 4-bit actually works on this platform
    try:
        test_linear = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
        test_linear = test_linear.cuda()
        test_x = torch.randn(1, 64, dtype=torch.bfloat16, device="cuda")
        _ = test_linear(test_x)
        del test_linear, test_x
        torch.cuda.empty_cache()
    except Exception as e:
        print_failure_banner(
            "qlora",
            f"bitsandbytes 4-bit not functional on this platform: {e}",
            "This may be an ARM/aarch64 compatibility issue.",
        )
        raise RuntimeError(f"QLoRA 4-bit not functional: {e}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(config.dtype, torch.bfloat16)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.qlora_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.qlora_double_quant,
    )

    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        attn_implementation=config.attn_implementation,
        device_map="auto",
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

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

    console.print(f"  Quantization:         {config.qlora_bits}-bit {config.qlora_quant_type}")
    console.print(f"  Double quantization:  {config.qlora_double_quant}")
    console.print(f"  Total parameters:     {total:,}")
    console.print(f"  Trainable parameters: {trainable:,} ({pct:.2f}%)")
    console.print(f"  LoRA rank:            {config.lora_r}")
    console.print(f"  LoRA alpha:           {config.lora_alpha}")
    console.print()

    return model, tokenizer
