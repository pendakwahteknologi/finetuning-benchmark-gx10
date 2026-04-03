"""Benchmark configuration dataclass with sensible defaults."""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import os
from datetime import datetime


@dataclass
class BenchmarkConfig:
    # Identity
    machine_label: str = "gx10"
    mode: str = "lora"  # lora | qlora | fullft

    # Model & data
    model_name: str = "meta-llama/Llama-3.1-8B"
    dataset_name: str = "databricks/databricks-dolly-15k"
    data_dir: str = "./data/processed"
    eval_preset_path: str = "./data/eval/preset_questions.jsonl"

    # Training hyperparams
    max_steps: int = 500
    warmup_steps: int = 3  # excluded from timing metrics
    seq_len: int = 1024
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    seed: int = 42

    # LoRA / QLoRA params
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # QLoRA quantization
    qlora_bits: int = 4
    qlora_quant_type: str = "nf4"
    qlora_double_quant: bool = True

    # Precision & hardware
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = False
    pin_memory: bool = False
    dataloader_workers: int = 4

    # Logging & eval
    logging_steps: int = 10
    eval_steps: int = 100
    save_checkpoints: bool = False
    checkpoint_steps: int = 100

    # Evaluation / inference
    eval_temperature: float = 0.0
    eval_top_p: float = 1.0
    eval_max_new_tokens: int = 256
    eval_do_sample: bool = False
    skip_eval: bool = False

    # Output
    output_dir: str = "./results"

    # Runtime (set automatically)
    run_id: str = ""
    run_dir: str = ""
    fairness_notes: list = field(default_factory=list)
    status: str = "pending"  # pending | running | success | oom | failed | interrupted | fallback_used
    failure_reason: str = ""

    def __post_init__(self):
        if not self.run_id:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{self.machine_label}_{self.mode}_{ts}"
        if not self.run_dir:
            self.run_dir = os.path.join(self.output_dir, self.run_id)

        # Mode-specific overrides
        if self.mode == "fullft":
            if self.micro_batch_size > 4:
                self.micro_batch_size = 4
                self.gradient_accumulation_steps = 8
            if self.learning_rate > 2e-5:
                self.learning_rate = 2e-5

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def add_fairness_note(self, note: str):
        self.fairness_notes.append(note)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Optional[str] = None):
        path = path or os.path.join(self.run_dir, "config.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_file(cls, path: str) -> "BenchmarkConfig":
        with open(path) as f:
            data = json.load(f)
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
