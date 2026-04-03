"""Training metrics collection and export."""

import csv
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class StepRecord:
    step: int
    loss: float
    learning_rate: float
    step_time_sec: float
    gpu_memory_mb: float
    tokens_processed: int = 0
    is_warmup: bool = False


@dataclass
class BenchmarkMetrics:
    # Identity
    machine_label: str = ""
    mode: str = ""
    model_name: str = ""
    dataset_name: str = ""
    seed: int = 42
    seq_len: int = 1024
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = 1

    # Timing
    training_start: Optional[str] = None
    training_end: Optional[str] = None
    total_wall_clock_sec: float = 0.0

    # Step data
    steps_completed: int = 0
    max_steps: int = 0
    step_records: list = field(default_factory=list)

    # Aggregated metrics (computed at end)
    avg_step_time: float = 0.0
    median_step_time: float = 0.0
    p95_step_time: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    final_loss: float = 0.0
    tokens_per_sec: float = 0.0
    samples_per_sec: float = 0.0

    # Status
    status: str = "pending"
    failure_reason: str = ""
    fallbacks_used: list = field(default_factory=list)
    fairness_notes: list = field(default_factory=list)

    # Internal timing
    _start_time: float = field(default=0.0, repr=False)
    _warmup_steps: int = field(default=3, repr=False)

    def start_timer(self):
        self._start_time = time.time()
        self.training_start = datetime.now().isoformat()

    def stop_timer(self):
        self.total_wall_clock_sec = time.time() - self._start_time
        self.training_end = datetime.now().isoformat()

    def record_step(self, record: StepRecord):
        self.step_records.append(record)
        self.steps_completed = record.step
        self.final_loss = record.loss
        if record.gpu_memory_mb > self.peak_gpu_memory_mb:
            self.peak_gpu_memory_mb = record.gpu_memory_mb

    def compute_aggregates(self):
        # Only use non-warmup steps for timing stats
        timed_steps = [
            r for r in self.step_records if not r.is_warmup and r.step_time_sec > 0
        ]
        if not timed_steps:
            return

        times = [r.step_time_sec for r in timed_steps]
        self.avg_step_time = statistics.mean(times)
        self.median_step_time = statistics.median(times)
        self.p95_step_time = (
            sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max(times)
        )

        total_tokens = sum(r.tokens_processed for r in timed_steps)
        total_time = sum(times)
        if total_time > 0:
            self.tokens_per_sec = total_tokens / total_time
            self.samples_per_sec = len(timed_steps) * self.micro_batch_size / total_time

    def to_dict(self) -> dict:
        d = {
            "machine_label": self.machine_label,
            "mode": self.mode,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "seed": self.seed,
            "seq_len": self.seq_len,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "total_wall_clock_sec": round(self.total_wall_clock_sec, 2),
            "steps_completed": self.steps_completed,
            "max_steps": self.max_steps,
            "avg_step_time": round(self.avg_step_time, 4),
            "median_step_time": round(self.median_step_time, 4),
            "p95_step_time": round(self.p95_step_time, 4),
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 1),
            "peak_gpu_memory_gb": round(self.peak_gpu_memory_mb / 1024, 2),
            "final_loss": round(self.final_loss, 4),
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "samples_per_sec": round(self.samples_per_sec, 2),
            "status": self.status,
            "failure_reason": self.failure_reason,
            "fallbacks_used": self.fallbacks_used,
            "fairness_notes": self.fairness_notes,
        }
        return d

    def save(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)

        # JSON summary
        with open(os.path.join(run_dir, "benchmark_metrics.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # CSV per-step
        csv_path = os.path.join(run_dir, "benchmark_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "loss", "learning_rate", "step_time_sec",
                "gpu_memory_mb", "tokens_processed", "is_warmup",
            ])
            for r in self.step_records:
                writer.writerow([
                    r.step, round(r.loss, 4), f"{r.learning_rate:.2e}",
                    round(r.step_time_sec, 4), round(r.gpu_memory_mb, 1),
                    r.tokens_processed, r.is_warmup,
                ])

        # Human-readable summary
        self._save_summary(run_dir)

    def _save_summary(self, run_dir: str):
        d = self.to_dict()
        lines = [
            "=" * 60,
            "BENCHMARK SUMMARY",
            "=" * 60,
            f"Machine:              {d['machine_label']}",
            f"Mode:                 {d['mode']}",
            f"Model:                {d['model_name']}",
            f"Dataset:              {d['dataset_name']}",
            f"Status:               {d['status']}",
            "",
            f"Steps completed:      {d['steps_completed']} / {d['max_steps']}",
            f"Total time:           {d['total_wall_clock_sec']:.1f}s ({d['total_wall_clock_sec']/60:.1f}m)",
            f"Avg step time:        {d['avg_step_time']:.4f}s",
            f"Median step time:     {d['median_step_time']:.4f}s",
            f"P95 step time:        {d['p95_step_time']:.4f}s",
            f"Peak GPU memory:      {d['peak_gpu_memory_gb']:.2f} GB",
            f"Final loss:           {d['final_loss']:.4f}",
            f"Tokens/sec:           {d['tokens_per_sec']:.1f}",
            f"Samples/sec:          {d['samples_per_sec']:.2f}",
            "",
            f"Batch size:           {d['micro_batch_size']} x {d['gradient_accumulation_steps']} = {d['effective_batch_size']}",
            f"Sequence length:      {d['seq_len']}",
            f"Seed:                 {d['seed']}",
            "",
        ]
        if d["failure_reason"]:
            lines.append(f"Failure reason:       {d['failure_reason']}")
        if d["fallbacks_used"]:
            lines.append(f"Fallbacks used:       {', '.join(d['fallbacks_used'])}")
        if d["fairness_notes"]:
            lines.append("Fairness notes:")
            for note in d["fairness_notes"]:
                lines.append(f"  - {note}")

        lines.append("=" * 60)

        with open(os.path.join(run_dir, "summary.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
