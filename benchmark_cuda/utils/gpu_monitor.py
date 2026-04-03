"""Background GPU memory monitoring thread."""

import threading
import time
from dataclasses import dataclass, field


@dataclass
class GPUStats:
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    samples: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, current_mb: float):
        with self._lock:
            self.current_memory_mb = current_mb
            if current_mb > self.peak_memory_mb:
                self.peak_memory_mb = current_mb
            self.samples.append(current_mb)

    def get_peak_gb(self) -> float:
        with self._lock:
            return self.peak_memory_mb / 1024

    def get_current_gb(self) -> float:
        with self._lock:
            return self.current_memory_mb / 1024


class GPUMonitor:
    def __init__(self, device_index: int = 0, interval: float = 0.5):
        self.device_index = device_index
        self.interval = interval
        self.stats = GPUStats()
        self._stop_event = threading.Event()
        self._thread = None

    def _monitor_loop(self):
        import torch

        while not self._stop_event.is_set():
            try:
                allocated = torch.cuda.memory_allocated(self.device_index) / 1024**2
                self.stats.update(allocated)
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def start(self):
        import torch

        if not torch.cuda.is_available():
            return
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def snapshot(self) -> dict:
        import torch

        try:
            return {
                "allocated_mb": round(torch.cuda.memory_allocated(self.device_index) / 1024**2, 1),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated(self.device_index) / 1024**2, 1),
                "reserved_mb": round(torch.cuda.memory_reserved(self.device_index) / 1024**2, 1),
            }
        except Exception:
            return {}
