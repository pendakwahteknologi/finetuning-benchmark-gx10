"""Capture system and GPU information."""

import json
import os
import platform
import socket
import subprocess


def get_system_info() -> dict:
    info = {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }

    # CPU info
    try:
        if platform.machine() == "aarch64":
            result = subprocess.run(
                ["lscpu"], capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split("\n")
            cpu_models = []
            for line in lines:
                if "Model name" in line:
                    cpu_models.append(line.split(":")[-1].strip())
            info["cpu_model"] = " + ".join(cpu_models) if cpu_models else "Unknown ARM"
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        info["cpu_model"] = line.split(":")[-1].strip()
                        break
    except Exception:
        info["cpu_model"] = "Unknown"

    try:
        result = subprocess.run(["nproc"], capture_output=True, text=True, timeout=5)
        info["cpu_cores"] = int(result.stdout.strip())
    except Exception:
        info["cpu_cores"] = os.cpu_count() or 0

    # RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except Exception:
        info["ram_gb"] = 0

    # PyTorch & CUDA
    try:
        import torch

        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    except ImportError:
        info["pytorch_version"] = "NOT INSTALLED"
        info["cuda_available"] = False

    return info


def get_gpu_info() -> dict:
    info = {"gpu_count": 0, "gpus": []}

    try:
        import torch

        if not torch.cuda.is_available():
            return info

        info["gpu_count"] = torch.cuda.device_count()
        for i in range(info["gpu_count"]):
            props = torch.cuda.get_device_properties(i)
            gpu = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(props.total_memory / 1024**3, 1),
                "multi_processor_count": props.multi_processor_count,
                "compute_capability": f"{props.major}.{props.minor}",
            }
            info["gpus"].append(gpu)
    except Exception as e:
        info["error"] = str(e)

    # C2C / unified memory detection
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q"], capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split("\n"):
            if "C2C Mode" in line:
                info["c2c_mode"] = line.split(":")[-1].strip()
            if "Product Architecture" in line:
                info["architecture"] = line.split(":")[-1].strip()
            if "Addressing Mode" in line:
                info["addressing_mode"] = line.split(":")[-1].strip()
    except Exception:
        pass

    # NVCC version
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "release" in line:
                info["nvcc_version"] = line.strip()
                break
    except Exception:
        info["nvcc_version"] = "N/A"

    # Driver version
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        info["driver_version"] = result.stdout.strip()
    except Exception:
        info["driver_version"] = "N/A"

    return info


def save_system_info(run_dir: str) -> tuple:
    os.makedirs(run_dir, exist_ok=True)

    sys_info = get_system_info()
    sys_path = os.path.join(run_dir, "system_info.json")
    with open(sys_path, "w") as f:
        json.dump(sys_info, f, indent=2)

    gpu_info = get_gpu_info()
    gpu_path = os.path.join(run_dir, "gpu_info.json")
    with open(gpu_path, "w") as f:
        json.dump(gpu_info, f, indent=2)

    return sys_info, gpu_info
