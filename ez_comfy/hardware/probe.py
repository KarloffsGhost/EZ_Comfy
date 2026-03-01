from __future__ import annotations

import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class HardwareProfile:
    gpu_name: str
    gpu_vram_gb: float
    system_ram_gb: float
    cuda_version: str
    platform: str
    gpu_count: int = 1


def probe_hardware() -> HardwareProfile:
    gpu_name, gpu_vram_gb, cuda_version, gpu_count = _probe_gpu()
    system_ram_gb = _probe_ram()
    plat = sys.platform

    return HardwareProfile(
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        system_ram_gb=system_ram_gb,
        cuda_version=cuda_version,
        platform=plat,
        gpu_count=gpu_count,
    )


def _probe_gpu() -> tuple[str, float, str, int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return _no_gpu()

        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return _no_gpu()

        gpu_count = len(lines)
        # Use first GPU
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) < 2:
            return _no_gpu()

        gpu_name = parts[0]
        vram_mb = float(parts[1])
        gpu_vram_gb = round(vram_mb / 1024, 1)

        # Parse CUDA version from driver version
        cuda_version = "unknown"
        if len(parts) >= 3:
            driver = parts[2].strip()
            # CUDA version is derivable but nvidia-smi also exposes it separately
            cuda_version = _cuda_from_driver(driver)

        return gpu_name, gpu_vram_gb, cuda_version, gpu_count

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return _no_gpu()


def _cuda_from_driver(driver_version: str) -> str:
    # Probe CUDA version directly
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Try nvcc
        nvcc = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if nvcc.returncode == 0:
            match = re.search(r"release (\d+\.\d+)", nvcc.stdout)
            if match:
                return match.group(1)
    except Exception:
        pass
    return driver_version  # fallback: use driver as proxy


def _no_gpu() -> tuple[str, float, str, int]:
    return "No GPU detected", 0.0, "unknown", 0


def _probe_ram() -> float:
    try:
        import psutil

        mem = psutil.virtual_memory()
        return round(mem.total / (1024**3), 1)
    except ImportError:
        return _probe_ram_fallback()


def _probe_ram_fallback() -> float:
    """Fallback RAM detection without psutil."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["wmic", "OS", "get", "TotalVisibleMemorySize", "/Value"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            match = re.search(r"TotalVisibleMemorySize=(\d+)", result.stdout)
            if match:
                kb = int(match.group(1))
                return round(kb / (1024**2), 1)
        except Exception:
            pass
    return 0.0
