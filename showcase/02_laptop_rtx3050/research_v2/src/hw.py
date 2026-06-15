"""
Hardware-aware setup helpers.

Centralises tuning that depends on which GPU we actually got handed. On
A100 / H100 we want bf16 autocast (not fp16), TF32 matmul, channels-last
memory layout, and substantially larger batches than the RTX 3050 default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

def gpu_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    # Ampere (sm_80, A100) and later have native bf16.
    return major >= 8


def gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def gpu_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def autocast_dtype() -> torch.dtype:
    """bf16 on Ampere+, fp16 elsewhere with CUDA, fp32 on CPU."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if gpu_supports_bf16() else torch.float16


# ---------------------------------------------------------------------------
# Apply tuning
# ---------------------------------------------------------------------------

def tune_for_device(*, allow_tf32: bool = True,
                    cudnn_benchmark: bool = True) -> None:
    """Enable A100-friendly defaults. Safe no-op on CPU."""
    if not torch.cuda.is_available():
        return
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Deterministic mode is incompatible with cudnn.benchmark; the trainer
    # explicitly opts back into benchmark for throughput on fixed-shape inputs.
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = False


# ---------------------------------------------------------------------------
# A100 preset
# ---------------------------------------------------------------------------

@dataclass
class HardwarePreset:
    name: str
    batch_size: int
    grad_accum_steps: int
    lr: float
    workers: int
    channels_last: bool
    use_bf16: bool


def recommend_preset() -> HardwarePreset:
    """Return sane defaults for the GPU we landed on."""
    if not torch.cuda.is_available():
        return HardwarePreset("cpu", batch_size=8, grad_accum_steps=1, lr=5e-4,
                              workers=0, channels_last=False, use_bf16=False)
    vram = gpu_vram_gb()
    if gpu_supports_bf16() and vram >= 32:           # A100 / H100
        return HardwarePreset("a100", batch_size=256, grad_accum_steps=1,
                              lr=2e-3, workers=8, channels_last=True,
                              use_bf16=True)
    if vram >= 20:                                   # 3090 / 4090 / A6000
        return HardwarePreset("big_consumer", batch_size=128,
                              grad_accum_steps=1, lr=1.5e-3, workers=6,
                              channels_last=True, use_bf16=gpu_supports_bf16())
    if vram >= 10:                                   # 3080 12 GB, etc.
        return HardwarePreset("mid_consumer", batch_size=64,
                              grad_accum_steps=1, lr=1e-3, workers=4,
                              channels_last=False, use_bf16=gpu_supports_bf16())
    return HardwarePreset("small_consumer", batch_size=32, grad_accum_steps=2,
                          lr=1e-3, workers=2, channels_last=False,
                          use_bf16=False)


def describe_environment() -> str:
    if not torch.cuda.is_available():
        return "device=cpu"
    return (f"device=cuda  gpu={gpu_name()}  vram={gpu_vram_gb():.1f}GB  "
            f"bf16={gpu_supports_bf16()}  tf32=on")
