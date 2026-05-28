"""
Shared training utilities used by Stage 1 (baselines) and Stage 2 (MDIE).

Centralising these prevents copy-paste drift between the two trainers and
makes the long ML training runs survivable on a 4 GB consumer GPU:

  * ``warmup_cosine_lr``      — stable LR schedule for ArcFace-style margins.
  * ``is_finite_loss``        — skip a step instead of poisoning the scaler.
  * ``save_resumable``        — full optimizer + scaler + epoch + best snapshot.
  * ``load_resumable``        — picks up where the previous run died.
  * ``dump_run_manifest``     — JSON snapshot of every run for reproducibility.
  * ``autodetect_workers``    — sane DataLoader worker count for the host.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def warmup_cosine_lr(step: int, total_steps: int,
                     warmup_steps: int, base_lr: float,
                     min_lr_ratio: float = 0.01) -> float:
    """Linear warmup → cosine decay to ``min_lr_ratio * base_lr``."""
    if total_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    if step >= total_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def set_lr(optim: torch.optim.Optimizer, lr: float) -> None:
    for pg in optim.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# Stability guards
# ---------------------------------------------------------------------------

def is_finite_loss(loss: torch.Tensor) -> bool:
    """Return False for NaN / +Inf / -Inf so the caller can skip the step."""
    return bool(torch.isfinite(loss).all().item())


# ---------------------------------------------------------------------------
# Resume-aware checkpointing
# ---------------------------------------------------------------------------

def extract_state_dict(path: Path, map_location: str | torch.device = "cpu") -> dict:
    """Return a plain ``state_dict`` from any checkpoint format we write.

    Handles three layouts in the wild:
      * resumable: ``{"model": sd, "optim": ..., "scaler": ..., "epoch": ...}``
      * insightface / HF style: bare ``state_dict``
      * legacy: ``{"state_dict": sd}``
    """
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
    return obj


def save_resumable(path: Path, *, model: torch.nn.Module,
                   optim: torch.optim.Optimizer,
                   scaler: torch.amp.GradScaler | None,
                   epoch: int, best_metric: float,
                   model_config: dict | None = None) -> None:
    """Atomic ``torch.save`` of the full resumable training state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "config": model_config or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_resumable(path: Path, *, model: torch.nn.Module,
                   optim: torch.optim.Optimizer | None = None,
                   scaler: torch.amp.GradScaler | None = None,
                   map_location: str | torch.device = "cpu") -> dict:
    """Reverse of :func:`save_resumable`. Returns the loaded metadata dict."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=False)
    if optim is not None and isinstance(state, dict) and state.get("optim"):
        try:
            optim.load_state_dict(state["optim"])
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] could not restore optimizer state: {e}")
    if scaler is not None and isinstance(state, dict) and state.get("scaler"):
        try:
            scaler.load_state_dict(state["scaler"])
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] could not restore AMP scaler state: {e}")
    return {
        "epoch": int(state.get("epoch", 0)) if isinstance(state, dict) else 0,
        "best_metric": float(state.get("best_metric", float("inf"))) if isinstance(state, dict) else float("inf"),
        "config": state.get("config", {}) if isinstance(state, dict) else {},
    }


# ---------------------------------------------------------------------------
# Run manifests
# ---------------------------------------------------------------------------

def _to_json_safe(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _git_sha() -> str:
    if shutil.which("git") is None:
        return "no-git"
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def dump_run_manifest(out_dir: Path, *, run_name: str, settings: Any,
                      extra: dict | None = None) -> Path:
    """Write ``<out_dir>/run_<timestamp>_<run_name>.json`` with full context."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"run_{ts}_{run_name}.json"
    payload = {
        "run_name": run_name,
        "timestamp": ts,
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
        "argv": sys.argv,
        "settings": _to_json_safe(settings),
        "extra": _to_json_safe(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# DataLoader sizing
# ---------------------------------------------------------------------------

def autodetect_workers(requested: int | None = None) -> int:
    """Conservative worker count: respects the user, then halves the CPU count.

    Windows + spawn-start has measurable per-worker overhead, so we cap at 4.
    """
    if requested is not None:
        return max(0, int(requested))
    cpu = os.cpu_count() or 2
    return max(0, min(4, cpu // 2))


# ---------------------------------------------------------------------------
# torch.compile helper (safe on hardware that doesn't support it)
# ---------------------------------------------------------------------------

def try_torch_compile(model: torch.nn.Module, *, enable: bool,
                       mode: str = "reduce-overhead") -> torch.nn.Module:
    """Wrap ``model`` with ``torch.compile`` when enabled and supported.

    Falls back to the original module on any failure (CPU, old GPU, Windows
    Triton missing, etc.) so this never breaks a training run.
    """
    if not enable:
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        print("  [compile] torch.compile unavailable — skipping")
        return model
    try:
        compiled = compile_fn(model, mode=mode)
        print(f"  [compile] torch.compile(mode={mode!r}) active")
        return compiled
    except Exception as e:  # noqa: BLE001
        print(f"  [compile] disabled (fallback to eager): {e}")
        return model


# ---------------------------------------------------------------------------
# Throughput meter
# ---------------------------------------------------------------------------

class Throughput:
    """Running images-per-second tracker for an epoch."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.n_images = 0

    def add(self, n: int) -> None:
        self.n_images += int(n)

    def imgs_per_sec(self) -> float:
        dt = max(time.time() - self.t0, 1e-9)
        return self.n_images / dt

