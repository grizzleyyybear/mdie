"""
Project-wide configuration.

All paths, hyperparameters, and reproducibility seeds live here so that every
script in the project sees the same values.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


# ---------- Paths ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]              # research_v2/
DATA_DIR = ROOT / "datasets_cache"
# Output dirs default to research_v2/{results,figures}. They are env-overridable
# (MDIE_RESULTS_DIR / MDIE_FIGURES_DIR) so parallel fan-out jobs can write to
# isolated directories; unset = the original in-repo paths (nothing changes).
RESULTS_DIR = Path(os.environ.get("MDIE_RESULTS_DIR", ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("MDIE_FIGURES_DIR", ROOT / "figures"))
CKPT_DIR = Path(os.environ.get("MDIE_CKPT_DIR", ROOT / "checkpoints"))
PAPER_DIR = ROOT / "paper"

for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, CKPT_DIR, PAPER_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------- Hardware ---------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = get_device()
USE_AMP = torch.cuda.is_available()


# ---------- Reproducibility --------------------------------------------------

SEED = 42


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------- Experiment hyperparameters --------------------------------------

@dataclass
class TrainConfig:
    image_size: int = 112
    batch_size: int = 32           # safe for 4 GB VRAM at fp16
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 5e-4
    warmup_epochs: int = 2
    grad_accum_steps: int = 2      # effective batch 64
    embedding_dim: int = 512
    arcface_margin: float = 0.50
    arcface_scale: float = 64.0
    cosface_margin: float = 0.35
    triplet_margin: float = 0.30


@dataclass
class EvalConfig:
    n_pos_pairs: int = 3000
    n_neg_pairs: int = 3000
    far_targets: tuple = (1e-1, 1e-2, 1e-3)


@dataclass
class NovelConfig:
    use_region_prior: bool = True
    use_amd: bool = True            # adversarial modification disentanglement
    use_iccl: bool = True           # identity-consistency contrastive loss
    amd_lambda: float = 0.10
    iccl_lambda: float = 0.50
    n_modification_classes: int = 10   # see data.modifications


@dataclass
class Settings:
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    novel: NovelConfig = field(default_factory=NovelConfig)


SETTINGS = Settings()
