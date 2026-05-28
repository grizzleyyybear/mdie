"""
Pretrained seed loader for the **face.evoLVe IR-50** backbone used by MDIE
training (``src/models/backbones.py::IR50``).

Honest summary:

* I could not find a public PyTorch checkpoint on the open web whose key
  layout matches face.evoLVe's IR-50. Every "ArcFace IR-50" file on
  HuggingFace turned out to be InsightFace's ``iresnet50`` layout, which
  is a different module tree and is **not** a drop-in here.

* For the *evaluation* side this does not matter — the eval harness
  exposes ``insightface_w600k_r50`` as a strong external baseline that
  auto-downloads from HuggingFace (see ``src/models/iresnet.py``).

* For *seeding MDIE training*, two options remain:
    1. Drop a face.evoLVe-layout state_dict at
       ``research_v2/checkpoints/ir50_pretrained.pth`` (any
       ``{state_dict|model|module|backbone.}`` wrapping is unwrapped).
    2. Run ``python -m research_v2.src.train.pretrain_backbone
       --from-scratch <imagefolder>`` on a real GPU server.

If neither happens, ``load_pretrained_ir50()`` returns ``None`` and the
caller proceeds with a freshly-initialised IR-50.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .config import CKPT_DIR
from .models.backbones import IR50


def _find_local() -> Optional[Path]:
    candidates = [
        CKPT_DIR / "ir50_pretrained.pth",
        CKPT_DIR / "backbone_ir50_ms1m_epoch120.pth",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    for p in CKPT_DIR.glob("*ir50*.pth"):
        if p.stat().st_size > 1_000_000:
            return p
    return None


def _strip_prefix(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _coerce_state_dict(raw) -> dict:
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    if not isinstance(raw, dict):
        raise ValueError(f"unexpected checkpoint type: {type(raw)}")
    for prefix in ("module.", "backbone."):
        if any(k.startswith(prefix) for k in raw):
            raw = _strip_prefix(raw, prefix)
    return raw


def load_pretrained_ir50(embedding_dim: int = 512,
                          return_maps: bool = False) -> Optional[IR50]:
    """Return an IR-50 with a locally-dropped pretrained state_dict, or
    ``None`` if no compatible file is present in ``CKPT_DIR``."""
    local = _find_local()
    if local is None:
        print("  [pretrained] no local IR-50 checkpoint found in "
              f"{CKPT_DIR} — see checkpoints/README.md")
        return None
    try:
        raw = torch.load(local, map_location="cpu", weights_only=False)
        sd = _coerce_state_dict(raw)
        model = IR50(embedding_dim=embedding_dim, return_maps=return_maps)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  [pretrained] loaded {local.name}  "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > len(model.state_dict()) // 2:
            print("  [pretrained] WARNING: more than half the keys are missing "
                  "— this checkpoint is probably the wrong layout "
                  "(InsightFace iresnet50 vs face.evoLVe IR-50).")
        return model
    except Exception as e:  # noqa: BLE001
        print(f"  [pretrained] {local.name} failed to load: {e}")
        return None
