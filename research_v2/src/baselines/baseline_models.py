"""
Wraps a backbone + head into a single ``nn.Module`` per SOTA baseline.

The four baselines required by the project plan:
    facenet       — Inception-Resnet-V1 + Triplet (FaceNet, Schroff et al. 2015)
    arcface       — IR-50 + ArcFace head (Deng et al. 2019)
    cosface       — IR-50 + CosFace head (Wang et al. 2018)
    mobilefacenet — MobileFaceNet + ArcFace head (Chen et al. 2018)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.backbones import build_backbone
from ..models.heads import ArcFaceHead, CosFaceHead, TripletHead


class BaselineModel(nn.Module):
    """Wraps backbone + (optional) classification head."""

    def __init__(self, backbone: nn.Module, head: nn.Module, head_kind: str):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.head_kind = head_kind   # "arcface" | "cosface" | "triplet"

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return out["embedding"]

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        emb = out["embedding"]
        return self.head(emb, labels)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINE_REGISTRY = ["facenet", "arcface", "cosface", "mobilefacenet"]


def build_baseline(name: str, n_classes: int, embedding_dim: int = 512) -> BaselineModel:
    name = name.lower()
    if name == "facenet":
        backbone = build_backbone("facenet", embedding_dim=embedding_dim)
        head = TripletHead(margin=0.30)
        return BaselineModel(backbone, head, "triplet")
    if name == "arcface":
        backbone = build_backbone("ir50", embedding_dim=embedding_dim)
        head = ArcFaceHead(embedding_dim, n_classes)
        return BaselineModel(backbone, head, "arcface")
    if name == "cosface":
        backbone = build_backbone("ir50", embedding_dim=embedding_dim)
        head = CosFaceHead(embedding_dim, n_classes)
        return BaselineModel(backbone, head, "cosface")
    if name == "mobilefacenet":
        backbone = build_backbone("mobilefacenet", embedding_dim=embedding_dim)
        head = ArcFaceHead(embedding_dim, n_classes)
        return BaselineModel(backbone, head, "arcface")
    raise ValueError(f"unknown baseline {name!r}; valid: {BASELINE_REGISTRY}")
