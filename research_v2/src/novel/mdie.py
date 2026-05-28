"""
MDIE — Modification-Disentangled Identity Encoder.

The full novel architecture:

      Image ─► IR-50 backbone ─► feature_maps (B, 512, 7, 7)
                                  │
                                  ▼
                        RegionAwareTokenAttention   ─►  attended map
                                  │                        │
                                  ▼                        │
                       global pool + linear   ─►   identity embedding
                                  │
                                  ├──► ArcFace head (identity classification)
                                  │
                                  ├──► Modification classifier head (gradient-reversed)
                                  │
                                  └──► fed to ICCL by the trainer
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.backbones import build_backbone
from ..models.heads import ArcFaceHead
from ..models.losses import GradientReversal
from .region_attention import RegionAwareTokenAttention


class MDIE(nn.Module):
    def __init__(self,
                 n_identity_classes: int,
                 n_modification_classes: int = 9,
                 embedding_dim: int = 512,
                 use_region_prior: bool = True,
                 use_amd: bool = True,
                 amd_lambda: float = 0.10,
                 backbone: str = "ir50"):
        super().__init__()
        self.use_region_prior = use_region_prior
        self.use_amd = use_amd
        self.amd_lambda = amd_lambda

        self.backbone = build_backbone(backbone, embedding_dim=embedding_dim, return_maps=True)

        # IR-50 produces 512×7×7 feature maps; MobileFaceNet produces 512×7×7 also.
        if use_region_prior:
            self.attn = RegionAwareTokenAttention(channels=512, dim=256, heads=4,
                                                   layers=2, grid=7, n_regions=7)
            self.post_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(512, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim))

        self.identity_head = ArcFaceHead(embedding_dim, n_identity_classes)

        if use_amd:
            self.grl = GradientReversal(lambda_=amd_lambda)
            self.mod_head = nn.Sequential(
                nn.Linear(embedding_dim, 256), nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, n_modification_classes))

    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor):
        out = self.backbone(x)
        if self.use_region_prior:
            attended, attn_map = self.attn(out["feature_maps"])
            emb = self.post_pool(attended)
            emb = F.normalize(emb, dim=1)
            return emb, attn_map
        return out["embedding"], None

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        emb, _ = self.encode(x)
        return emb

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, identity_labels: torch.Tensor,
                modification_labels: torch.Tensor | None = None):
        emb, attn_map = self.encode(x)
        outputs = {"embedding": emb, "attn": attn_map}
        outputs["loss_identity"] = self.identity_head(emb, identity_labels)
        if self.use_amd and modification_labels is not None:
            mod_logits = self.mod_head(self.grl(emb))
            outputs["loss_mod"] = F.cross_entropy(mod_logits, modification_labels)
            outputs["mod_logits"] = mod_logits
        return outputs
