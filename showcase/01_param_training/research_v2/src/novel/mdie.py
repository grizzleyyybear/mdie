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
from ..models.heads import ArcFaceHead, GradientReversal
from .region_attention import RegionAwareTokenAttention


class MDIE(nn.Module):
    def __init__(self,
                 n_identity_classes: int,
                 n_modification_classes: int = 9,
                 embedding_dim: int = 512,
                 use_region_prior: bool = True,
                 use_amd: bool = True,
                 amd_lambda: float = 0.10,
                 backbone: str = "ir50",
                 pretrained_backbone: bool = False,
                 freeze_backbone: bool = False,
                 residual_fusion: bool = False):
        super().__init__()
        self.use_region_prior = use_region_prior
        self.use_amd = use_amd
        self.amd_lambda = amd_lambda
        # Residual (identity-initialised) fusion. When enabled together with a
        # frozen pretrained backbone, the deployed embedding at initialisation is
        # EXACTLY the backbone's native (production) embedding -- the attention
        # pathway contributes a zero-initialised, gated residual that training
        # grows only where it helps (worn occlusion). This makes the specialist
        # start at production parity and improve on its niche, instead of the
        # random-init concat fusion that overwrites the native embedding.
        self.residual_fusion = residual_fusion

        if pretrained_backbone:
            self.backbone = build_backbone(
                "iresnet50_w600k", embedding_dim=embedding_dim,
                return_maps=True, freeze=freeze_backbone)
        else:
            self.backbone = build_backbone(
                backbone, embedding_dim=embedding_dim, return_maps=True)

        # Region attention operates on a FUSED 14x14 feature: the identity-rich
        # final 512x7x7 map (upsampled) concatenated with the spatially finer
        # layer3 256x14x14 map -> 768x14x14. The finer grid lets attention land
        # precisely on individual rigid bone landmarks instead of a coarse 7x7
        # blob, while the upsampled final features preserve identity quality.
        if use_region_prior:
            self.attn_grid = 14
            self.attn_channels = 768
            self.attn = RegionAwareTokenAttention(channels=self.attn_channels,
                                                   dim=256, heads=4,
                                                   layers=2, grid=self.attn_grid,
                                                   n_regions=7)
            self.post_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(self.attn_channels, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim))
            # Fusion head: combine the bone-anchored attention embedding with the
            # backbone's native identity embedding into a SINGLE 512-d vector.
            # This is the deployed embedding -- one forward pass, one unit-norm
            # 512-d vector, plain cosine matching -> a drop-in replacement for an
            # ArcFace encoder. The fusion is learned end-to-end under the identity
            # + ICCL objectives, so it keeps the backbone's discriminative power
            # while inheriting the attention pathway's occlusion robustness. The
            # spatial attention map is still produced and supervised (RATA loss),
            # so interpretability is unaffected.
            if self.residual_fusion:
                # Identity-initialised residual: emb = norm(native + gate * delta),
                # delta = W[attn_emb; native]. W is zero-initialised so at init
                # delta == 0 and the deployed embedding is EXACTLY the native
                # (production) embedding. The gate is initialised to 1 (not 0):
                # zero-initialising BOTH W and the gate would make the residual a
                # dead branch (grad to the gate is proportional to delta==0 and
                # grad to W is proportional to gate==0), so it could never turn
                # on. With W=0, gate=1 the projection receives a live gradient on
                # the very first step and grows the occlusion-robust correction,
                # while parity-at-init is still exact (delta==0).
                self.fuse_residual = nn.Linear(2 * embedding_dim, embedding_dim,
                                               bias=False)
                nn.init.zeros_(self.fuse_residual.weight)
                self.res_gate = nn.Parameter(torch.ones(1))
            else:
                self.fuse = nn.Sequential(
                    nn.Linear(2 * embedding_dim, embedding_dim, bias=False),
                    nn.BatchNorm1d(embedding_dim))

        self.identity_head = ArcFaceHead(embedding_dim, n_identity_classes)

        if use_amd:
            self.grl = GradientReversal(lambda_=amd_lambda)
            self.mod_head = nn.Sequential(
                nn.Linear(embedding_dim, 256), nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, n_modification_classes))

    # ------------------------------------------------------------------

    def _fuse_maps(self, out: dict) -> torch.Tensor:
        """Build the 768x14x14 fused attention input from backbone maps."""
        f7 = out["feature_maps"]                                  # (B, 512, 7, 7)
        g = self.attn_grid
        f7u = F.interpolate(f7, size=(g, g), mode="bilinear", align_corners=False)
        f_hi = out.get("feature_maps_hi")
        if f_hi is None:
            # fallback: derive a 256-ch hi map from the final features
            f_hi = f7u[:, :256]
        elif f_hi.shape[-1] != g:
            f_hi = F.interpolate(f_hi, size=(g, g), mode="bilinear",
                                 align_corners=False)
        return torch.cat([f7u, f_hi], dim=1)                     # (B, 768, g, g)

    def encode(self, x: torch.Tensor, bone_target: torch.Tensor | None = None):
        out = self.backbone(x)
        if self.use_region_prior:
            fused = self._fuse_maps(out)
            attended, attn_map = self.attn(fused, bone_target)
            attn_emb = self.post_pool(attended)               # (B, 512)
            native = out["embedding"]                         # (B, 512)
            if self.residual_fusion:
                delta = self.fuse_residual(torch.cat([attn_emb, native], dim=1))
                emb = F.normalize(native + self.res_gate * delta, dim=1)
            else:
                emb = self.fuse(torch.cat([attn_emb, native], dim=1))
                emb = F.normalize(emb, dim=1)
            return emb, attn_map
        return F.normalize(out["embedding"], dim=1), None

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        emb, _ = self.encode(x)
        return emb

    @torch.no_grad()
    def encode_verify(self, x: torch.Tensor, tta: bool = False) -> torch.Tensor:
        """Embedding used for VERIFICATION / deployment.

        Returns the single, unit-norm **512-d** fused embedding from one forward
        pass -- the same call shape and output dimensionality as a standard
        ArcFace encoder, so MDIE is a drop-in replacement on edge/security
        devices (enrol once, match by cosine). The bone-anchored attention and
        the backbone identity features are already merged inside the learned
        fusion head, so no score-level mixing or extra hyper-parameter is needed.

        ``tta`` (default OFF for the strict single-forward ArcFace-compatible
        path) optionally averages the embedding of the image and its horizontal
        mirror -- a standard, free accuracy boost when latency allows.
        """
        e, _ = self.encode(x)
        if tta:
            e2, _ = self.encode(torch.flip(x, dims=[3]))
            e = F.normalize(e + e2, dim=1)
        return e

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, identity_labels: torch.Tensor,
                modification_labels: torch.Tensor | None = None,
                bone_target: torch.Tensor | None = None):
        emb, attn_map = self.encode(x, bone_target)
        outputs = {"embedding": emb, "attn": attn_map}
        outputs["loss_identity"] = self.identity_head(emb, identity_labels)
        if self.use_region_prior:
            # Keeps RATA attention on each face's detected rigid bone landmarks
            # (penalises off-face leakage + per-image landmark coverage); see
            # RegionAwareTokenAttention.
            outputs["loss_attn"] = self.attn.last_attn_loss
        if self.use_amd and modification_labels is not None:
            mod_logits = self.mod_head(self.grl(emb))
            outputs["loss_mod"] = F.cross_entropy(mod_logits, modification_labels)
            outputs["mod_logits"] = mod_logits
        return outputs
