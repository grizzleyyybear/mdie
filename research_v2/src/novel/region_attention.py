"""
Region-Aware Token Attention (RATA) — first novelty of MDIE.

A grid of spatial tokens (7×7 for IR-50 at 112×112 input) is augmented with
a *facial-region prior* (one of {orbital, periocular, nasal, oral, jaw,
forehead, cheek}) injected as an additional learnable embedding. A 2-layer
transformer encoder then learns *per-input* down-weighting of unreliable
regions (e.g. nasal under rhinoplasty, oral under masking), producing a
re-weighted feature map for global pooling.

Why this is novel:
- Existing region-aware face methods either (a) hard-mask regions (Wang 2020),
  (b) use multi-stream CNNs (Song 2019), or (c) use plain self-attention with
  no anatomical prior. Combining anatomical priors with learned per-input
  attention over a transformer is, to our knowledge, unpublished.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Facial-region soft mask on the spatial grid (computed once; values in [0,1]
# per region). Region indices match REGION_NAMES.
REGION_NAMES = ["orbital", "periocular", "nasal", "oral", "jaw", "forehead", "cheek"]


def _build_region_priors(grid: int = 7) -> torch.Tensor:
    """Return shape (n_regions, grid, grid) with smooth membership masks."""
    g = grid
    yy, xx = torch.meshgrid(torch.linspace(0, 1, g), torch.linspace(0, 1, g), indexing="ij")
    masks = []
    # orbital — eyes, two blobs around y≈0.40
    m = torch.exp(-((xx - 0.32) ** 2 + (yy - 0.40) ** 2) / 0.012) + \
        torch.exp(-((xx - 0.68) ** 2 + (yy - 0.40) ** 2) / 0.012)
    masks.append(m)
    # periocular — wider band around eyes
    m = torch.exp(-((yy - 0.40) ** 2) / 0.025)
    m = m * (torch.exp(-((xx - 0.5) ** 2) / 0.20))
    masks.append(m)
    # nasal
    masks.append(torch.exp(-((xx - 0.5) ** 2 / 0.012 + (yy - 0.55) ** 2 / 0.020)))
    # oral
    masks.append(torch.exp(-((xx - 0.5) ** 2 / 0.020 + (yy - 0.78) ** 2 / 0.015)))
    # jaw
    m = torch.exp(-((yy - 0.92) ** 2) / 0.008)
    m = m * (torch.exp(-((xx - 0.5) ** 2) / 0.18))
    masks.append(m)
    # forehead
    m = torch.exp(-((yy - 0.18) ** 2) / 0.012)
    m = m * (torch.exp(-((xx - 0.5) ** 2) / 0.15))
    masks.append(m)
    # cheek
    m = torch.exp(-((xx - 0.20) ** 2 / 0.025 + (yy - 0.60) ** 2 / 0.020)) + \
        torch.exp(-((xx - 0.80) ** 2 / 0.025 + (yy - 0.60) ** 2 / 0.020))
    masks.append(m)
    out = torch.stack(masks)
    out = out / out.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return out                                                   # (R, g, g)


class RegionAwareTokenAttention(nn.Module):
    """
    Inputs:  feature map (B, C, H, W) — typically from the IR-50 body output.
    Output:  re-weighted feature map (B, C, H, W) and (B, H, W) attention.

    Mechanism:
        token_emb_i = proj(spatial_feat_i) + Σ_r region_prior[r,i]·region_emb[r] + pos[i]
        token_emb   = TransformerEncoder(token_emb)
        weights     = softmax(MLP(token_emb))
        output      = (weights · token_emb)  reshaped to spatial map
    """

    def __init__(self, channels: int = 512, dim: int = 192, heads: int = 4,
                 layers: int = 2, grid: int = 14, n_regions: int = 7,
                 dropout: float = 0.3):
        super().__init__()
        self.grid = grid
        self.n_regions = n_regions
        self.proj_in = nn.Conv2d(channels, dim, 1)
        self.region_emb = nn.Parameter(torch.randn(n_regions, dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, grid * grid, dim) * 0.02)
        self.register_buffer("region_priors", _build_region_priors(grid))   # (R, g, g)

        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.tok_drop = nn.Dropout(dropout)
        self.attn_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim // 2, 1))
        self.proj_out = nn.Conv2d(dim, channels, 1)
        # residual gate so RATA starts as identity and learns a small correction
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, feat: torch.Tensor):
        B, C, H, W = feat.shape
        if H != self.grid or W != self.grid:
            feat = F.adaptive_avg_pool2d(feat, (self.grid, self.grid))
            H = W = self.grid

        x = self.proj_in(feat)                                    # (B, dim, H, W)
        d = x.shape[1]
        tokens = x.flatten(2).transpose(1, 2)                     # (B, HW, dim)
        # region prior contribution
        rp = self.region_priors.view(self.n_regions, H * W).t()   # (HW, R)
        region_term = rp @ self.region_emb                        # (HW, dim)
        tokens = self.tok_drop(tokens + region_term.unsqueeze(0) + self.pos)
        z = self.encoder(tokens)                                  # (B, HW, dim)
        attn = F.softmax(self.attn_head(z).squeeze(-1), dim=1)    # (B, HW)
        weighted = z * attn.unsqueeze(-1) * (H * W)
        weighted = weighted.transpose(1, 2).reshape(B, d, H, W)
        delta = self.proj_out(weighted)
        # residual gate: out = feat + tanh(gate) * delta  (starts as identity)
        out = feat + torch.tanh(self.gate) * delta
        return out, attn.reshape(B, H, W)
