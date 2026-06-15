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


# Rigid skeletal landmarks that survive facial alterations (surgery, ageing,
# expression, disguise). These are bone-anchored points whose geometry stays
# constant: the frontal bone (glabella), temples, brow ridge, orbital rims,
# nasion, nasal bridge, cheekbones (zygomatic), jaw angles (mandible) and chin.
# The soft, easily-altered mouth/lips and nose tip are deliberately *excluded*
# — they change with expression and surgery. Positions are aligned to where a
# face-trained backbone actually shows rigid-structure energy on the 7x7 grid.
# Each entry is (x, y, weight) with x→right, y→down in [0, 1].
BONE_LANDMARKS = [
    (0.50, 0.27, 1.2),                        # glabella (frontal bone, mid-brow)
    (0.16, 0.30, 0.8), (0.84, 0.30, 0.8),    # temple / frontal process L/R
    (0.31, 0.34, 1.0), (0.69, 0.34, 1.0),    # brow ridge (supraorbital) L/R
    (0.50, 0.40, 0.9),                        # nasion (nose root between eyes)
    (0.30, 0.44, 1.0), (0.70, 0.44, 1.0),    # orbital rim (eye socket) L/R
    (0.50, 0.52, 1.0),                        # nasal bridge (bony, not the tip)
    (0.19, 0.60, 1.1), (0.81, 0.60, 1.1),    # cheekbone (zygomatic) L/R
    (0.17, 0.78, 1.0), (0.83, 0.78, 1.0),    # jaw angle (mandible) L/R
    (0.50, 0.79, 1.4),                        # chin (mental protuberance)
]

# Human-readable names, index-aligned with BONE_LANDMARKS.
BONE_LANDMARK_NAMES = [
    "glabella", "temple_L", "temple_R", "brow_L", "brow_R", "nasion",
    "orbital_L", "orbital_R", "nasal_bridge", "cheekbone_L", "cheekbone_R",
    "jaw_L", "jaw_R", "chin",
]


def _build_bone_prior(grid: int = 7, sigma2: float = 0.008) -> torch.Tensor:
    """Multi-peak map over the rigid skeletal landmarks; shape (grid, grid)."""
    g = grid
    yy, xx = torch.meshgrid(torch.linspace(0, 1, g), torch.linspace(0, 1, g), indexing="ij")
    m = torch.zeros(g, g)
    for x0, y0, w in BONE_LANDMARKS:
        m = m + w * torch.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2) / sigma2))
    return m / m.amax().clamp_min(1e-6)                          # (g, g) in [0,1]


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
        priors = _build_region_priors(grid)                       # (R, g, g)
        self.register_buffer("region_priors", priors)
        # Anatomical face envelope: union of all facial-region priors. ~1 on
        # cells covering eyes/brow/nose/cheek/jaw/forehead, ~0 on the image
        # border, hat and background. Used to anatomically gate the attention.
        face = priors.amax(dim=0)                                 # (g, g)
        face = face / face.amax().clamp_min(1e-6)
        self.register_buffer("face_envelope", face.reshape(1, grid * grid))
        # Rigid bone-landmark supervision is provided *per image* at training
        # time (see forward's ``bone_target`` argument), detected from each
        # individual face. No fixed canonical bone prior is baked in, so the
        # attention is genuinely data-driven and varies face to face; inference
        # needs no landmark detector (the head learns to find the bones).

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
        # Background suppression strength (learnable, kept positive via softplus).
        self.bg_gamma = nn.Parameter(torch.tensor(2.0))
        # Set in forward(): off-face penalty + (training) per-image bone-landmark
        # matching, so attention is supervised onto each face's detected bones.
        self.last_attn_loss = torch.zeros(())

    def forward(self, feat: torch.Tensor, bone_target: torch.Tensor | None = None):
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

        logits = self.attn_head(z).squeeze(-1)                    # (B, HW)

        # --- anatomical gating -------------------------------------------------
        # Suppress off-face cells (hat/background). Attention is otherwise fully
        # data-driven: no fixed bone bias and no feature-energy bias (the energy
        # bias used to drag attention onto the high-energy central T-zone, making
        # every face look alike). The per-face bone focus is learned purely via
        # the matching loss below, so attention reads each individual's geometry.
        with torch.no_grad():
            face = self.face_envelope                             # (1, HW) in [0,1]
            background = (1.0 - face).clamp(0.0, 1.0)             # ~1 off-face
        gamma = F.softplus(self.bg_gamma)
        logits = logits - gamma * background

        attn = F.softmax(logits, dim=1)                           # (B, HW)
        # Supervision = off-face leakage + (training) per-image bone *matching*.
        # We match the attention to each face's own detected rigid-bone target
        # in BOTH directions:
        #   forward  CE(target, attn)  pulls attention onto the bone landmarks;
        #   reverse  CE(attn, target)  penalises attention mass that lands away
        #                              from the bones (kills the lazy central
        #                              blob), forcing a faithful per-face map.
        # Samples with no detected face (all-zero target) are ignored.
        bg_penalty = (attn * background).sum(dim=1).mean()
        coverage = attn.new_zeros(())
        if bone_target is not None:
            tgt = bone_target.reshape(B, -1).to(attn.dtype)       # (B, HW)
            valid = (tgt.sum(dim=1) > 0.5).to(attn.dtype)         # (B,)
            # floor the target so the reverse term is finite everywhere on-face
            tfloor = tgt + 1e-4
            tfloor = tfloor / tfloor.sum(dim=1, keepdim=True).clamp_min(1e-9)
            fwd = -(tgt * (attn + 1e-9).log()).sum(dim=1)        # (B,)
            rev = -(attn * (tfloor + 1e-9).log()).sum(dim=1)     # (B,)
            denom = valid.sum().clamp_min(1.0)
            coverage = ((fwd + rev) * valid).sum() / denom
        self.last_attn_loss = bg_penalty + coverage

        weighted = z * attn.unsqueeze(-1) * (H * W)
        weighted = weighted.transpose(1, 2).reshape(B, d, H, W)
        delta = self.proj_out(weighted)
        # residual gate: out = feat + tanh(gate) * delta  (starts as identity)
        out = feat + torch.tanh(self.gate) * delta
        return out, attn.reshape(B, H, W)
