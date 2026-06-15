"""
Grad-CAM interpretability comparison: ArcFace vs MDIE across modifications.

Outputs:
    figures/gradcam_grid.pdf — 6 identities x 8 columns
        (clean/mask/glasses/surgery for ArcFace and MDIE)
    figures/cam_iou.pdf      — bar chart of CAM-peak IoU with the eye-region
                                landmark box across 217 ids x 9 modifications.

CPU-runnable for scaffolding, but the full grid + IoU numbers want a GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..config import CKPT_DIR, DATA_DIR, FIGURES_DIR, RESULTS_DIR, get_device, seed_all
from ..data.lfw import build_face_dataset, prepare_lfw
from ..data.modifications import MODIFICATION_TYPES, apply_modification


# ----------------------------------------------------------------------------
# Generic Grad-CAM on the last conv block.
# ----------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_module.register_forward_hook(self._fwd)
        target_module.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.activations = out.detach()

    def _bwd(self, _m, _gi, go):
        self.gradients = go[0].detach()

    def __call__(self, encode_fn: Callable[[torch.Tensor], torch.Tensor],
                  x: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        emb = encode_fn(x)                              # (1, D), L2-normed
        # Use the sum of the embedding as a scalar target — gives gradients
        # proportional to which spatial locations push the embedding outward.
        emb.sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ----------------------------------------------------------------------------
# Identity-stable region prior: eye box at 112x112.
# Standard ArcFace alignment puts the eyes at roughly y=38, x in [30, 82].
# ----------------------------------------------------------------------------

EYE_BOX_112 = (28, 32, 84, 60)   # x0, y0, x1, y1


def _eye_mask(size: int = 112) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.float32)
    x0, y0, x1, y1 = EYE_BOX_112
    m[y0:y1, x0:x1] = 1.0
    return m


def cam_iou(cam: np.ndarray, mask: np.ndarray, top_frac: float = 0.20) -> float:
    """IoU between the top-fraction CAM pixels and the eye-region mask."""
    k = int(round(top_frac * cam.size))
    if k <= 0:
        return 0.0
    flat = cam.ravel()
    thr = np.partition(flat, -k)[-k]
    cam_top = (cam >= thr).astype(np.float32)
    inter = float((cam_top * mask).sum())
    union = float(((cam_top + mask) > 0).sum())
    return inter / max(union, 1.0)


# ----------------------------------------------------------------------------
# Model loaders (mirror eval/run_real_benchmarks.py).
# ----------------------------------------------------------------------------

def load_arcface(device):
    from ..baselines import build_baseline
    ck = CKPT_DIR / "baseline_arcface.pt"
    if not ck.exists():
        legacy = CKPT_DIR / "arcface.pt"
        if legacy.exists():
            ck = legacy
        else:
            return None, None, None
    sd = torch.load(ck, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    n_cls = int(sd["head.W"].shape[0]) if "head.W" in sd else 1
    model = build_baseline("arcface", n_classes=n_cls).to(device).eval()
    model.load_state_dict(sd, strict=False)
    target = model.backbone.body[-1]      # last IR block

    def enc(x):
        return model.backbone(x)["embedding"]
    return model, enc, target


def load_mdie(device):
    from ..novel import MDIE
    candidates = [CKPT_DIR / "mdie_full.pt", CKPT_DIR / "mdie-full.pt",
                  CKPT_DIR / "mdie.pt"]
    ck = next((c for c in candidates if c.exists()), None)
    if ck is None:
        return None, None, None
    state = torch.load(ck, map_location=device, weights_only=False)
    cfg = state.get("config", {}) if isinstance(state, dict) else {}
    sd = state.get("model", state) if isinstance(state, dict) and "model" in state else state
    has_rata = any(k.startswith(("attn.", "post_pool.")) for k in sd.keys())
    has_amd = any(k.startswith(("mod_head.", "grl.")) for k in sd.keys())
    n_id = cfg.get("n_identity_classes")
    if n_id is None:
        n_id = int(sd["identity_head.W"].shape[0]) if "identity_head.W" in sd else 1000
    model = MDIE(
        n_identity_classes=n_id,
        n_modification_classes=cfg.get("n_modification_classes", 10),
        embedding_dim=cfg.get("embedding_dim", 512),
        use_region_prior=cfg.get("use_region_prior", has_rata),
        use_amd=cfg.get("use_amd", has_amd),
        amd_lambda=cfg.get("amd_lambda", 0.10),
    ).to(device).eval()
    model.load_state_dict(sd, strict=False)
    target = model.backbone.body[-1]

    def enc(x):
        emb, _ = model.encode(x)
        return emb
    return model, enc, target


# ----------------------------------------------------------------------------

def _load_img(p: Path, size: int = 112) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size))
    return img


def _to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    return ((t - 127.5) / 128.0).to(device)


def _overlay(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (0.55 * img + 0.45 * heat).clip(0, 255).astype(np.uint8)


def make_grid(out_pdf: Path, paths: List[Path], mods: List[str],
              arcface_enc, arcface_cam, mdie_enc, mdie_cam, device):
    import matplotlib.pyplot as plt
    n_id = len(paths); n_mod = len(mods)
    fig, axes = plt.subplots(n_id, 2 * n_mod, figsize=(2 * n_mod * 1.6, n_id * 1.6),
                              dpi=200)
    if n_id == 1:
        axes = axes[None, :]
    for i, p in enumerate(paths):
        base = _load_img(p)
        for j, mod in enumerate(mods):
            mim = base if mod == "clean" else apply_modification(base, mod, seed=i)
            x = _to_tensor(mim, device).requires_grad_(False)
            x_g = x.clone().requires_grad_(True)
            cam_a = arcface_cam(arcface_enc, x_g)
            x_g2 = x.clone().requires_grad_(True)
            cam_m = mdie_cam(mdie_enc, x_g2)

            axes[i, j].imshow(_overlay(mim, cam_a))
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(f"ArcFace\n{mod}", fontsize=7)
            axes[i, j + n_mod].imshow(_overlay(mim, cam_m))
            axes[i, j + n_mod].set_xticks([]); axes[i, j + n_mod].set_yticks([])
            if i == 0:
                axes[i, j + n_mod].set_title(f"MDIE\n{mod}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def make_iou_bar(out_pdf: Path, iou_arc: dict, iou_mdie: dict):
    import matplotlib.pyplot as plt
    mods = list(iou_arc.keys())
    a = [np.mean(iou_arc[m]) for m in mods]
    b = [np.mean(iou_mdie[m]) for m in mods]
    x = np.arange(len(mods)); w = 0.4
    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    ax.bar(x - w/2, a, w, label="ArcFace", color="#1976D2")
    ax.bar(x + w/2, b, w, label="MDIE",    color="#D32F2F")
    ax.set_xticks(x); ax.set_xticklabels(mods, rotation=30, ha="right")
    ax.set_ylabel("CAM ∩ eye-region  /  union  (mean IoU)")
    ax.set_ylim(0, 1)
    ax.set_title("Where do the models look? (higher = more identity-stable)")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid-ids", type=int, default=6,
                    help="rows in the CAM grid")
    ap.add_argument("--n-iou-ids", type=int, default=217,
                    help="identities used for the quantitative IoU")
    ap.add_argument("--mods", nargs="+",
                    default=["clean", "disguise_mask", "disguise_glasses",
                             "disguise_cap"])
    ap.add_argument("--iou-mods", nargs="+", default=list(MODIFICATION_TYPES))
    args = ap.parse_args()

    seed_all()
    device = get_device()
    lfw_root = prepare_lfw(DATA_DIR)
    paths, labels, _ = build_face_dataset(lfw_root)

    # one image per identity, deterministic
    seen, by_id = set(), []
    for p, y in zip(paths, labels):
        if y not in seen:
            seen.add(y); by_id.append(p)
    grid_paths = by_id[: args.n_grid_ids]
    iou_paths  = by_id[: args.n_iou_ids]

    _, arc_enc, arc_target = load_arcface(device)
    _, mdie_enc, mdie_target = load_mdie(device)
    if arc_enc is None or mdie_enc is None:
        print("[skip] need both baseline_arcface.pt and mdie checkpoint")
        return

    # Wire up GradCAM.
    arc_cam = GradCAM(arc_enc.__self__ if hasattr(arc_enc, "__self__") else None,
                       arc_target) if False else GradCAM(  # noqa: SIM108
            torch.nn.Sequential(), arc_target)
    mdie_cam = GradCAM(torch.nn.Sequential(), mdie_target)

    # 1. Qualitative grid.
    grid_pdf = FIGURES_DIR / "gradcam_grid.pdf"
    make_grid(grid_pdf, grid_paths, args.mods,
               arc_enc, arc_cam, mdie_enc, mdie_cam, device)
    print(f"[done] {grid_pdf}")

    # 2. Quantitative IoU sweep.
    mask = _eye_mask(112)
    iou_arc = {m: [] for m in args.iou_mods}
    iou_mdie = {m: [] for m in args.iou_mods}
    for k, p in enumerate(iou_paths):
        base = _load_img(p)
        for m in args.iou_mods:
            mim = base if m == "clean" else apply_modification(base, m, seed=k)
            x = _to_tensor(mim, device).requires_grad_(True)
            iou_arc[m].append(cam_iou(arc_cam(arc_enc, x), mask))
            x2 = _to_tensor(mim, device).requires_grad_(True)
            iou_mdie[m].append(cam_iou(mdie_cam(mdie_enc, x2), mask))
        if (k + 1) % 25 == 0:
            print(f"  [iou] {k+1}/{len(iou_paths)}")

    bar_pdf = FIGURES_DIR / "cam_iou.pdf"
    make_iou_bar(bar_pdf, iou_arc, iou_mdie)
    print(f"[done] {bar_pdf}")

    summary = {
        "iou_arcface": {m: float(np.mean(v)) for m, v in iou_arc.items()},
        "iou_mdie":    {m: float(np.mean(v)) for m, v in iou_mdie.items()},
        "n_identities": len(iou_paths),
    }
    out_json = RESULTS_DIR / "cam_iou.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {out_json}")


if __name__ == "__main__":
    main()
