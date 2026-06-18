"""Attention-bone IoU: a paper-grade interpretability metric for MDIE RATA.

The thesis claim is that the learned attention lands on each face's OWN rigid
bone structure. Cosine(attention, target) already shows agreement, but a
reviewer wants a thresholded, region-overlap metric with proper CONTROLS. This
script reports, on the held-out (unseen-identity) split:

  matched      IoU( attention(face i) , bone-target(face i) )      <- the claim
  mismatched   IoU( attention(face i) , bone-target(face j!=i) )   <- per-face
                                                                       specificity
  random-null  IoU( random map        , bone-target(face i) )      <- chance floor

If matched >> mismatched >> random, the attention is genuinely anchored on each
individual's bones (not a shared template, not chance). We also report per
anatomical-group coverage (brow ridge / orbital rim / nose bridge / cheekbone /
jaw-chin) and a Mann-Whitney significance test on matched vs mismatched.

Run from showcase/02_laptop_rtx3050 with PYTHONPATH=cwd:
    python <this>.py [n_faces]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research_v2.src.config import DATA_DIR
from research_v2.src.data import build_face_dataset, prepare_lfw
from research_v2.src.data.landmarks import (
    GRID, rigid_bone_points, target_from_landmarks)
from research_v2.src.data.landmarks import RIGID_GROUPS
from research_v2.src.data.landmarks import configure_onnxruntime_threads
from research_v2.src.novel.mdie import MDIE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRACTIONS = [0.10, 0.15, 0.20, 0.25]
HEADLINE_F = 0.15


def load_mdie():
    ck = Path("research_v2/checkpoints/mdie_full_best_auc.pt")
    m = MDIE(n_identity_classes=173, pretrained_backbone=True).to(DEVICE).eval()
    sd = torch.load(ck, map_location=DEVICE)
    st = sd.get("model", sd.get("model_state", sd))
    st = {k.replace("_orig_mod.", ""): v for k, v in st.items()}
    st = {k: v for k, v in st.items()
          if not k.startswith(("mod_head.", "identity_head."))}
    miss, unexp = m.load_state_dict(st, strict=False)
    print(f"  loaded {ck.name} (missing={len(miss)}, unexpected={len(unexp)})")
    return m


@torch.no_grad()
def attention_of(model, rgb):
    x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = (x - 0.5) / 0.5
    _, a = model.encode(x.to(DEVICE))
    return a[0].cpu().numpy()


def topk_mask(m: np.ndarray, k: int) -> np.ndarray:
    """Binary mask of the k highest cells of m."""
    flat = m.flatten()
    idx = np.argpartition(flat, -k)[-k:]
    out = np.zeros(flat.size, dtype=bool)
    out[idx] = True
    return out.reshape(m.shape)


def iou(a: np.ndarray, b: np.ndarray, frac: float) -> float:
    k = max(1, int(round(frac * a.size)))
    ma, mb = topk_mask(a, k), topk_mask(b, k)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter) / float(union) if union else 0.0


def group_cells(lm68, w, h):
    """Return {group_name: set of (r,c) cells} for this face's bone groups."""
    pts = rigid_bone_points(lm68)
    cells = {}
    for gname, (names, _color) in RIGID_GROUPS.items():
        s = set()
        for nm in names:
            x, y = pts[nm]
            c = min(int((x / max(w, 1)) * GRID), GRID - 1)
            r = min(int((y / max(h, 1)) * GRID), GRID - 1)
            s.add((r, c))
        cells[gname] = s
    return cells


def mannwhitney_u(a, b):
    """Two-sided Mann-Whitney U p-value via normal approx (no scipy dep)."""
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sig = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sig == 0:
        return 1.0
    z = (u1 - mu) / sig
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return float(min(max(p, 0.0), 1.0))


def main():
    n_faces = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    lfw_dir = prepare_lfw(DATA_DIR, min_faces_per_person=8)
    paths, labels, names = build_face_dataset(lfw_dir, min_imgs=4)
    n_classes = len(names)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n_classes)
    test_ids = set(perm[int(0.8 * n_classes):].tolist())   # held-out identities
    # one clean image per held-out identity
    seen, picked = set(), []
    for p, l in zip(paths, labels):
        if l in test_ids and l not in seen:
            seen.add(l); picked.append(p)
        if len(picked) >= n_faces:
            break
    print(f"  held-out identities sampled: {len(picked)}")

    model = load_mdie()

    from insightface.app import FaceAnalysis
    configure_onnxruntime_threads()
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "landmark_3d_68"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(160, 160))

    attns, targets, groupcells = [], [], []
    for p in picked:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (112, 112))
        a = attention_of(model, rgb)
        faces = app.get(bgr)
        if not faces:
            continue
        f = max(faces, key=lambda z: (z.bbox[2] - z.bbox[0]) *
                (z.bbox[3] - z.bbox[1]))
        lm = getattr(f, "landmark_3d_68", None)
        if lm is None:
            continue
        h, w = bgr.shape[:2]
        t = target_from_landmarks(lm, w, h, grid=GRID)
        if t.sum() <= 0:
            continue
        attns.append(a); targets.append(t)
        groupcells.append(group_cells(lm, w, h))
    N = len(attns)
    print(f"  faces with attention + detected bones: {N}\n")

    # fixed derangement for the mismatched control
    shift = max(1, N // 2)
    rng2 = np.random.RandomState(1)

    print(f"  {'frac':>5s} | {'matched':>8s} {'mismatch':>8s} {'random':>8s} "
          f"| {'gap':>6s}")
    print("  " + "-" * 48)
    head_matched, head_mismatch = None, None
    for frac in FRACTIONS:
        m_iou, x_iou, r_iou = [], [], []
        for i in range(N):
            j = (i + shift) % N
            m_iou.append(iou(attns[i], targets[i], frac))
            x_iou.append(iou(attns[i], targets[j], frac))
            rnd = rng2.rand(*attns[i].shape)
            r_iou.append(iou(rnd, targets[i], frac))
        mm, xx, rr = np.mean(m_iou), np.mean(x_iou), np.mean(r_iou)
        print(f"  {frac:5.2f} | {mm:8.3f} {xx:8.3f} {rr:8.3f} | {mm - xx:6.3f}")
        if abs(frac - HEADLINE_F) < 1e-9:
            head_matched, head_mismatch = m_iou, x_iou

    p = mannwhitney_u(head_matched, head_mismatch)
    print(f"\n  Mann-Whitney U (matched vs mismatched, frac={HEADLINE_F}): "
          f"p = {p:.2e}")
    print(f"  matched   IoU = {np.mean(head_matched):.3f} "
          f"+/- {np.std(head_matched):.3f}")
    print(f"  mismatch  IoU = {np.mean(head_mismatch):.3f} "
          f"+/- {np.std(head_mismatch):.3f}")

    # per anatomical-group coverage: fraction of attention mass at each group's
    # cells vs the uniform share of those cells (>1 = group is attended)
    print(f"\n  PER-ANATOMICAL-GROUP COVERAGE (attention mass / uniform share):")
    uniform = 1.0 / (GRID * GRID)
    group_ratio = {}
    for gname in RIGID_GROUPS:
        ratios = []
        for i in range(N):
            cells = groupcells[i][gname]
            mass = sum(attns[i][r, c] for (r, c) in cells)
            ratios.append(mass / (uniform * len(cells)))
        group_ratio[gname] = float(np.mean(ratios))
        bar = "#" * int(min(group_ratio[gname], 6) * 4)
        print(f"    {gname:12s} {group_ratio[gname]:5.2f}x  {bar}")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    labels_b = ["matched\n(own bones)", "mismatched\n(other face)",
                "random\nattention"]
    vals = [np.mean(head_matched), np.mean(head_mismatch),
            np.mean([iou(np.random.RandomState(k).rand(GRID, GRID),
                         targets[k], HEADLINE_F) for k in range(N)])]
    cols = ["#2a9d8f", "#e9c46a", "#bbbbbb"]
    ax1.bar(labels_b, vals, color=cols)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10,
                 fontweight="bold")
    ax1.set_ylabel(f"attention-bone IoU (top {int(HEADLINE_F*100)}% cells)")
    ax1.set_title(f"Attention lands on each face's OWN bones\n"
                  f"(held-out identities, n={N}, p={p:.1e})", fontsize=10)
    ax1.set_ylim(0, max(vals) * 1.25)

    gnames = list(group_ratio.keys())
    gvals = [group_ratio[g] for g in gnames]
    gcols = [RIGID_GROUPS[g][1] for g in gnames]
    ax2.barh(gnames, gvals, color=gcols)
    ax2.axvline(1.0, color="k", ls="--", lw=1, label="uniform (1.0x)")
    for i, v in enumerate(gvals):
        ax2.text(v + 0.05, i, f"{v:.2f}x", va="center", fontsize=9)
    ax2.set_xlabel("attention mass / uniform share")
    ax2.set_title("Every rigid-bone group is attended\n(> 1.0x = above uniform)",
                  fontsize=10)
    ax2.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out = Path("research_v2/figures/attention_bone_iou.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
