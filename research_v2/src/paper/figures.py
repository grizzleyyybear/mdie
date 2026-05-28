"""Publication-quality figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_roc_curves(roc_data: Dict[str, Dict], out: Path, title: str = "ROC — verification"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, d in roc_data.items():
        ax.plot(d["fpr"], d["tpr"], label=f"{name} (AUC={d['auc']:.3f})", linewidth=1.6)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1.0); ax.set_ylim(0, 1.02)
    ax.set_title(title); ax.legend(loc="lower right", fontsize=9)
    _save(fig, out)


def plot_score_distributions(genuine: np.ndarray, impostor: np.ndarray,
                              out: Path, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(genuine, bins=40, alpha=0.55, label=f"Genuine (n={len(genuine)})", color="#2C7A7B")
    ax.hist(impostor, bins=40, alpha=0.55, label=f"Impostor (n={len(impostor)})", color="#9B2C2C")
    ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Count")
    ax.set_title(f"{model_name} — genuine vs impostor distributions")
    ax.legend()
    _save(fig, out)


def plot_per_modification_bars(per_mod_results: Dict[str, Dict[str, float]],
                                metric: str, out: Path,
                                title: str | None = None):
    """
    per_mod_results = {model_name: {modification: metric_value}}
    """
    models = list(per_mod_results.keys())
    mods = sorted({m for d in per_mod_results.values() for m in d.keys()})
    x = np.arange(len(mods))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(7, 0.6*len(mods)), 4.5))
    for i, m in enumerate(models):
        vals = [per_mod_results[m].get(mod, np.nan) for mod in mods]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(mods, rotation=30, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} per modification")
    ax.legend(fontsize=9, ncol=2)
    _save(fig, out)


def plot_occlusion_heatmap(drops: np.ndarray, out: Path, model_name: str):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(drops, cmap="Reds")
    ax.set_xticks(range(drops.shape[1])); ax.set_yticks(range(drops.shape[0]))
    ax.set_title(f"{model_name} — occlusion sensitivity (Δsim)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for r in range(drops.shape[0]):
        for c in range(drops.shape[1]):
            ax.text(c, r, f"{drops[r,c]:.02f}", ha="center", va="center",
                     color="black" if drops[r,c] < drops.max()*0.6 else "white",
                     fontsize=7)
    _save(fig, out)


def plot_attention_overlay(image: np.ndarray, attn: np.ndarray, out: Path,
                            title: str = "RATA attention"):
    """image: HxWx3 uint8; attn: small grid (g,g) array."""
    import cv2
    h, w = image.shape[:2]
    a = cv2.resize(attn.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    heat = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * image + 0.45 * heat).clip(0, 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))
    ax[0].imshow(image); ax[0].set_title("Input"); ax[0].axis("off")
    ax[1].imshow(a, cmap="jet"); ax[1].set_title("Attention"); ax[1].axis("off")
    ax[2].imshow(overlay); ax[2].set_title("Overlay"); ax[2].axis("off")
    fig.suptitle(title)
    _save(fig, out)


def plot_training_curves(histories: Dict[str, Dict[str, List[float]]], out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, h in histories.items():
        ax.plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], label=name, linewidth=1.6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss")
    ax.set_title("Training curves")
    ax.legend(fontsize=9)
    _save(fig, out)
