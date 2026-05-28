"""ROC / EER / TAR@FAR / score-distribution helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_curve, auc


def compute_roc(scores: np.ndarray, labels: np.ndarray):
    """Returns (fpr, tpr, thresholds, auc)."""
    fpr, tpr, th = roc_curve(labels, scores)
    return fpr, tpr, th, auc(fpr, tpr)


def compute_eer(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray):
    """Equal Error Rate (intersection of FPR and FNR)."""
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def compute_tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target_far: float) -> float:
    """True-Accept Rate (TAR) at a fixed False-Accept Rate (FAR)."""
    idx = int(np.searchsorted(fpr, target_far, side="left"))
    idx = min(max(idx, 0), len(tpr) - 1)
    return float(tpr[idx])


def summarize_run(scores: np.ndarray, labels: np.ndarray,
                  far_targets=(1e-1, 1e-2, 1e-3)) -> Dict:
    fpr, tpr, th, roc_auc = compute_roc(scores, labels)
    eer, eer_th = compute_eer(fpr, tpr, th)
    return {
        "auc": float(roc_auc),
        "eer": eer,
        "eer_threshold": eer_th,
        **{f"tar_at_far={t:g}": compute_tar_at_far(fpr, tpr, t) for t in far_targets},
        "n_pairs": int(len(labels)),
        "n_pos": int(labels.sum()),
        "n_neg": int(len(labels) - labels.sum()),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }
