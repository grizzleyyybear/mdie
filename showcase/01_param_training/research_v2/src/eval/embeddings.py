"""Embedding extraction, pair scoring, and quick verification checks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.metrics import auc, roc_curve

from ..data import PairSet


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for image loading. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return cv2


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for embedding extraction. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return torch


def _load(path: Path, size: int = 112) -> np.ndarray:
    cv2 = _require_cv2()
    # cv2.imread cannot handle non-ASCII paths on Windows; use np.fromfile.
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    except OSError:
        img = None
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size))
    return img


def _to_batch(imgs: list[np.ndarray], device: torch.device) -> torch.Tensor:
    torch = _require_torch()
    arr = np.stack(imgs).astype(np.float32)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)
    t = (t - 127.5) / 128.0
    return t.to(device, non_blocking=True)


def extract_embeddings_for_pairs(
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    pair_set: PairSet,
    device: torch.device,
    batch_size: int = 64,
    image_size: int = 112,
    modification: str | None = None,
    apply_to: str = "right",   # "left", "right", "both", "none"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (left_embeddings, right_embeddings, labels) with one row per pair.
    `encode_fn` should accept a batched tensor and return L2-normalised embeddings.
    """
    lefts, rights, labels = [], [], []
    for p1, p2, lbl in pair_set.pairs:
        lefts.append(p1); rights.append(p2); labels.append(lbl)
    labels = np.array(labels, dtype=np.int64)

    def _process(side_paths, do_mod):
        torch = _require_torch()
        emb_chunks = []
        for i in range(0, len(side_paths), batch_size):
            batch_paths = side_paths[i:i+batch_size]
            imgs = [_load(p, image_size) for p in batch_paths]
            if do_mod and modification and modification != "clean":
                from ..data.modifications import apply_modification
                imgs = [apply_modification(im, modification, seed=i + j)
                         for j, im in enumerate(imgs)]
            x = _to_batch(imgs, device)
            with torch.no_grad():
                emb = encode_fn(x).detach().cpu().numpy()
            emb_chunks.append(emb)
        return np.concatenate(emb_chunks, axis=0)

    do_left = apply_to in ("left", "both")
    do_right = apply_to in ("right", "both")
    left_emb = _process(lefts, do_left)
    right_emb = _process(rights, do_right)
    return left_emb, right_emb, labels


def score_pairs(left_emb: np.ndarray, right_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity per row pair."""
    a = left_emb / (np.linalg.norm(left_emb, axis=1, keepdims=True) + 1e-12)
    b = right_emb / (np.linalg.norm(right_emb, axis=1, keepdims=True) + 1e-12)
    return (a * b).sum(axis=1)


def compute_roc(scores: np.ndarray, labels: np.ndarray):
    """Return false-positive rate, true-positive rate, thresholds, and AUC."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return fpr, tpr, thresholds, auc(fpr, tpr)


def compute_eer(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray):
    """Equal Error Rate at the closest FPR/FNR intersection."""
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def compute_tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target_far: float) -> float:
    """True-Accept Rate at a fixed False-Accept Rate."""
    idx = int(np.searchsorted(fpr, target_far, side="left"))
    idx = min(max(idx, 0), len(tpr) - 1)
    return float(tpr[idx])


def summarize_run(
    scores: np.ndarray,
    labels: np.ndarray,
    far_targets=(1e-1, 1e-2, 1e-3),
) -> Dict:
    """Summarize verification scores into paper/report metrics."""
    fpr, tpr, thresholds, roc_auc = compute_roc(scores, labels)
    eer, eer_threshold = compute_eer(fpr, tpr, thresholds)
    return {
        "auc": float(roc_auc),
        "eer": eer,
        "eer_threshold": eer_threshold,
        **{f"tar_at_far={t:g}": compute_tar_at_far(fpr, tpr, t) for t in far_targets},
        "n_pairs": int(len(labels)),
        "n_pos": int(labels.sum()),
        "n_neg": int(len(labels) - labels.sum()),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def quick_verification_auc(
    encode: Callable[[torch.Tensor], torch.Tensor],
    pairs: List[Tuple[Path, Path, int]],
    device: torch.device,
    batch_size: int = 64,
    image_size: int = 112,
) -> float:
    """
    Return a cheap ROC AUC on a fixed pair pool during training.

    This avoids importing sklearn inside the training loop and gives checkpoint
    selection a real verification signal instead of relying only on train loss.
    """
    if not pairs:
        return float("nan")
    torch = _require_torch()

    left = [_load(p[0], image_size) for p in pairs]
    right = [_load(p[1], image_size) for p in pairs]
    labels = np.asarray([int(p[2]) for p in pairs], dtype=np.int32)

    def _embed(images: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            x = _to_batch(images[i:i + batch_size], device)
            with torch.no_grad():
                emb = encode(x)
                emb = torch.nn.functional.normalize(emb, dim=1)
            out.append(emb.detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    left_emb = _embed(left)
    right_emb = _embed(right)
    scores = (left_emb * right_emb).sum(axis=1)

    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    auc = (ranks[labels == 1].sum() - pos.size * (pos.size + 1) / 2.0) / (
        pos.size * neg.size
    )
    return float(auc)
