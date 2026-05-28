"""Embedding extraction + pair scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Tuple

import cv2
import numpy as np
import torch

from ..data.modifications import apply_modification
from ..data.pairs import PairSet


def _load(path: Path, size: int = 112) -> np.ndarray:
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
    arr = np.stack(imgs).astype(np.float32)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)
    t = (t - 127.5) / 128.0
    return t.to(device, non_blocking=True)


@torch.no_grad()
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
        emb_chunks = []
        for i in range(0, len(side_paths), batch_size):
            batch_paths = side_paths[i:i+batch_size]
            imgs = [_load(p, image_size) for p in batch_paths]
            if do_mod and modification and modification != "clean":
                imgs = [apply_modification(im, modification, seed=i + j)
                         for j, im in enumerate(imgs)]
            x = _to_batch(imgs, device)
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
