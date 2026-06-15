"""
Occlusion-sensitivity analysis: which facial regions, when occluded, cause the
biggest drop in genuine-pair similarity? Produces a 7×7 heatmap per model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch

from .embeddings import _load, _to_batch


@torch.no_grad()
def region_sensitivity_map(
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    paths: List[Path],
    pair_indices: List[Tuple[int, int]],
    device: torch.device,
    grid: int = 7,
    image_size: int = 112,
    patch_color: Tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """
    For each grid cell (r, c), occlude the corresponding patch in the *left*
    image of every pair, then measure the drop in cosine similarity.
    Returns (grid, grid) array of mean similarity drops.
    """
    cell = image_size // grid
    drops = np.zeros((grid, grid), dtype=np.float32)
    counts = np.zeros((grid, grid), dtype=np.float32)

    # cache right embeddings + clean similarity baseline
    right_imgs = [_load(paths[j], image_size) for _, j in pair_indices]
    right_emb = encode_fn(_to_batch(right_imgs, device)).cpu().numpy()
    left_imgs_clean = [_load(paths[i], image_size) for i, _ in pair_indices]
    left_emb_clean = encode_fn(_to_batch(left_imgs_clean, device)).cpu().numpy()
    base_sim = (left_emb_clean * right_emb).sum(axis=1)

    for r in range(grid):
        for c in range(grid):
            occ_imgs = []
            for img in left_imgs_clean:
                occ = img.copy()
                y0, x0 = r * cell, c * cell
                occ[y0:y0+cell, x0:x0+cell] = patch_color
                occ_imgs.append(occ)
            emb = encode_fn(_to_batch(occ_imgs, device)).cpu().numpy()
            sim = (emb * right_emb).sum(axis=1)
            drops[r, c] = float(np.mean(base_sim - sim))
            counts[r, c] = len(occ_imgs)
    return drops
