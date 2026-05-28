"""
Cheap during-training verification AUC.

Computes verification AUC on a small fixed pair pool every epoch so the
trainer can keep ``best.pt`` = best AUC, not best train loss. Catches
overfitting and gives a real selection signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch

from ..config import SETTINGS


def _load_face_tensor(path: Path, size: int) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
    return (t - 127.5) / 128.0


@torch.no_grad()
def quick_verification_auc(
    encode: Callable[[torch.Tensor], torch.Tensor],
    pairs: List[Tuple[Path, Path, int]],
    device: torch.device,
    batch_size: int = 64,
    image_size: int | None = None,
) -> float:
    """Return ROC AUC on the given verification pair list.

    Pairs are tuples ``(path_a, path_b, is_same)``.
    """
    if image_size is None:
        image_size = SETTINGS.train.image_size
    if not pairs:
        return float("nan")

    left = torch.stack([_load_face_tensor(p[0], image_size) for p in pairs])
    right = torch.stack([_load_face_tensor(p[1], image_size) for p in pairs])
    labels = np.asarray([int(p[2]) for p in pairs], dtype=np.int32)

    def _embed(x: torch.Tensor) -> np.ndarray:
        out = []
        for i in range(0, x.size(0), batch_size):
            chunk = x[i:i + batch_size].to(device, non_blocking=True)
            emb = encode(chunk)
            emb = torch.nn.functional.normalize(emb, dim=1)
            out.append(emb.detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    le = _embed(left); re = _embed(right)
    scores = (le * re).sum(axis=1)

    # Lightweight AUC via Mann-Whitney U — avoids importing sklearn here.
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    auc = (ranks[labels == 1].sum() - pos.size * (pos.size + 1) / 2.0) / (
        pos.size * neg.size)
    return float(auc)
