"""PyTorch dataset wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .modifications import MODIFICATION_TYPES, apply_modification


# ---------------------------------------------------------------------------

def _load_face(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    # standard face-recognition normalization: (x-127.5)/128
    t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
    t = (t - 127.5) / 128.0
    return t


# ---------------------------------------------------------------------------

class FaceClassificationDataset(Dataset):
    """Standard (image, identity-label) classification dataset."""

    def __init__(self, paths: List[Path], labels: List[int], image_size: int = 112,
                 augment: bool = True):
        assert len(paths) == len(labels)
        self.paths = paths
        self.labels = labels
        self.size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = _load_face(self.paths[idx], self.size)
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()  # horizontal flip
        return _to_tensor(img), int(self.labels[idx])


class PairedModificationDataset(Dataset):
    """
    Yields (clean, modified, identity, modification_class) tuples.

    Used by the novel method's identity-consistency loss and the adversarial
    modification-disentanglement head.
    """

    def __init__(self, paths: List[Path], labels: List[int], image_size: int = 112,
                 modifications=None, p_clean: float = 0.15):
        self.paths = paths
        self.labels = labels
        self.size = image_size
        self.modifications = modifications or MODIFICATION_TYPES
        self.p_clean = p_clean

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = _load_face(self.paths[idx], self.size)
        if np.random.rand() < self.p_clean:
            mod_kind = "clean"
        else:
            mod_kind = self.modifications[np.random.randint(0, len(self.modifications))]
        modded = apply_modification(img, mod_kind, seed=int(np.random.randint(0, 1 << 30)))
        return (
            _to_tensor(img),
            _to_tensor(modded),
            int(self.labels[idx]),
            int(self.modifications.index(mod_kind)),
        )


# ---------------------------------------------------------------------------

def make_loaders(train_ds: Dataset, val_ds: Dataset | None = None,
                 batch_size: int = 32, num_workers: int = 0):
    pin = torch.cuda.is_available()
    persistent = num_workers > 0
    prefetch = 4 if num_workers > 0 else None
    common = dict(
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    if prefetch is not None:
        common["prefetch_factor"] = prefetch
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common,
        )
    return train_loader, val_loader
