"""Dataset, sampler, and verification-pair utilities for MDIE."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Sampler
except ModuleNotFoundError:  # lets pair/sampler utilities import without PyTorch
    torch = None
    DataLoader = None

    class Dataset:
        pass

    class Sampler:
        def __class_getitem__(cls, _item):
            return cls


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
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for datasets and loaders. Install dependencies with "
            "`pip install -r requirements.txt`."
        )
    return torch


# ---------------------------------------------------------------------------

def _load_face(path: Path, size: int) -> np.ndarray:
    cv2 = _require_cv2()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    torch_mod = _require_torch()
    # standard face-recognition normalization: (x-127.5)/128
    t = torch_mod.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
    t = (t - 127.5) / 128.0
    return t


# ---------------------------------------------------------------------------

@dataclass
class PairSet:
    """Balanced image-pair protocol for verification evaluation."""

    pairs: List[Tuple[Path, Path, int]]

    def __len__(self) -> int:
        return len(self.pairs)

    def split(self, n_folds: int = 10):
        """Yield (train_idx, test_idx) per fold using the LFW-style protocol."""
        n = len(self.pairs)
        if n_folds <= 0:
            raise ValueError("n_folds must be positive")
        if n < n_folds:
            raise ValueError(f"cannot split {n} pairs into {n_folds} folds")
        idx = np.arange(n)
        fold_size = n // n_folds
        for k in range(n_folds):
            test_idx = idx[k * fold_size:(k + 1) * fold_size]
            train_idx = np.concatenate([idx[:k * fold_size], idx[(k + 1) * fold_size:]])
            yield train_idx, test_idx


def build_verification_pairs(
    paths: List[Path],
    labels: List[int],
    n_pos: int = 1500,
    n_neg: int = 1500,
    seed: int = 0,
) -> PairSet:
    """Build a balanced set of positive and negative verification pairs."""
    if len(paths) != len(labels):
        raise ValueError("paths and labels must have the same length")
    if n_pos < 0 or n_neg < 0:
        raise ValueError("n_pos and n_neg must be non-negative")

    rng = np.random.RandomState(seed)
    paths = [Path(p) for p in paths]
    labels = np.array(labels)

    by_id: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        by_id.setdefault(int(label), []).append(i)

    eligible_ids = [identity for identity, ix in by_id.items() if len(ix) >= 2]
    all_ids = list(by_id.keys())
    if n_pos and not eligible_ids:
        raise ValueError("positive pairs require at least one identity with two images")
    if n_neg and len(all_ids) < 2:
        raise ValueError("negative pairs require at least two identities")
    pairs: List[Tuple[Path, Path, int]] = []

    for _ in range(n_pos):
        pid = eligible_ids[rng.randint(len(eligible_ids))]
        i, j = rng.choice(by_id[pid], size=2, replace=False)
        pairs.append((paths[i], paths[j], 1))

    for _ in range(n_neg):
        a, b = rng.choice(all_ids, size=2, replace=False)
        i = by_id[int(a)][rng.randint(len(by_id[int(a)]))]
        j = by_id[int(b)][rng.randint(len(by_id[int(b)]))]
        pairs.append((paths[i], paths[j], 0))

    rng.shuffle(pairs)
    return PairSet(pairs)


# ---------------------------------------------------------------------------

class IdentityBalancedSampler(Sampler[List[int]]):
    """
    Emit batches with multiple identities and multiple samples per identity.

    Standard random sampling rarely puts more than one example of the same
    identity in a mini-batch when the identity count is high. ArcFace and ICCL
    both benefit from meaningful in-batch positives and hard negatives.
    """

    def __init__(self, labels, classes_per_batch: int, samples_per_class: int,
                 num_batches: int | None = None, seed: int = 0,
                 rank: int | None = None, num_replicas: int | None = None):
        if classes_per_batch <= 0 or samples_per_class <= 0:
            raise ValueError("classes_per_batch and samples_per_class must be positive")
        if num_batches is not None and num_batches <= 0:
            raise ValueError("num_batches must be positive when provided")

        self.labels = np.asarray(labels)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)

        self._by_class: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self._by_class[int(label)].append(idx)

        self.usable = [
            cls for cls, indices in self._by_class.items()
            if len(indices) >= self.samples_per_class
        ]
        if len(self.usable) < self.classes_per_batch:
            raise ValueError(
                f"Need {self.classes_per_batch} usable classes with at least "
                f"{self.samples_per_class} samples; found {len(self.usable)}."
            )

        self.batch_size = self.classes_per_batch * self.samples_per_class
        full_batches = (
            int(num_batches)
            if num_batches is not None
            else max(1, len(self.labels) // self.batch_size)
        )
        # Distributed: each rank draws a disjoint slice of the epoch's batches
        # (a different RNG seed per rank) so the global epoch covers ~all data
        # while keeping the in-batch identity structure ICCL relies on.
        if num_replicas and num_replicas > 1:
            self.num_batches = max(1, full_batches // int(num_replicas))
            self.rng = np.random.RandomState(seed + int(rank or 0))
        else:
            self.num_batches = full_batches
            self.rng = np.random.RandomState(seed)

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_batches):
            chosen_classes = self.rng.choice(
                self.usable, size=self.classes_per_batch, replace=False
            )
            indices: List[int] = []
            for cls in chosen_classes:
                pool = self._by_class[int(cls)]
                pick = self.rng.choice(pool, size=self.samples_per_class, replace=False)
                indices.extend(int(i) for i in pick)
            self.rng.shuffle(indices)
            yield indices

    def __len__(self) -> int:
        return self.num_batches


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
                 modifications=None, p_clean: float = 0.15, bone_targets=None,
                 grid: int = 14):
        if not 0.0 <= p_clean <= 1.0:
            raise ValueError("p_clean must be between 0 and 1")
        self.paths = paths
        self.labels = labels
        self.size = image_size
        if modifications is None:
            from .modifications import MODIFICATION_TYPES
            modifications = MODIFICATION_TYPES
        self.modifications = modifications
        self.p_clean = p_clean
        # Optional per-image rigid bone-landmark attention targets (dict keyed by
        # str(path) -> (grid, grid) float array summing to 1). Used to supervise
        # RATA attention onto each face's detected bones. Missing/undetected
        # faces get an all-zero target (the attention loss ignores those).
        self.bone_targets = bone_targets
        self.grid = grid

    def __len__(self) -> int:
        return len(self.paths)

    def _bone_target(self, idx: int):
        torch_mod = _require_torch()
        g = self.grid
        if self.bone_targets is not None:
            t = self.bone_targets.get(str(self.paths[idx]))
            if t is not None:
                return torch_mod.from_numpy(np.asarray(t, dtype=np.float32))
        return torch_mod.zeros((g, g), dtype=torch_mod.float32)

    def __getitem__(self, idx: int):
        img = _load_face(self.paths[idx], self.size)
        if np.random.rand() < self.p_clean:
            mod_kind = "clean"
        else:
            mod_kind = self.modifications[np.random.randint(0, len(self.modifications))]
        from .modifications import apply_modification
        modded = apply_modification(img, mod_kind, seed=int(np.random.randint(0, 1 << 30)))
        return (
            _to_tensor(img),
            _to_tensor(modded),
            int(self.labels[idx]),
            int(self.modifications.index(mod_kind)),
            self._bone_target(idx),
        )


# ---------------------------------------------------------------------------

def make_loaders(train_ds: Dataset, val_ds: Dataset | None = None,
                 batch_size: int = 32, num_workers: int = 0):
    torch_mod = _require_torch()
    if DataLoader is None:
        raise ModuleNotFoundError(
            "PyTorch DataLoader is unavailable. Install dependencies with "
            "`pip install -r requirements.txt`."
        )
    pin = torch_mod.cuda.is_available()
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
