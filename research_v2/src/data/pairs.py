"""Verification pair construction (LFW-style 10-fold protocol)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class PairSet:
    pairs: List[Tuple[Path, Path, int]]   # (img1, img2, same_identity)

    def __len__(self) -> int:
        return len(self.pairs)

    def split(self, n_folds: int = 10):
        """Yield (train_idx, test_idx) per fold (LFW 10-fold protocol)."""
        n = len(self.pairs)
        idx = np.arange(n)
        fold_size = n // n_folds
        for k in range(n_folds):
            test_idx = idx[k*fold_size:(k+1)*fold_size]
            train_idx = np.concatenate([idx[:k*fold_size], idx[(k+1)*fold_size:]])
            yield train_idx, test_idx


def build_verification_pairs(
    paths: List[Path],
    labels: List[int],
    n_pos: int = 1500,
    n_neg: int = 1500,
    seed: int = 0,
) -> PairSet:
    """
    Build a balanced set of positive / negative verification pairs.

    Positives: two distinct images of the same identity.
    Negatives: two images of distinct identities.
    """
    rng = np.random.RandomState(seed)
    paths = [Path(p) for p in paths]
    labels = np.array(labels)

    # group images by identity
    by_id: dict = {}
    for i, l in enumerate(labels):
        by_id.setdefault(int(l), []).append(i)
    eligible_ids = [k for k, v in by_id.items() if len(v) >= 2]
    all_ids = list(by_id.keys())

    pairs: List[Tuple[Path, Path, int]] = []
    # positives
    for _ in range(n_pos):
        pid = eligible_ids[rng.randint(len(eligible_ids))]
        i, j = rng.choice(by_id[pid], size=2, replace=False)
        pairs.append((paths[i], paths[j], 1))
    # negatives
    for _ in range(n_neg):
        a, b = rng.choice(all_ids, size=2, replace=False)
        i = by_id[int(a)][rng.randint(len(by_id[int(a)]))]
        j = by_id[int(b)][rng.randint(len(by_id[int(b)]))]
        pairs.append((paths[i], paths[j], 0))

    rng.shuffle(pairs)
    return PairSet(pairs)
