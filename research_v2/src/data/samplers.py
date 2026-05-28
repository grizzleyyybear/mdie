"""
Lightweight, identity-balanced sampler for face-classification training.

Standard random sampling rarely puts > 1 example of the same identity in a
mini-batch when N >> batch_size. ArcFace / ICCL both benefit from seeing
multiple samples per identity per batch (in-batch hard negatives become
meaningful). This sampler emits ``batch_size = n_classes_per_batch *
n_per_class`` indices per yielded batch.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, List

import numpy as np
from torch.utils.data import Sampler


class IdentityBalancedSampler(Sampler[List[int]]):
    def __init__(self, labels, classes_per_batch: int, samples_per_class: int,
                 num_batches: int | None = None, seed: int = 0):
        # Sampler.__init__ takes no args in modern PyTorch; pre-2.0 ignored.
        self.labels = np.asarray(labels)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)

        self._by_class: dict[int, list[int]] = defaultdict(list)
        for idx, lbl in enumerate(self.labels):
            self._by_class[int(lbl)].append(idx)
        # Discard ultra-rare classes that cannot fill samples_per_class.
        self.usable = [c for c, ix in self._by_class.items()
                       if len(ix) >= self.samples_per_class]
        if len(self.usable) < self.classes_per_batch:
            raise ValueError(
                f"Need {self.classes_per_batch} usable classes with >= "
                f"{self.samples_per_class} samples; only {len(self.usable)} "
                f"meet the bar.")

        self.batch_size = self.classes_per_batch * self.samples_per_class
        self.num_batches = int(num_batches) if num_batches is not None else \
            max(1, len(self.labels) // self.batch_size)
        self.rng = np.random.RandomState(seed)

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_batches):
            chosen_classes = self.rng.choice(
                self.usable, size=self.classes_per_batch, replace=False)
            indices: List[int] = []
            for c in chosen_classes:
                pool = self._by_class[int(c)]
                pick = self.rng.choice(pool, size=self.samples_per_class,
                                       replace=False)
                indices.extend(int(i) for i in pick)
            self.rng.shuffle(indices)
            yield indices

    def __len__(self) -> int:
        return self.num_batches
