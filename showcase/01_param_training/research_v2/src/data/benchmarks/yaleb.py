"""
Extended Yale-B — the canonical illumination-sweep benchmark.

~28-38 subjects captured under a 64-illumination sweep, the standard stress
test for low-light and harsh-shadow robustness. This loader is OFFLINE-FIRST
and gated: stage the data and point ``YALEB_ROOT`` at a directory of
per-identity subfolders::

    $YALEB_ROOT/<subject_id>/<img>.{pgm,jpg,png,jpeg}

Genuine pairs span different illuminations of the same identity; impostor pairs
are cross-identity, balanced one-to-one with the genuine count. When nothing is
staged the loader returns an empty Benchmark without raising.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

from . import Benchmark, register
from ._common import cache_dir, write_pairs_tsv, read_pairs_tsv, env_root

_IMG_SUFFIXES = {".pgm", ".jpg", ".jpeg", ".png"}


def _images(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES)


def _identity_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if _images(d):
            out.append(d)
    return out


def _build_pairs(root: Path, cap: int = 3000,
                 seed: int = 42) -> List[Tuple[Path, Path, int]]:
    rng = random.Random(seed)
    id_dirs = _identity_dirs(root)
    if not id_dirs:
        return []

    imgs_by_id = {d.name: _images(d) for d in id_dirs}
    ids = [d.name for d in id_dirs]

    genuine: List[Tuple[Path, Path, int]] = []
    for ident in ids:
        imgs = imgs_by_id[ident]
        if len(imgs) < 2:
            continue
        for i in range(len(imgs) - 1):
            genuine.append((imgs[i], imgs[i + 1], 1))
    rng.shuffle(genuine)
    if len(genuine) > cap:
        genuine = genuine[:cap]

    impostor: List[Tuple[Path, Path, int]] = []
    n_neg = len(genuine)
    multi = [i for i in ids if imgs_by_id[i]]
    attempts = 0
    max_attempts = n_neg * 20 + 100
    seen = set()
    while len(impostor) < n_neg and attempts < max_attempts and len(multi) >= 2:
        attempts += 1
        i1, i2 = rng.sample(multi, 2)
        a = rng.choice(imgs_by_id[i1])
        b = rng.choice(imgs_by_id[i2])
        key = (str(a), str(b))
        if key in seen:
            continue
        seen.add(key)
        impostor.append((a, b, 0))

    pairs = genuine + impostor
    rng.shuffle(pairs)
    return pairs


def _materialise() -> Path:
    root = env_root("YALEB_ROOT") or cache_dir("yaleb")
    pairs_tsv = root / "pairs.tsv"
    if pairs_tsv.exists():
        return root
    if _identity_dirs(root):
        write_pairs_tsv(pairs_tsv, _build_pairs(root))
    return root


@register("yaleb")
def load() -> Benchmark:
    root = _materialise()
    pairs_tsv = root / "pairs.tsv"
    pairs = read_pairs_tsv(pairs_tsv) if pairs_tsv.exists() else []
    return Benchmark(
        name="yaleb", pairs=pairs, folds=None,
        notes="Extended Yale-B — 64-illumination sweep for low-light/shadow robustness; staged offline via YALEB_ROOT.",
    )
