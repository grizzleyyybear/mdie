"""
IIITD Plastic Surgery Face Dataset.

Gated dataset — requires manual download from IIITD after signing a research
agreement. Once obtained, set ``IIITD_ROOT`` to the directory containing the
``before`` and ``after`` subdirectories of cropped face images.

Standard layout assumed:
    $IIITD_ROOT/before/<subject_id>/<img>.jpg
    $IIITD_ROOT/after/<subject_id>/<img>.jpg
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

from . import Benchmark, register
from ._common import cache_dir, write_pairs_tsv, read_pairs_tsv, env_root


def _build_pairs(root: Path, n_pos: int = 900, n_neg: int = 900,
                  seed: int = 0) -> List[Tuple[Path, Path, int]]:
    rng = random.Random(seed)
    before = root / "before"
    after = root / "after"
    if not before.exists() or not after.exists():
        return []
    subj = sorted(p.name for p in before.iterdir()
                  if p.is_dir() and (after / p.name).exists())
    if not subj:
        return []

    def imgs(d: Path) -> List[Path]:
        return sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))

    pairs: List[Tuple[Path, Path, int]] = []
    for _ in range(n_pos):
        s = rng.choice(subj)
        ba, aa = imgs(before / s), imgs(after / s)
        if ba and aa:
            pairs.append((rng.choice(ba), rng.choice(aa), 1))
    for _ in range(n_neg):
        if len(subj) < 2:
            break
        s1, s2 = rng.sample(subj, 2)
        ba, ab = imgs(before / s1), imgs(after / s2)
        if ba and ab:
            pairs.append((rng.choice(ba), rng.choice(ab), 0))
    rng.shuffle(pairs)
    return pairs


def _materialise() -> Path:
    root = cache_dir("iiitd_surgery")
    pairs_tsv = root / "pairs.tsv"
    if pairs_tsv.exists():
        return root
    src = env_root("IIITD_ROOT")
    if src is None:
        return root
    write_pairs_tsv(pairs_tsv, _build_pairs(src))
    return root


@register("iiitd_surgery")
def load() -> Benchmark:
    root = _materialise()
    pairs_tsv = root / "pairs.tsv"
    pairs = read_pairs_tsv(pairs_tsv) if pairs_tsv.exists() else []
    return Benchmark(name="iiitd_surgery", pairs=pairs, folds=None,
                     notes="IIITD Plastic Surgery — gated (set IIITD_ROOT).")
