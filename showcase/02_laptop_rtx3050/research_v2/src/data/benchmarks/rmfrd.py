"""
RMFRD / RMFD — Real-world Masked Face Recognition Dataset.

Real worn-mask + unmasked face images organised as per-identity folders
(ImageFolder layout). PARAM has no internet, so this loader is OFFLINE-FIRST
and gated: it returns an empty Benchmark when nothing is staged.

Stage the data either by pointing ``RMFRD_ROOT`` at a directory of per-identity
subfolders, or by dropping such subfolders into the cache dir for "rmfrd".

Standard layout assumed:
    $RMFRD_ROOT/<identity_id>/<img>.jpg   (masked + unmasked images mixed)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

from . import Benchmark, register
from ._common import cache_dir, write_pairs_tsv, read_pairs_tsv, env_root

# MIRRORS kept for audit trail only; the active path is offline/staged-only
# (no network access on PARAM).
# MIRRORS = [
#     "https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset",
# ]

SEED = 42
MAX_GENUINE = 3000
MAX_IMPOSTOR = 3000

_IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")


def _imgs(d: Path) -> List[Path]:
    out: List[Path] = []
    for ext in _IMG_EXTS:
        out.extend(d.glob(ext))
    return sorted(out)


def _identities(root: Path) -> List[Tuple[str, List[Path]]]:
    ids: List[Tuple[str, List[Path]]] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = _imgs(sub)
        if imgs:
            ids.append((sub.name, imgs))
    return ids


def _build_pairs(root: Path) -> List[Tuple[Path, Path, int]]:
    ids = _identities(root)
    if not ids:
        return []

    rng = random.Random(SEED)

    genuine: List[Tuple[Path, Path, int]] = []
    for _, imgs in ids:
        if len(imgs) < 2:
            continue
        a, b = rng.sample(imgs, 2)
        genuine.append((a, b, 1))
    rng.shuffle(genuine)
    genuine = genuine[:MAX_GENUINE]

    impostor: List[Tuple[Path, Path, int]] = []
    if len(ids) >= 2:
        target = min(MAX_IMPOSTOR, len(genuine)) if genuine else 0
        attempts = 0
        max_attempts = target * 20 + 1
        while len(impostor) < target and attempts < max_attempts:
            attempts += 1
            (n1, i1), (n2, i2) = rng.sample(ids, 2)
            impostor.append((rng.choice(i1), rng.choice(i2), 0))

    pairs = genuine + impostor
    rng.shuffle(pairs)
    return pairs


def _materialise() -> Path:
    root = env_root("RMFRD_ROOT") or cache_dir("rmfrd")
    pairs_tsv = root / "pairs.tsv"
    if pairs_tsv.exists():
        return root
    pairs = _build_pairs(root)
    if pairs:
        write_pairs_tsv(pairs_tsv, pairs)
    return root


@register("rmfrd")
def load() -> Benchmark:
    root = _materialise()
    pairs_tsv = root / "pairs.tsv"
    pairs = read_pairs_tsv(pairs_tsv) if pairs_tsv.exists() else []
    return Benchmark(
        name="rmfrd", pairs=pairs, folds=None,
        notes="Real-world Masked Face Recognition Dataset — staged offline "
              "via RMFRD_ROOT; real worn masks.",
    )
