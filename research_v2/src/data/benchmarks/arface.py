"""
AR Face DB — controlled-condition portraits with real occlusions.

Each subject is captured under varied expression, illumination, and two
canonical occlusions: sunglasses and a scarf (the exact occlusion set that
PDSN / FROM report on). This loader is OFFLINE-FIRST and gated: stage the
data and point ``ARFACE_ROOT`` at a directory of per-identity subfolders::

    $ARFACE_ROOT/<subject_id>/<img>.{jpg,png,bmp,pgm,jpeg}

AR filenames usually encode the capture condition; when a filename hints at an
occlusion (e.g. contains ``sunglass`` or ``scarf``) the genuine pairs prefer an
occluded-vs-neutral pairing. When nothing is staged the loader returns an empty
Benchmark without raising.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

from . import Benchmark, register
from ._common import cache_dir, write_pairs_tsv, read_pairs_tsv, env_root

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".pgm"}
_OCCLUSION_HINTS = ("sunglass", "scarf")


def _images(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES)


def _is_occluded(p: Path) -> bool:
    name = p.name.lower()
    return any(h in name for h in _OCCLUSION_HINTS)


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
        occluded = [p for p in imgs if _is_occluded(p)]
        neutral = [p for p in imgs if not _is_occluded(p)]
        local: List[Tuple[Path, Path]] = []
        if occluded and neutral:
            for occ in occluded:
                local.append((occ, rng.choice(neutral)))
        else:
            for i in range(len(imgs) - 1):
                local.append((imgs[i], imgs[i + 1]))
        for a, b in local:
            genuine.append((a, b, 1))
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
    root = env_root("ARFACE_ROOT") or cache_dir("arface")
    pairs_tsv = root / "pairs.tsv"
    if pairs_tsv.exists():
        return root
    if _identity_dirs(root):
        write_pairs_tsv(pairs_tsv, _build_pairs(root))
    return root


@register("arface")
def load() -> Benchmark:
    root = _materialise()
    pairs_tsv = root / "pairs.tsv"
    pairs = read_pairs_tsv(pairs_tsv) if pairs_tsv.exists() else []
    return Benchmark(
        name="arface", pairs=pairs, folds=None,
        notes="AR Face DB — real sunglasses/scarf occlusion; staged offline via ARFACE_ROOT.",
    )
