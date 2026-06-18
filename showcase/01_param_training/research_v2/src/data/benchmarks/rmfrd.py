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

# Real RMFRD ships as two parallel identity trees: masked and unmasked. We detect
# them so genuine pairs can be the HARD masked<->unmasked case rather than two
# images of the same condition.
_MASKED_DIRNAMES = ("AFDB_masked_face_dataset", "masked", "masked_face",
                    "self-built-masked-face-recognition-dataset")
_UNMASKED_DIRNAMES = ("AFDB_face_dataset", "unmasked", "face", "normal")


def _imgs(d: Path) -> List[Path]:
    out: List[Path] = []
    for ext in _IMG_EXTS:
        out.extend(d.glob(ext))
    return sorted(out)


def _find_tree(root: Path, names: Tuple[str, ...]) -> Path | None:
    """Locate a masked/unmasked sub-tree by known directory names (any depth
    up to 2), so RMFRD_ROOT can point at either the dataset root or a parent."""
    for name in names:
        cand = root / name
        if cand.is_dir():
            return cand
    for sub in root.iterdir() if root.is_dir() else []:
        if sub.is_dir() and sub.name in names:
            return sub
        if sub.is_dir():
            for sub2 in sub.iterdir():
                if sub2.is_dir() and sub2.name in names:
                    return sub2
    return None


def _identities(root: Path) -> List[Tuple[str, List[Path]]]:
    ids: List[Tuple[str, List[Path]]] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = _imgs(sub)
        if imgs:
            ids.append((sub.name, imgs))
    return ids


def _identities_split(root: Path):
    """Return {identity: (masked_imgs, unmasked_imgs)} if the two-tree RMFRD
    layout is present, else None (fall back to the flat ImageFolder builder)."""
    masked_root = _find_tree(root, _MASKED_DIRNAMES)
    unmasked_root = _find_tree(root, _UNMASKED_DIRNAMES)
    if masked_root is None or unmasked_root is None:
        return None
    masked = {name: imgs for name, imgs in _identities(masked_root)}
    unmasked = {name: imgs for name, imgs in _identities(unmasked_root)}
    combined: dict = {}
    for name in set(masked) | set(unmasked):
        combined[name] = (masked.get(name, []), unmasked.get(name, []))
    return combined


def _build_pairs_split(split: dict) -> List[Tuple[Path, Path, int]]:
    """Build pairs from the masked/unmasked split. Genuine pairs are the HARD
    masked<->unmasked case (same identity); impostors are masked-vs-unmasked
    across different identities."""
    rng = random.Random(SEED)
    # identities that have at least one masked AND one unmasked image
    usable = [(n, m, u) for n, (m, u) in split.items() if m and u]
    if len(usable) < 2:
        return []

    genuine: List[Tuple[Path, Path, int]] = []
    for _, m, u in usable:
        genuine.append((rng.choice(m), rng.choice(u), 1))
    rng.shuffle(genuine)
    genuine = genuine[:MAX_GENUINE]

    impostor: List[Tuple[Path, Path, int]] = []
    target = min(MAX_IMPOSTOR, len(genuine))
    attempts, max_attempts = 0, target * 20 + 1
    while len(impostor) < target and attempts < max_attempts:
        attempts += 1
        (n1, m1, _u1), (n2, _m2, u2) = rng.sample(usable, 2)
        if m1 and u2:
            impostor.append((rng.choice(m1), rng.choice(u2), 0))

    pairs = genuine + impostor
    rng.shuffle(pairs)
    return pairs


def _build_pairs(root: Path) -> List[Tuple[Path, Path, int]]:
    # Prefer the real two-tree (masked/unmasked) layout when present.
    split = _identities_split(root)
    if split is not None:
        pairs = _build_pairs_split(split)
        if pairs:
            return pairs

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
              "via RMFRD_ROOT; genuine pairs are masked<->unmasked of the same "
              "identity (hard case) when the two-tree layout is present.",
    )
