"""
MeGlass — real eyeglass face dataset (public; cleardusk/MeGlass, CCBR 2018).

1,710 identities / 47,917 cropped 120x120 images; every identity has at least
two black-eyeglass and two no-eyeglass photos. We build a REAL disguise
verification benchmark that complements the synthetic ``disguise_glasses``
modification: a genuine pair is (no-glass, glass) of the SAME person, an
impostor pair is (no-glass, glass) of DIFFERENT people. Matching across this
pair tests exactly the thesis claim — identity survives a real, worn alteration
that occludes the orbital region.

Materialisation
---------------
- ``meta.txt`` (``filename<space>label``; 1 = black-glass, 0 = no-glass) is
  fetched from the GitHub repo over plain HTTP.
- The 120x120 image archive is a large Google-Drive file, fetched with
  ``gdown`` if installed; otherwise the user is told how to drop the zip in
  place. The benchmark is skipped gracefully (empty pair list) if neither is
  available, mirroring the gated-loader behaviour.

Identity is the substring before the SECOND ``@`` in a filename, e.g.
``10032527@N08_identity_4@2582182573_0.jpg`` -> ``10032527@N08_identity_4``.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

from . import Benchmark, register
from ._common import cache_dir, download, unzip, write_pairs_tsv, read_pairs_tsv

META_URL = "https://raw.githubusercontent.com/cleardusk/MeGlass/master/meta.txt"
DRIVE_ID = "1V0c8p6MOlSFY5R-Hu9LxYZYLXd8B8j9q"
IMG_SUBDIR = "MeGlass_120x120"
N_POS = 1500
N_NEG = 1500
SEED = 0


def _identity(fname: str) -> str:
    parts = fname.split("@")
    return "@".join(parts[:2]) if len(parts) >= 2 else fname


def _ensure_meta(root: Path) -> Path:
    meta = root / "meta.txt"
    if not meta.exists():
        download(META_URL, meta, label="meglass meta.txt")
    return meta


def _ensure_images(root: Path) -> Path:
    img_dir = root / IMG_SUBDIR
    if img_dir.exists() and any(img_dir.glob("*.jpg")):
        return img_dir
    zip_path = root / "MeGlass_120x120.zip"
    if not (zip_path.exists() and zip_path.stat().st_size > 0):
        try:
            import gdown  # optional dependency, only needed on first download
            gdown.download(id=DRIVE_ID, output=str(zip_path), quiet=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [meglass] image archive unavailable ({e}); "
                  f"`pip install gdown` or place MeGlass_120x120.zip in {root}")
            return img_dir
    try:
        unzip(zip_path, root)
    except Exception:  # noqa: BLE001
        pass
    return img_dir


def _build_pairs(root: Path) -> List[Tuple[Path, Path, int]]:
    meta = _ensure_meta(root)
    img_dir = _ensure_images(root)
    if not meta.exists() or not img_dir.exists():
        return []

    glass: Dict[str, List[str]] = {}
    noglass: Dict[str, List[str]] = {}
    with open(meta, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            fname, lab = parts
            ident = _identity(fname)
            (glass if lab == "1" else noglass).setdefault(ident, []).append(fname)

    ids = sorted(set(glass) & set(noglass))
    if len(ids) < 2:
        return []

    rng = random.Random(SEED)
    pairs: List[Tuple[Path, Path, int]] = []

    def p(fn: str) -> Path:
        return img_dir / fn

    for _ in range(N_POS):
        i = rng.choice(ids)
        a, b = p(rng.choice(noglass[i])), p(rng.choice(glass[i]))
        if a.exists() and b.exists():
            pairs.append((a, b, 1))
    for _ in range(N_NEG):
        i, j = rng.sample(ids, 2)
        a, b = p(rng.choice(noglass[i])), p(rng.choice(glass[j]))
        if a.exists() and b.exists():
            pairs.append((a, b, 0))
    rng.shuffle(pairs)
    return pairs


def _materialise() -> Path:
    root = cache_dir("meglass")
    pairs_tsv = root / "pairs.tsv"
    if pairs_tsv.exists():
        return root
    pairs = _build_pairs(root)
    if pairs:
        write_pairs_tsv(pairs_tsv, pairs)
    return root


@register("meglass")
def load() -> Benchmark:
    root = _materialise()
    pairs_tsv = root / "pairs.tsv"
    pairs = read_pairs_tsv(pairs_tsv) if pairs_tsv.exists() else []
    return Benchmark(
        name="meglass", pairs=pairs, folds=None,
        notes="MeGlass real eyeglasses (public, MIT repo) — same-identity "
              "no-glass vs glass verification.",
    )
