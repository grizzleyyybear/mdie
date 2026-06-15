"""
MFR2 — Masked Face Recognition v2 (public).
~53 identities, ~269 verification pairs.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from . import Benchmark, register
from ._common import cache_dir, download, unzip, write_pairs_tsv, read_pairs_tsv

MIRRORS = [
    # Canonical source (Google Drive, file ID from MaskTheFace/datasets/download_links.txt).
    "https://drive.usercontent.google.com/download?id=1ukk0n_srRqcsotK2MjlFPj7L0sXcR2fH&export=download&authuser=0",
    "https://drive.google.com/uc?export=download&id=1ukk0n_srRqcsotK2MjlFPj7L0sXcR2fH",
    # Dead mirrors kept for audit trail (404 / 401):
    # "https://github.com/aqeelanwar/MaskTheFace/raw/master/mfr2.zip",
    # "https://huggingface.co/datasets/MDIE-mirror/mfr2/resolve/main/mfr2.zip",
]


def _materialise() -> Path:
    root = cache_dir("mfr2")
    pairs_tsv = root / "pairs.tsv"
    if pairs_tsv.exists():
        return root

    bin_path = next(root.glob("*.bin"), None)
    if bin_path is not None:
        from ._bin_parser import extract_bin
        pairs = extract_bin(bin_path, root)
        write_pairs_tsv(pairs_tsv, pairs)
        return root

    zip_path = root / "mfr2.zip"
    extracted = root / "mfr2"
    if not extracted.exists() or not any(extracted.rglob("*.png")):
        for url in MIRRORS:
            download(url, zip_path, label="mfr2.zip")
            if zip_path.exists() and zip_path.stat().st_size > 0:
                try:
                    unzip(zip_path, root)
                    break
                except Exception:
                    continue
    if not extracted.exists() or not any(extracted.rglob("*.png")):
        return root

    def _img(name: str, idx: str) -> Path:
        return extracted / name / f"{name}_{int(idx):04d}.png"

    pairs_txt = next(extracted.rglob("pairs.txt"), None)
    pairs: List[Tuple[Path, Path, int]] = []
    if pairs_txt is not None:
        with open(pairs_txt, encoding="utf-8") as f:
            for raw in f:
                parts = raw.strip().split()
                if len(parts) == 3:
                    name, a, b = parts
                    pa, pb = _img(name, a), _img(name, b)
                    same = 1
                elif len(parts) == 4:
                    n1, a, n2, b = parts
                    pa, pb = _img(n1, a), _img(n2, b)
                    same = 0
                else:
                    continue
                if pa.exists() and pb.exists():
                    pairs.append((pa, pb, same))
    write_pairs_tsv(pairs_tsv, pairs)
    return root


@register("mfr2")
def load() -> Benchmark:
    root = _materialise()
    pairs_tsv = root / "pairs.tsv"
    pairs = read_pairs_tsv(pairs_tsv) if pairs_tsv.exists() else []
    return Benchmark(
        name="mfr2", pairs=pairs, folds=None,
        notes="Masked Face Recognition 2 — public, no folds.",
    )
