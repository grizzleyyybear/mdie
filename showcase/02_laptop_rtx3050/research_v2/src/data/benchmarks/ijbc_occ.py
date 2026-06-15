"""
IJB-C with occlusion protocol.

Gated dataset — requires manual download from NIST. Once obtained, set
``IJBC_ROOT`` to the directory containing the IJB-C release.

If the meta file flagging occluded templates is absent, falls back to the full
1:1 protocol so the benchmark still produces a signal (documented in paper).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from . import Benchmark, register
from ._common import cache_dir, write_pairs_tsv, read_pairs_tsv, env_root


def _build_pairs(root: Path) -> List[Tuple[Path, Path, int]]:
    proto = root / "protocol" / "ijbc_template_pair_label.txt"
    meta = root / "protocol" / "ijbc_metadata.csv"
    img_root = root / "loose_crop"
    if not proto.exists() or not img_root.exists():
        return []

    tmpl_to_img: dict[int, Path] = {}
    tmpl_meta = root / "protocol" / "ijbc_face_tid_mid.txt"
    if tmpl_meta.exists():
        with open(tmpl_meta, encoding="utf-8") as f:
            for raw in f:
                parts = raw.strip().split()
                if len(parts) >= 2:
                    fname, tid = parts[0], int(parts[1])
                    tmpl_to_img.setdefault(tid, img_root / fname)

    occluded: set[int] | None = None
    if meta.exists():
        occluded = set()
        with open(meta, encoding="utf-8") as f:
            for raw in f:
                parts = raw.strip().split(",")
                if not parts:
                    continue
                try:
                    tid = int(parts[0])
                except ValueError:
                    continue
                if "occlu" in raw.lower():
                    occluded.add(tid)

    pairs: List[Tuple[Path, Path, int]] = []
    with open(proto, encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) != 3:
                continue
            a, b, lab = int(parts[0]), int(parts[1]), int(parts[2])
            if occluded and a not in occluded and b not in occluded:
                continue
            pa = tmpl_to_img.get(a); pb = tmpl_to_img.get(b)
            if pa and pb and pa.exists() and pb.exists():
                pairs.append((pa, pb, lab))
    return pairs


def _materialise() -> Path:
    root = cache_dir("ijbc_occ")
    pairs_tsv = root / "pairs.tsv"
    if pairs_tsv.exists():
        return root
    src = env_root("IJBC_ROOT")
    if src is None:
        return root
    write_pairs_tsv(pairs_tsv, _build_pairs(src))
    return root


@register("ijbc_occ")
def load() -> Benchmark:
    root = _materialise()
    pairs_tsv = root / "pairs.tsv"
    pairs = read_pairs_tsv(pairs_tsv) if pairs_tsv.exists() else []
    return Benchmark(name="ijbc_occ", pairs=pairs, folds=None,
                     notes="IJB-C occlusion subset — gated (set IJBC_ROOT).")
