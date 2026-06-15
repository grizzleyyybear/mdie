"""CALFW — Cross-Age LFW. 6000 pairs, 10-fold. Public."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

from . import Benchmark, register
from ._common import cache_dir, write_pairs_tsv, read_pairs_tsv


def _materialise() -> Path:
    root = cache_dir("calfw")
    pairs_tsv = root / "pairs.tsv"
    folds_tsv = root / "folds.tsv"
    if pairs_tsv.exists():
        return root

    # InsightFace .bin fallback (if user drops one in)
    bin_path = next(root.glob("*.bin"), None)
    if bin_path is not None:
        from ._bin_parser import extract_bin
        pairs = extract_bin(bin_path, root)
        write_pairs_tsv(pairs_tsv, pairs)
        fold_size = max(1, len(pairs) // 10)
        with open(folds_tsv, "w", encoding="utf-8") as f:
            for k in range(10):
                lo, hi = k * fold_size, (k + 1) * fold_size
                f.write(",".join(str(i) for i in range(lo, hi)) + "\n")
        return root

    # Hugging Face mirror (marcelohaps/calfw): pairs.csv + aligned/images/<shard>/<file>.jpg
    hf_root = root / "hf"
    pairs_csv = hf_root / "pairs.csv"
    img_root = hf_root / "aligned"
    if not pairs_csv.exists() or not img_root.exists():
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="marcelohaps/calfw", repo_type="dataset",
                local_dir=str(hf_root),
                allow_patterns=["pairs.csv", "aligned/**", "README.md"],
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [calfw] HF download failed: {e}")
            return root

    if not pairs_csv.exists():
        return root

    pairs: List[Tuple[Path, Path, int]] = []
    fold_indices: List[List[int]] = [[] for _ in range(10)]
    with open(pairs_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pa = img_root / row["image_a_aligned_path"]
            pb = img_root / row["image_b_aligned_path"]
            if not (pa.exists() and pb.exists()):
                continue
            same = int(row["is_same"])
            fold_id = int(row["fold_id"]) - 1
            pairs.append((pa, pb, same))
            if 0 <= fold_id < 10:
                fold_indices[fold_id].append(len(pairs) - 1)
    write_pairs_tsv(pairs_tsv, pairs)
    with open(folds_tsv, "w", encoding="utf-8") as f:
        for fold in fold_indices:
            f.write(",".join(str(x) for x in fold) + "\n")
    return root


@register("calfw")
def load() -> Benchmark:
    root = _materialise()
    pairs_tsv = root / "pairs.tsv"
    folds_tsv = root / "folds.tsv"
    pairs = read_pairs_tsv(pairs_tsv) if pairs_tsv.exists() else []
    folds = None
    if folds_tsv.exists():
        folds = []
        for line in folds_tsv.read_text(encoding="utf-8").splitlines():
            if line.strip():
                folds.append([int(x) for x in line.split(",")])
    return Benchmark(name="calfw", pairs=pairs, folds=folds,
                     notes="Cross-Age LFW — 6000 pairs, 10-fold.")
