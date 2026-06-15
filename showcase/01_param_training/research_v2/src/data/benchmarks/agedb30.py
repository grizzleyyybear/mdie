"""AgeDB-30 — 30-year-gap age protocol. Pairs are constructed from
the marcelohaps/agedb metadata using a fixed random seed so the protocol
is reproducible.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

from . import Benchmark, register
from ._common import cache_dir, write_pairs_tsv, read_pairs_tsv

# AgeDB-30 canonical protocol: 6000 pairs, 10 folds of 600, balanced.
N_PAIRS = 6000
N_FOLDS = 10
MIN_AGE_GAP = 30
SEED = 20251127


def _build_pairs(metadata_csv: Path, img_root: Path):
    rows: list[dict] = []
    with open(metadata_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_id[r["identity"]].append(r)

    rng = random.Random(SEED)

    # positives: same identity, |age_a - age_b| >= 30
    positives: list[tuple[str, str]] = []
    eligible = [imgs for imgs in by_id.values() if len(imgs) >= 2 and
                (max(int(i["age"]) for i in imgs) -
                 min(int(i["age"]) for i in imgs)) >= MIN_AGE_GAP]
    while len(positives) < N_PAIRS // 2 and eligible:
        imgs = rng.choice(eligible)
        a, b = rng.sample(imgs, 2)
        if abs(int(a["age"]) - int(b["age"])) >= MIN_AGE_GAP:
            positives.append((a["file_name"], b["file_name"]))

    # negatives: different identity
    all_ids = list(by_id.keys())
    negatives: list[tuple[str, str]] = []
    while len(negatives) < N_PAIRS // 2:
        ida, idb = rng.sample(all_ids, 2)
        a = rng.choice(by_id[ida])
        b = rng.choice(by_id[idb])
        negatives.append((a["file_name"], b["file_name"]))

    # interleave so folds balance same/diff
    pairs: list[tuple[Path, Path, int]] = []
    for i in range(N_PAIRS // 2):
        for (fa, fb), same in ((positives[i], 1), (negatives[i], 0)):
            pa = img_root / fa
            pb = img_root / fb
            if pa.exists() and pb.exists():
                pairs.append((pa, pb, same))
    return pairs


def _materialise() -> Path:
    root = cache_dir("agedb30")
    pairs_tsv = root / "pairs.tsv"
    folds_tsv = root / "folds.tsv"
    if pairs_tsv.exists():
        return root

    # InsightFace .bin fallback
    bin_path = next(root.glob("*.bin"), None)
    if bin_path is not None:
        from ._bin_parser import extract_bin
        pairs = extract_bin(bin_path, root)
        write_pairs_tsv(pairs_tsv, pairs)
        fold_size = max(1, len(pairs) // N_FOLDS)
        with open(folds_tsv, "w", encoding="utf-8") as f:
            for k in range(N_FOLDS):
                lo, hi = k * fold_size, (k + 1) * fold_size
                f.write(",".join(str(i) for i in range(lo, hi)) + "\n")
        return root

    # Hugging Face mirror (marcelohaps/agedb)
    hf_root = root / "hf"
    metadata_csv = hf_root / "train" / "metadata.csv"
    img_root = hf_root / "train"
    if not metadata_csv.exists():
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="marcelohaps/agedb", repo_type="dataset",
                local_dir=str(hf_root),
                allow_patterns=["train/**", "README.md"],
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [agedb30] HF download failed: {e}")
            return root

    if not metadata_csv.exists():
        return root

    pairs = _build_pairs(metadata_csv, img_root)
    write_pairs_tsv(pairs_tsv, pairs)
    fold_size = max(1, len(pairs) // N_FOLDS)
    with open(folds_tsv, "w", encoding="utf-8") as f:
        for k in range(N_FOLDS):
            lo, hi = k * fold_size, (k + 1) * fold_size
            f.write(",".join(str(i) for i in range(lo, hi)) + "\n")
    return root


@register("agedb30")
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
    return Benchmark(name="agedb30", pairs=pairs, folds=folds,
                     notes="AgeDB-30 - 6000 pairs, 10-fold, >=30y age gap "
                           "(constructed from marcelohaps/agedb metadata, seed 20251127).")
