"""
Parser for InsightFace evaluation .bin files.

The InsightFace project distributes the standard face-verification benchmarks
(LFW, CFP-FP, CFP-FF, AgeDB-30, CALFW, CPLFW) as pickled .bin files containing:

    bins   : list[bytes]   length 2N, JPEG-encoded 112x112 aligned crops
    issame : list[bool]    length N, ground-truth same-identity flag

These bins are the canonical evaluation format. Download URLs (mirror often):
    https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_
or any HuggingFace mirror of MS1MV3 (the bins are bundled with the train set).

We extract each pair into JPEG files on disk + emit a pairs.tsv, so the same
evaluation harness can score InsightFace-bin benchmarks and our custom
LFW-mods benchmark with identical code paths.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple


def extract_bin(bin_path: Path, out_dir: Path) -> List[Tuple[Path, Path, int]]:
    """
    Returns a list of (img_a_path, img_b_path, same?) tuples after writing
    JPEGs to ``out_dir/images/``.
    """
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    with open(bin_path, "rb") as f:
        try:
            payload = pickle.load(f, encoding="bytes")
        except Exception:
            f.seek(0)
            payload = pickle.load(f, encoding="latin1")

    # Different release tuples
    if isinstance(payload, tuple) and len(payload) == 2:
        bins, issame = payload
    elif isinstance(payload, dict):
        bins, issame = payload["bins"], payload["issame_list"]
    else:
        raise ValueError(f"unrecognised .bin payload type: {type(payload)}")

    assert len(bins) == 2 * len(issame), f"{len(bins)=}, {len(issame)=}"

    pairs: List[Tuple[Path, Path, int]] = []
    for i, same in enumerate(issame):
        a_bytes, b_bytes = bins[2 * i], bins[2 * i + 1]
        pa = images_dir / f"pair{i:05d}_a.jpg"
        pb = images_dir / f"pair{i:05d}_b.jpg"
        if not pa.exists():
            with open(pa, "wb") as f:
                f.write(a_bytes if isinstance(a_bytes, (bytes, bytearray))
                        else bytes(a_bytes))
        if not pb.exists():
            with open(pb, "wb") as f:
                f.write(b_bytes if isinstance(b_bytes, (bytes, bytearray))
                        else bytes(b_bytes))
        pairs.append((pa, pb, int(bool(same))))
    return pairs
