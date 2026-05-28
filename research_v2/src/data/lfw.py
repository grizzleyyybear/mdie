"""
LFW dataset preparation.

Uses scikit-learn's downloader which has multiple working mirrors. Builds an
identity-indexed directory layout compatible with the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def prepare_lfw(cache_dir: Path, min_faces_per_person: int = 5) -> Path:
    """
    Make LFW available at ``<cache_dir>/lfw/<identity>/<image>.jpg`` and return
    that directory.

    Strategy (in order):
      1. If ``<cache_dir>/lfw/`` already has per-identity subdirs, use it.
      2. If ``hpc/stage_datasets.sh`` has placed the canonical UMass tarball
         layout at ``<cache_dir>/lfw/lfw/<identity>/<image>.jpg`` (the tar
         creates a nested ``lfw`` dir), reuse it without re-downloading.
      3. Fall back to sklearn's ``fetch_lfw_people`` (only path that needs
         outbound HTTP; flaky on CDAC because figshare blocks the proxy).
    """
    out_dir = cache_dir / "lfw"

    # Case 1: already in the right layout (one or more per-identity subdirs
    # that are NOT the nested 'lfw/lfw' from the tarball).
    if out_dir.exists():
        subs = [p for p in out_dir.iterdir() if p.is_dir() and p.name != "lfw"]
        if subs:
            return out_dir

    # Case 2: re-use the UMass tarball already extracted by stage_datasets.sh.
    tar_root = out_dir / "lfw"  # tarball expands to lfw/<identity>/...
    if tar_root.exists() and any(tar_root.iterdir()):
        print(f"  [lfw] using staged UMass tarball at {tar_root}")
        return tar_root

    # Case 3: sklearn download (only if neither of the above worked).
    from sklearn.datasets import fetch_lfw_people

    print(f"  [lfw] downloading via sklearn (min_faces_per_person={min_faces_per_person}) ...")
    bunch = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        resize=1.0,
        color=True,
        funneled=True,
        data_home=str(cache_dir / "_sklearn_cache"),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict = {}
    for img, label in zip(bunch.images, bunch.target):
        name = bunch.target_names[label].replace(" ", "_")
        person_dir = out_dir / name
        person_dir.mkdir(exist_ok=True)
        idx = counts.get(name, 0)
        counts[name] = idx + 1
        arr = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        Image.fromarray(arr).save(person_dir / f"{name}_{idx:04d}.jpg", quality=95)
    print(f"  [lfw] wrote {sum(counts.values())} images for {len(counts)} identities to {out_dir}")
    return out_dir


def build_face_dataset(
    root: Path, min_imgs: int = 3, max_per_id: int | None = None
) -> Tuple[List[Path], List[int], List[str]]:
    """
    Walk ``root`` (per-identity subfolders) and build (paths, labels, names).
    Identities with fewer than ``min_imgs`` are dropped.
    """
    paths: List[Path] = []
    labels: List[int] = []
    names: List[str] = []
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.png"))
        if len(imgs) < min_imgs:
            continue
        if max_per_id:
            imgs = imgs[:max_per_id]
        label = len(names)
        names.append(person_dir.name)
        for p in imgs:
            paths.append(p)
            labels.append(label)
    return paths, labels, names
