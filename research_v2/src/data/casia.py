"""
CASIA-WebFace dataset preparation and a pluggable training-source dispatcher.

CASIA is never downloaded on PARAM (no outbound internet). It must be staged
ahead of time on the laptop (extract the RecordIO to an ImageFolder layout, tar
it, scp it to PARAM under ``<cache_dir>/casia``). This module only resolves an
already-staged ImageFolder and reuses the generic ``build_face_dataset`` walker.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .lfw import build_face_dataset, prepare_lfw


def prepare_casia(cache_dir: Path) -> Path:
    """
    Resolve a staged CASIA-WebFace ImageFolder at ``<cache_dir>/casia`` and
    return the directory that holds per-identity subfolders.

    Strategy (in order); CASIA is NEVER downloaded:
      1. If ``<cache_dir>/casia/`` already has per-identity subdirs (excluding a
         nested ``casia`` dir), use it.
      2. If the tarball expanded to a nested ``<cache_dir>/casia/casia/<identity>``
         layout, reuse that nested dir without re-downloading.
      3. Otherwise raise ``FileNotFoundError`` instructing the user to stage
         CASIA first via the laptop staging helper.
    """
    out_dir = cache_dir / "casia"

    # Case 1: already in the right layout (one or more per-identity subdirs
    # that are NOT the nested 'casia/casia' from the tarball).
    if out_dir.exists():
        subs = [p for p in out_dir.iterdir() if p.is_dir() and p.name != "casia"]
        if subs:
            return out_dir

    # Case 2: re-use the tarball already extracted under a nested 'casia' dir.
    tar_root = out_dir / "casia"  # tarball expands to casia/<identity>/...
    if tar_root.exists() and any(tar_root.iterdir()):
        print(f"  [casia] using staged tarball at {tar_root}")
        return tar_root

    # Case 3: not staged; CASIA is never downloaded on PARAM.
    raise FileNotFoundError(
        f"CASIA-WebFace not found at {out_dir}. Stage it first: on the laptop "
        "extract the RecordIO to an ImageFolder layout (per-identity subfolders "
        "of .jpg/.png), tar it, and scp it to PARAM under "
        f"{out_dir} (or the nested {out_dir / 'casia'}). CASIA is never "
        "downloaded on PARAM (no outbound internet)."
    )


def build_train_dataset(
    source: str,
    cache_dir: Path,
    min_imgs: int = 3,
    max_per_id: int | None = None,
) -> Tuple[List[Path], List[int], List[str]]:
    """
    Dispatch to a training source and build (paths, labels, names).

    ``source="lfw"`` prepares LFW (min_faces_per_person=8) and walks it;
    ``source="casia"`` resolves a staged CASIA ImageFolder and walks it. The
    given ``min_imgs`` and ``max_per_id`` are passed through to
    ``build_face_dataset`` unchanged for both sources.
    """
    if source == "lfw":
        root = prepare_lfw(cache_dir, min_faces_per_person=8)
    elif source == "casia":
        root = prepare_casia(cache_dir)
    else:
        raise ValueError(
            f"unknown training source {source!r}; valid sources are: 'lfw', 'casia'"
        )
    return build_face_dataset(root, min_imgs=min_imgs, max_per_id=max_per_id)
