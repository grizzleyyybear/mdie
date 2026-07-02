#!/usr/bin/env python3
"""
Extract a **WebDataset** face-recognition dataset (a directory of ``*.tar`` or
``*.tar.gz`` shards) to a plain per-identity **ImageFolder**, using only the
Python standard library (``tarfile`` + ``gzip``) so no ``webdataset`` dependency
is needed on PARAM.

WebDataset convention (gaunernst/glint360k-wds-gz, gaunernst/ms1mv3-wds[-gz]):
each sample is two members sharing a key::

    <key>.jpg   encoded JPEG image bytes
    <key>.cls   the integer identity label, as ASCII text (e.g. b"42")

We stream every shard, pair ``.jpg`` with its ``.cls``, and write::

    <out_dir>/<0000042>/<key>.jpg

which is exactly the per-identity ImageFolder the trainer reads. Image bytes are
written verbatim (no decode/re-encode round-trip).

Usage:
    python hpc/webdataset_to_imagefolder.py <shard_dir> <out_dir>
    #   <shard_dir> contains glint360k-0000.tar.gz ... (or *.tar)
"""
from __future__ import annotations

import gzip
import io
import os
import sys
import tarfile
from pathlib import Path


def _iter_shards(shard_dir: Path):
    """Yield shard paths in stable order (*.tar and *.tar.gz)."""
    shards = sorted(
        [p for p in shard_dir.iterdir()
         if p.is_file() and (p.name.endswith(".tar") or p.name.endswith(".tar.gz"))]
    )
    return shards


def _open_tar(path: Path) -> tarfile.TarFile:
    """Open a .tar or .tar.gz shard as a streaming TarFile.

    We decompress .gz into a BytesIO so the tar stream is seekable-enough for
    tarfile's sequential read; shards are ~90 MB so this is memory-cheap.
    """
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as gz:
            return tarfile.open(fileobj=io.BytesIO(gz.read()), mode="r:")
    return tarfile.open(path, mode="r:")


def _img_ext(body: bytes) -> str:
    if body[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if body[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return ".jpg"


def _is_image(body: bytes) -> bool:
    return body[:3] == b"\xff\xd8\xff" or body[:8] == b"\x89PNG\r\n\x1a\n"


def _parse_cls(raw: bytes) -> int | None:
    """Parse the .cls payload into an integer identity label."""
    try:
        return int(raw.decode("ascii").strip())
    except (ValueError, UnicodeDecodeError):
        pass
    # Fallback: some writers store the label as a little-endian int32.
    if len(raw) >= 4:
        import struct
        try:
            return int(struct.unpack("<i", raw[:4])[0])
        except struct.error:
            return None
    return None


def extract(shard_dir: Path, out_dir: Path) -> int:
    shards = _iter_shards(shard_dir)
    if not shards:
        raise SystemExit(f"no *.tar / *.tar.gz shards found under {shard_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [wds] {len(shards)} shards -> {out_dir}", flush=True)

    n = 0
    reused = 0
    skipped = 0
    for si, shard in enumerate(shards):
        # Buffer one shard's samples keyed by sample key: {key: {"jpg":..,"cls":..}}.
        pending: dict[str, dict[str, bytes]] = {}
        try:
            tf = _open_tar(shard)
        except Exception as e:  # noqa: BLE001
            print(f"  [wds] WARN: cannot open {shard.name}: {e}", flush=True)
            continue
        with tf:
            for m in tf:
                if not m.isfile():
                    continue
                name = m.name
                dot = name.rfind(".")
                if dot < 0:
                    continue
                key, ext = name[:dot], name[dot + 1:].lower()
                if ext not in ("jpg", "jpeg", "png", "cls"):
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                blob = f.read()
                slot = pending.setdefault(key, {})
                slot["cls" if ext == "cls" else "img"] = blob
                pair = pending.get(key)
                if pair and "img" in pair and "cls" in pair:
                    body = pair["img"]
                    cls = _parse_cls(pair["cls"])
                    del pending[key]
                    if cls is None or not _is_image(body):
                        skipped += 1
                        continue
                    d = out_dir / f"{cls:07d}"
                    d.mkdir(exist_ok=True)
                    safe_key = key.replace("/", "_")
                    out_path = d / f"{safe_key}{_img_ext(body)}"
                    if out_path.exists() and out_path.stat().st_size == len(body):
                        reused += 1
                    else:
                        with open(out_path, "wb") as out:
                            out.write(body)
                    n += 1
                    if n % 100000 == 0:
                        msg = f"  [wds] processed {n} images "
                        msg += f"(shard {si + 1}/{len(shards)})"
                        if reused:
                            msg += f" ({reused} already present/resumed)"
                        print(msg + " ...", flush=True)
        # Any unmatched leftovers in this shard are incomplete samples.
        skipped += len(pending)
    if skipped:
        print(f"  [wds] skipped {skipped} incomplete/non-image samples", flush=True)
    if reused:
        print(f"  [wds] reused {reused} existing complete images", flush=True)
    return n


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: webdataset_to_imagefolder.py <shard_dir> <out_dir>")
    if not os.environ.get("SLURM_JOB_ID") and os.environ.get("MDIE_ALLOW_LOGIN_EXTRACT") != "1":
        raise SystemExit(
            "Refusing to run heavy WebDataset extraction outside SLURM. "
            "Use: sbatch hpc/slurm_stage_trainset.sh (or run "
            "bash hpc/fetch_trainset.sh <dataset>, which submits it). "
            "For a small local developer test only, set MDIE_ALLOW_LOGIN_EXTRACT=1."
        )
    shard_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    n = extract(shard_dir, out_dir)
    n_ids = sum(1 for p in out_dir.iterdir() if p.is_dir())
    print(f"  [wds] extracted {n} images across {n_ids} identities -> {out_dir}")
    if n < 1000:
        raise SystemExit(f"only {n} images extracted — check the shard format")


if __name__ == "__main__":
    main()
