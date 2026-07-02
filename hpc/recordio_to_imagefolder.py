#!/usr/bin/env python3
"""
Extract an InsightFace MXNet **RecordIO** dataset (``train.rec`` + ``train.idx``)
to a plain per-identity **ImageFolder** — using a PURE-PYTHON RecordIO reader so
no ``mxnet`` is required (mxnet is painful to install and we want zero extra deps
on PARAM).

The on-disk format is dmlc-core RecordIO:
    magic  : uint32  = 0xced7230a
    lrec   : uint32  -> cflag = lrec>>29 & 7 ; length = lrec & (2**29-1)
    data   : <length> bytes ; padded to a 4-byte boundary (we seek by .idx, so
             padding is irrelevant).
Each record's ``data`` is an IRecordIOHeader ('<IfQQ' = flag, label, id, id2)
followed by the payload. When ``flag > 0`` the label is an array of ``flag``
float32 values placed at the START of the payload (InsightFace stores the
(start,end) sample range in record 0 this way; per-image records use flag==0
with a single float identity label).

Usage:
    python hpc/recordio_to_imagefolder.py <rec_dir> <out_dir>
    #   <rec_dir> must contain train.rec and train.idx
    #   <out_dir>/<0000000>/<00000001>.jpg ...   (ImageFolder the loader reads)
"""
from __future__ import annotations

import struct
import sys
import os
from pathlib import Path

MAGIC = 0xCED7230A
IR_FORMAT = "<IfQQ"               # flag(uint32) label(float32) id(uint64) id2(uint64)
IR_SIZE = struct.calcsize(IR_FORMAT)  # 24


def load_idx(idx_path: Path) -> dict[int, int]:
    """Parse train.idx -> {record_index: byte_offset_in_rec}."""
    table: dict[int, int] = {}
    with open(idx_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            table[int(parts[0])] = int(parts[-1])
    return table


def read_record(rec, offset: int) -> bytes:
    """Read one record's raw ``data`` blob at ``offset`` in the open .rec file."""
    rec.seek(offset)
    magic = struct.unpack("<I", rec.read(4))[0]
    if magic != MAGIC:
        raise ValueError(f"bad RecordIO magic {magic:#x} at offset {offset}")
    lrec = struct.unpack("<I", rec.read(4))[0]
    length = lrec & ((1 << 29) - 1)
    return rec.read(length)


def unpack(data: bytes):
    """Return (flag, label, payload). label is a float (flag==0) or tuple."""
    flag, label0, _id1, _id2 = struct.unpack(IR_FORMAT, data[:IR_SIZE])
    body = data[IR_SIZE:]
    if flag > 0:
        label = struct.unpack("<%df" % flag, body[: 4 * flag])
        body = body[4 * flag:]
    else:
        label = label0
    return flag, label, body


def _label_to_int(label) -> int:
    if isinstance(label, (tuple, list)):
        return int(label[0])
    return int(label)


def _img_ext(body: bytes) -> str:
    """Guess the file extension from the encoded-image magic bytes."""
    if body[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if body[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return ".jpg"


def _is_image(body: bytes) -> bool:
    """True if body starts with a JPEG or PNG magic signature."""
    return body[:3] == b"\xff\xd8\xff" or body[:8] == b"\x89PNG\r\n\x1a\n"


def extract(rec_dir: Path, out_dir: Path) -> int:
    idx = load_idx(rec_dir / "train.idx")
    if not idx:
        raise SystemExit(f"empty/missing index at {rec_dir / 'train.idx'}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # InsightFace packs ~490k image records (flag==0, single float identity
    # label, JPEG body) followed by per-identity annotation records (flag>0,
    # empty body). Record 0's label points at where the annotation records
    # begin, NOT the image range — so rather than rely on that convention we
    # iterate every record and keep only those whose body is a real image.
    # Annotation/header records have non-image bodies and are skipped.
    n = 0
    reused = 0
    skipped = 0
    with open(rec_dir / "train.rec", "rb") as rec:
        for i in sorted(idx.keys()):
            off = idx[i]
            try:
                _flag, label, body = unpack(read_record(rec, off))
            except Exception:
                skipped += 1
                continue
            if not _is_image(body):
                skipped += 1
                continue
            d = out_dir / f"{_label_to_int(label):07d}"
            d.mkdir(exist_ok=True)
            # Write the original encoded bytes verbatim (lossless + fast — no
            # decode/re-encode round-trip on ~490k images).
            out_path = d / f"{i:08d}{_img_ext(body)}"
            if out_path.exists() and out_path.stat().st_size == len(body):
                reused += 1
            else:
                with open(out_path, "wb") as out:
                    out.write(body)
            n += 1
            if n % 50000 == 0:
                msg = f"  [recordio] processed {n} images"
                if reused:
                    msg += f" ({reused} already present/resumed)"
                print(msg + " ...", flush=True)
    if skipped:
        print(f"  [recordio] skipped {skipped} non-image records (headers/annotations)")
    if reused:
        print(f"  [recordio] reused {reused} existing complete images")
    return n


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: recordio_to_imagefolder.py <rec_dir> <out_dir>")
    if not os.environ.get("SLURM_JOB_ID") and os.environ.get("MDIE_ALLOW_LOGIN_EXTRACT") != "1":
        raise SystemExit(
            "Refusing to run heavy RecordIO extraction outside SLURM. "
            "Use: sbatch hpc/slurm_stage_trainset.sh (or run "
            "bash hpc/fetch_trainset.sh <dataset>, which submits it). "
            "For a small local developer test only, set MDIE_ALLOW_LOGIN_EXTRACT=1."
        )
    rec_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    if not (rec_dir / "train.rec").is_file() or not (rec_dir / "train.idx").is_file():
        raise SystemExit(f"{rec_dir} must contain train.rec and train.idx")
    n = extract(rec_dir, out_dir)
    n_ids = sum(1 for p in out_dir.iterdir() if p.is_dir())
    print(f"  [recordio] extracted {n} images across {n_ids} identities -> {out_dir}")
    if n < 1000:
        raise SystemExit(f"only {n} images extracted (expected ~490k for CASIA)")


if __name__ == "__main__":
    main()
