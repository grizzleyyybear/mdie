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


def extract(rec_dir: Path, out_dir: Path) -> int:
    idx = load_idx(rec_dir / "train.idx")
    if not idx:
        raise SystemExit(f"empty/missing index at {rec_dir / 'train.idx'}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(rec_dir / "train.rec", "rb") as rec:
        keys = sorted(idx.keys())
        # Record 0 usually holds the (start, end) sample range as a 2-float label.
        sample_indices = None
        if 0 in idx:
            try:
                _flag0, label0, _ = unpack(read_record(rec, idx[0]))
                if isinstance(label0, tuple) and len(label0) >= 2 and label0[1] > label0[0]:
                    sample_indices = range(int(label0[0]), int(label0[1]))
            except Exception:
                sample_indices = None
        if sample_indices is None:
            sample_indices = [k for k in keys if k != 0]

        n = 0
        for i in sample_indices:
            off = idx.get(i)
            if off is None:
                continue
            try:
                _flag, label, body = unpack(read_record(rec, off))
            except Exception:
                continue
            if not body:
                continue
            d = out_dir / f"{_label_to_int(label):07d}"
            d.mkdir(exist_ok=True)
            # Write the original encoded bytes verbatim (lossless + fast — no
            # decode/re-encode round-trip on ~490k images).
            with open(d / f"{i:08d}{_img_ext(body)}", "wb") as out:
                out.write(body)
            n += 1
            if n % 50000 == 0:
                print(f"  [recordio] wrote {n} images ...", flush=True)
    return n


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: recordio_to_imagefolder.py <rec_dir> <out_dir>")
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
