#!/bin/bash
# ============================================================================
#  Stage CASIA-WebFace as a plain ImageFolder for MDIE scale-up training.
#
#  Run on the LAPTOP (or any internet-capable box) — NOT on a PARAM compute
#  node. PARAM has no outbound internet and we deliberately avoid needing
#  `mxnet` there, so this script does the RecordIO -> ImageFolder extraction
#  locally and produces a tarball you scp to PARAM.
#
#  Usage:
#       # A) you already have an extracted ImageFolder (per-identity subdirs):
#       CASIA_SRC=/path/to/CASIA-WebFace  bash hpc/stage_casia.sh
#
#       # B) you have the InsightFace RecordIO (train.rec/train.idx/property):
#       CASIA_SRC=/path/to/faces_webface_112x112  bash hpc/stage_casia.sh
#       #   (requires `pip install mxnet` on THIS machine only)
#
#       # C) you have a .zip of an ImageFolder:
#       CASIA_SRC=/path/to/CASIA-WebFace.zip  bash hpc/stage_casia.sh
#
#  Output:
#       research_v2/datasets_cache/casia/<identity>/<img>.jpg   (ImageFolder)
#       research_v2/datasets_cache/casia.tar                    (scp this up)
#
#  Then copy to PARAM (compute nodes have no internet):
#       scp research_v2/datasets_cache/casia.tar \
#           bhatib@login.npsf.cdac.in:/nlsasfs/home/csaml/bhatib/mrinal/projects/mdie/research_v2/datasets_cache/
#       # on PARAM login node:
#       cd research_v2/datasets_cache && tar -xf casia.tar
#  The resulting research_v2/datasets_cache/casia/ is what --dataset casia reads.
# ============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE_ROOT="$REPO_ROOT/research_v2/datasets_cache"
OUT_DIR="$CACHE_ROOT/casia"
mkdir -p "$CACHE_ROOT"

CASIA_SRC="${CASIA_SRC:-}"
if [[ -z "$CASIA_SRC" ]]; then
    echo "[casia] ERROR: set CASIA_SRC to one of:"
    echo "         - an extracted ImageFolder (per-identity subdirs)"
    echo "         - an InsightFace RecordIO dir (train.rec/train.idx/property)"
    echo "         - a .zip of an ImageFolder"
    echo "       See the header of this script for examples."
    exit 1
fi

_count_imgs() { find "$1" -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' \) 2>/dev/null | wc -l; }

# ---- Resolve CASIA_SRC into an ImageFolder at $OUT_DIR ---------------------
if [[ -d "$CASIA_SRC" && -f "$CASIA_SRC/train.rec" ]]; then
    echo "[casia] RecordIO detected at $CASIA_SRC; extracting to ImageFolder ..."
    if ! python -c "import mxnet" 2>/dev/null; then
        echo "[casia] ERROR: reading RecordIO needs mxnet on THIS machine."
        echo "       Install it locally (NOT on PARAM):  pip install mxnet"
        exit 1
    fi
    mkdir -p "$OUT_DIR"
    python - "$CASIA_SRC" "$OUT_DIR" <<'PY'
import sys, os
from pathlib import Path
import mxnet as mx
import numpy as np
import cv2

rec_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
imgrec = mx.recordio.MXIndexedRecordIO(
    str(rec_dir / "train.idx"), str(rec_dir / "train.rec"), "r")
# header0 holds the (start, end) index range of real samples.
s = imgrec.read_idx(0)
header, _ = mx.recordio.unpack(s)
start, end = int(header.label[0]), int(header.label[1])
n = 0
for idx in range(start, end):
    s = imgrec.read_idx(idx)
    header, img = mx.recordio.unpack(s)
    label = header.label
    label = int(label if not hasattr(label, "__len__") else label[0])
    img = mx.image.imdecode(img).asnumpy()  # RGB
    d = out_dir / f"{label:07d}"
    d.mkdir(exist_ok=True)
    cv2.imwrite(str(d / f"{idx:08d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    n += 1
    if n % 50000 == 0:
        print(f"  [casia] wrote {n} images ...", flush=True)
print(f"  [casia] extracted {n} images into {out_dir}")
PY

elif [[ -d "$CASIA_SRC" ]]; then
    echo "[casia] ImageFolder detected at $CASIA_SRC"
    if [[ "$(cd "$CASIA_SRC" && pwd)" != "$(cd "$OUT_DIR" 2>/dev/null && pwd || echo /nope)" ]]; then
        mkdir -p "$OUT_DIR"
        # Copy (don't move) so the user's source is preserved.
        cp -r "$CASIA_SRC"/. "$OUT_DIR"/
    fi

elif [[ -f "$CASIA_SRC" && "$CASIA_SRC" == *.zip ]]; then
    echo "[casia] zip detected at $CASIA_SRC; unzipping ..."
    tmp="$CACHE_ROOT/_casia_unzip"
    rm -rf "$tmp"; mkdir -p "$tmp"
    unzip -q "$CASIA_SRC" -d "$tmp"
    # Find the dir that actually holds per-identity subfolders of images.
    root="$(find "$tmp" -type d -name '*' | while read -r d; do
        if [[ -n "$(find "$d" -maxdepth 2 -iname '*.jpg' -print -quit 2>/dev/null)" ]]; then echo "$d"; break; fi
    done)"
    root="${root:-$tmp}"
    mkdir -p "$OUT_DIR"
    cp -r "$root"/. "$OUT_DIR"/
    rm -rf "$tmp"

else
    echo "[casia] ERROR: CASIA_SRC '$CASIA_SRC' is not a dir or .zip"
    exit 1
fi

# ---- Validate + tar -------------------------------------------------------
n_imgs="$(_count_imgs "$OUT_DIR")"
n_ids="$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "[casia] staged $n_imgs images across $n_ids identities at $OUT_DIR"
if [[ "$n_imgs" -lt 1000 ]]; then
    echo "[casia] ERROR: only $n_imgs images found (expected ~494k for full CASIA)."
    exit 1
fi

TARBALL="$CACHE_ROOT/casia.tar"
echo "[casia] creating tarball $TARBALL (this can take a few minutes) ..."
tar -C "$CACHE_ROOT" -cf "$TARBALL" casia
echo "[casia] OK -> $TARBALL ($(du -sh "$TARBALL" | cut -f1))"

cat <<MSG

[next] Copy to PARAM (compute nodes have no internet):
   scp "$TARBALL" \\
       bhatib@login.npsf.cdac.in:/nlsasfs/home/csaml/bhatib/mrinal/projects/mdie/research_v2/datasets_cache/
   # then on the PARAM login node:
   cd research_v2/datasets_cache && tar -xf casia.tar
   # train with:  sbatch hpc/slurm_ddp_full.sh   (DATASET=casia)
MSG
