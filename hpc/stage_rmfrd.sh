#!/bin/bash
# ============================================================================
#  Stage the Real-World Masked Face Recognition Dataset (RMFRD / RMFD) for MDIE
#  evaluation. Run on the LAPTOP, then scp the tarball to PARAM (no internet on
#  compute nodes). This is an EVAL-only dataset (real worn masks) used to prove
#  that MDIE's synthetic-occlusion training transfers to real masked faces.
#
#  RMFRD ships as two parallel identity trees:
#       AFDB_masked_face_dataset/<identity>/<img>.jpg     (real masks)
#       AFDB_face_dataset/<identity>/<img>.jpg            (unmasked)
#  The loader pairs masked<->unmasked of the same identity (the hard case).
#
#  Usage:
#       # A) you have the extracted RMFRD root (contains the two AFDB_* dirs):
#       RMFRD_SRC=/path/to/RMFRD  bash hpc/stage_rmfrd.sh
#
#       # B) you have a .zip of it:
#       RMFRD_SRC=/path/to/RMFRD.zip  bash hpc/stage_rmfrd.sh
#
#  Output:
#       research_v2/datasets_cache/benchmarks/rmfrd/<two AFDB_* trees>
#       research_v2/datasets_cache/rmfrd.tar      (scp this up)
#
#  Then on PARAM:
#       scp research_v2/datasets_cache/rmfrd.tar \
#           bhatib@login.npsf.cdac.in:/nlsasfs/home/csaml/bhatib/mrinal/projects/mdie/research_v2/datasets_cache/
#       cd research_v2/datasets_cache && tar -xf rmfrd.tar
#       # in your eval sbatch: export RMFRD_ROOT=$PWD/research_v2/datasets_cache/benchmarks/rmfrd
# ============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE_ROOT="$REPO_ROOT/research_v2/datasets_cache"
OUT_DIR="$CACHE_ROOT/benchmarks/rmfrd"
mkdir -p "$CACHE_ROOT/benchmarks"

RMFRD_SRC="${RMFRD_SRC:-}"
if [[ -z "$RMFRD_SRC" ]]; then
    echo "[rmfrd] ERROR: set RMFRD_SRC to the extracted RMFRD root or a .zip of it."
    echo "        The root should contain AFDB_masked_face_dataset/ and AFDB_face_dataset/."
    echo "        See the header of this script for examples."
    exit 1
fi

_count_imgs() { find "$1" -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' \) 2>/dev/null | wc -l; }
_has_two_trees() {
    [[ -n "$(find "$1" -maxdepth 3 -type d -name 'AFDB_masked_face_dataset' -print -quit 2>/dev/null)" ]] && \
    [[ -n "$(find "$1" -maxdepth 3 -type d -name 'AFDB_face_dataset' -print -quit 2>/dev/null)" ]]
}

# ---- Resolve RMFRD_SRC into $OUT_DIR --------------------------------------
if [[ -f "$RMFRD_SRC" && "$RMFRD_SRC" == *.zip ]]; then
    echo "[rmfrd] zip detected; unzipping ..."
    tmp="$CACHE_ROOT/_rmfrd_unzip"
    rm -rf "$tmp"; mkdir -p "$tmp"
    unzip -q "$RMFRD_SRC" -d "$tmp"
    # locate the dir that holds the two AFDB_* trees
    src_root="$tmp"
    found="$(find "$tmp" -maxdepth 3 -type d -name 'AFDB_masked_face_dataset' -print -quit 2>/dev/null)"
    [[ -n "$found" ]] && src_root="$(dirname "$found")"
    mkdir -p "$OUT_DIR"
    cp -r "$src_root"/. "$OUT_DIR"/
    rm -rf "$tmp"

elif [[ -d "$RMFRD_SRC" ]]; then
    echo "[rmfrd] directory detected at $RMFRD_SRC"
    src_root="$RMFRD_SRC"
    found="$(find "$RMFRD_SRC" -maxdepth 3 -type d -name 'AFDB_masked_face_dataset' -print -quit 2>/dev/null)"
    [[ -n "$found" ]] && src_root="$(dirname "$found")"
    mkdir -p "$OUT_DIR"
    if [[ "$(cd "$src_root" && pwd)" != "$(cd "$OUT_DIR" 2>/dev/null && pwd || echo /nope)" ]]; then
        cp -r "$src_root"/. "$OUT_DIR"/
    fi

else
    echo "[rmfrd] ERROR: RMFRD_SRC '$RMFRD_SRC' is not a directory or .zip"
    exit 1
fi

# ---- Validate + tar -------------------------------------------------------
n_imgs="$(_count_imgs "$OUT_DIR")"
echo "[rmfrd] staged $n_imgs images at $OUT_DIR"
if [[ "$n_imgs" -lt 100 ]]; then
    echo "[rmfrd] ERROR: only $n_imgs images found (expected thousands)."
    exit 1
fi
if _has_two_trees "$OUT_DIR"; then
    echo "[rmfrd] OK: masked + unmasked AFDB trees present (hard masked<->unmasked pairs)."
else
    echo "[rmfrd] NOTE: AFDB_masked/AFDB_face trees not both found; the loader will"
    echo "        fall back to flat per-identity pairing. That still works, but the"
    echo "        masked<->unmasked hard pairing needs both trees."
fi

# Sanity: confirm the project loader can build pairs from the staged dir.
echo "[rmfrd] verifying with the project loader ..."
RMFRD_ROOT="$OUT_DIR" PYTHONPATH="$REPO_ROOT" python - <<'PY'
import os
from research_v2.src.data.benchmarks import load_benchmark
b = load_benchmark("rmfrd")
print(f"  [rmfrd] loader built {len(b.pairs)} pairs")
PY

TARBALL="$CACHE_ROOT/rmfrd.tar"
echo "[rmfrd] creating tarball $TARBALL ..."
tar -C "$CACHE_ROOT/benchmarks" -cf "$TARBALL" rmfrd
echo "[rmfrd] OK -> $TARBALL ($(du -sh "$TARBALL" | cut -f1))"

cat <<MSG

[next] Copy to PARAM:
   scp "$TARBALL" \\
       bhatib@login.npsf.cdac.in:/nlsasfs/home/csaml/bhatib/mrinal/projects/mdie/research_v2/datasets_cache/
   # on PARAM:
   cd research_v2/datasets_cache && mkdir -p benchmarks && tar -C benchmarks -xf rmfrd.tar
   # then in your eval sbatch:
   export RMFRD_ROOT=\$PWD/research_v2/datasets_cache/benchmarks/rmfrd
MSG
