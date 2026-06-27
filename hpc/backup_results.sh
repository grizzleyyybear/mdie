#!/bin/bash
# ============================================================================
#  MDIE - safeguard previous results, checkpoints, figures and the landmark
#  cache BEFORE launching a new training run. Pure file-copy + checksums; runs
#  on the PARAM LOGIN NODE (no GPU, no SLURM needed):
#
#       bash hpc/backup_results.sh
#
#  It snapshots everything irreplaceable into a timestamped directory under
#  research_v2/_backups/ and writes a MANIFEST (sizes + sha256 of the small
#  files) so you can verify integrity later. The originals are NOT moved or
#  modified — this only ever copies.
#
#  Options (env vars):
#    OUT=<dir>        backup destination       (default research_v2/_backups/<ts>)
#    SKIP_CKPTS=1     record only a checkpoint MANIFEST, don't copy the .pt files
#                     (checkpoints are large; use this if you only need the
#                      results/figures/cache safeguarded and trust the .pt paths)
#    TARBALL=1        also produce a single .tar.gz of the small artifacts
#                     (results+figures+caches) for easy scp off the cluster
# ============================================================================
set -eo pipefail

# Resolve repo root from this script's location (works login-node or anywhere).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RV2="research_v2"
TS="$(date -u +%Y%m%d_%H%M%SZ)"
OUT="${OUT:-$RV2/_backups/backup_$TS}"
mkdir -p "$OUT"

echo "============================================================"
echo "[backup] repo   : $REPO_ROOT"
echo "[backup] dest   : $OUT"
echo "[backup] started: $(date -u) (UTC)"
echo "============================================================"

# --- 1. small, irreplaceable-but-cheap artifacts (always copied) -----------
#   results/  : every *.json / *.csv metric (the actual deliverables)
#   figures/  : ROC curves, overlays, paper figures
copy_tree () {  # $1 = relative source dir
    local src="$1"
    if [[ -d "$src" ]]; then
        echo "[backup] copying $src/ ..."
        mkdir -p "$OUT/$src"
        cp -a "$src/." "$OUT/$src/"
    else
        echo "[backup] (skip) $src/ does not exist"
    fi
}
copy_tree "$RV2/results"
copy_tree "$RV2/figures"

# --- 2. landmark / bone-geometry caches (expensive to rebuild) -------------
#   bone_targets*.npz = hours of GPU 68-pt detection. Per-dataset files plus any
#   legacy bone_targets.npz are all preserved.
echo "[backup] copying landmark caches ..."
mkdir -p "$OUT/$RV2/datasets_cache"
shopt -s nullglob
_caches=("$RV2"/datasets_cache/bone_targets*.npz)
if (( ${#_caches[@]} )); then
    for c in "${_caches[@]}"; do
        echo "         $(basename "$c")  ($(du -h "$c" | cut -f1))"
        cp -a "$c" "$OUT/$RV2/datasets_cache/"
    done
else
    echo "         (no bone_targets*.npz found)"
fi
shopt -u nullglob

# --- 3. trained checkpoints (large; copied unless SKIP_CKPTS=1) -------------
CKPT_DIR="$RV2/checkpoints"
if [[ -d "$CKPT_DIR" ]]; then
    echo "[backup] checkpoint tree size: $(du -sh "$CKPT_DIR" | cut -f1)"
    if [[ "${SKIP_CKPTS:-0}" == "1" ]]; then
        echo "[backup] SKIP_CKPTS=1 -> recording checkpoint manifest only (not copying)"
        ( cd "$CKPT_DIR" && find . -name '*.pt' -printf '%p\t%s bytes\n' ) \
            > "$OUT/CHECKPOINTS_MANIFEST.txt"
    else
        echo "[backup] copying $CKPT_DIR/ (use SKIP_CKPTS=1 to skip large .pt files) ..."
        mkdir -p "$OUT/$CKPT_DIR"
        cp -a "$CKPT_DIR/." "$OUT/$CKPT_DIR/"
    fi
else
    echo "[backup] (skip) $CKPT_DIR/ does not exist"
fi

# --- 4. integrity manifest (sizes + sha256 of the small files) -------------
echo "[backup] writing MANIFEST.txt (sha256 of results/figures/caches) ..."
{
    echo "# MDIE backup manifest"
    echo "# created : $(date -u) UTC"
    echo "# repo    : $REPO_ROOT"
    echo "# git HEAD: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
    echo
    echo "## disk usage"
    du -sh "$OUT" 2>/dev/null || true
    echo
    echo "## sha256 (results, figures, caches)"
    find "$OUT/$RV2/results" "$OUT/$RV2/figures" "$OUT/$RV2/datasets_cache" \
         -type f \( -name '*.json' -o -name '*.csv' -o -name '*.npz' \) 2>/dev/null \
        | sort | while read -r f; do sha256sum "$f"; done
} > "$OUT/MANIFEST.txt"

# --- 5. optional tarball of the small artifacts for off-cluster scp ---------
if [[ "${TARBALL:-0}" == "1" ]]; then
    TAR="$OUT.small.tar.gz"
    echo "[backup] building $TAR (results + figures + caches) ..."
    tar -czf "$TAR" \
        -C "$OUT" "$RV2/results" "$RV2/figures" "$RV2/datasets_cache" "MANIFEST.txt" \
        2>/dev/null || true
    echo "[backup] tarball: $TAR  ($(du -h "$TAR" 2>/dev/null | cut -f1))"
fi

echo "============================================================"
echo "[backup] DONE -> $OUT"
echo "[backup] verify : cat $OUT/MANIFEST.txt"
echo "[backup] restore: cp -a $OUT/$RV2/<results|figures|checkpoints>/. $RV2/<...>/"
echo "[backup] off-cluster (run on your laptop):"
echo "         scp -r <user>@login.npsf.cdac.in:$REPO_ROOT/$OUT ./mdie_backup_$TS"
echo "============================================================"
