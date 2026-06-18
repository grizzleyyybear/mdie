#!/bin/bash
# ============================================================================
#  MDIE - eval-only: real benchmarks + interpretability + deployment proof.
#  Inference-only. Walltime: < 1.5 h on 1 A100-SXM4.
#
#  Submit:  sbatch hpc/slurm_eval.sh
# ============================================================================
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=02:00:00
#SBATCH --job-name=mdie-eval
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=dgxnp

# shellcheck source=./_prelude.sh
# Under SLURM the batch script is copied to a private spool dir, so $0 no longer
# sits next to _prelude.sh; resolve it via SLURM_SUBMIT_DIR (the dir sbatch was
# launched from = repo root). Fall back to this script's own dir when run
# directly without SLURM.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    source "$SLURM_SUBMIT_DIR/hpc/_prelude.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/_prelude.sh"
fi

BENCHMARKS="${BENCHMARKS:-mfr2 meglass calfw agedb30}"

# Auto-include staged niche occlusion/lighting benchmarks if their data is
# present under datasets_cache (or RMFRD_ROOT/ARFACE_ROOT/YALEB_ROOT are set).
_DC="$REPO_ROOT/research_v2/datasets_cache/benchmarks"
if [[ -n "${RMFRD_ROOT:-}" || -d "$_DC/rmfrd/AFDB_masked_face_dataset" || -d "$_DC/rmfrd" && -n "$(find "$_DC/rmfrd" -maxdepth 2 -iname '*.jpg' -print -quit 2>/dev/null)" ]]; then
    export RMFRD_ROOT="${RMFRD_ROOT:-$_DC/rmfrd}"
    BENCHMARKS="$BENCHMARKS rmfrd"
    echo "[eval] including RMFRD (real masked faces) from $RMFRD_ROOT"
fi
if [[ -n "${ARFACE_ROOT:-}" ]]; then BENCHMARKS="$BENCHMARKS arface"; echo "[eval] including AR Face"; fi
if [[ -n "${YALEB_ROOT:-}" ]]; then BENCHMARKS="$BENCHMARKS yaleb"; echo "[eval] including Yale-B"; fi

# Real worn-occlusion + lighting transfer (MFR2 masks, MeGlass glasses) + aging.
# shellcheck disable=SC2086
python -m research_v2.src.eval.run_real_benchmarks \
    --models mdie arcface cosface mobilefacenet \
    --benchmarks $BENCHMARKS

# Bone-anchored attention interpretability (matched/mismatched/random IoU).
python research_v2/scripts/attention_bone_iou.py 60 || true

# ArcFace-compatibility deployment proof (single 512-d, cosine==dot).
python research_v2/scripts/inference_compat_proof.py || true

echo "Finished at `date`"
