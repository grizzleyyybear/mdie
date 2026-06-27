#!/bin/bash
# ============================================================================
#  MDIE — one-off bone-landmark cache builder (BATCH job, no interactive login).
#
#  Builds research_v2/datasets_cache/bone_targets.npz for the chosen dataset on
#  a single A100, then exits. This is the dominant one-off cost of the CASIA
#  scale-up. Run it ONCE before the fan-out so the 4 variant jobs find the cache
#  already present (missing=[]) and skip straight to training — instead of all
#  four racing to build the same .npz.
#
#  Submit (batch only — no srun/interactive shell needed):
#       DATASET=casia sbatch hpc/slurm_build_cache.sh
#
#  Then watch:
#       squeue --me
#       ls -l research_v2/datasets_cache/bone_targets.npz     # appears when done
#
#  Once the .npz exists, launch the fan-out:
#       DATASET=casia bash hpc/submit_fanout.sh
# ============================================================================
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=02:00:00
#SBATCH --job-name=mdie-cache
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=dgxnp

# shellcheck source=./_prelude.sh
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    source "$SLURM_SUBMIT_DIR/hpc/_prelude.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/_prelude.sh"
fi

DATASET="${DATASET:-casia}"
BACKBONE="${BACKBONE:-ir50}"
BATCH="${BATCH:-256}"

PRETRAINED="${PRETRAINED:-1}"
PRETRAINED_FLAG="--pretrained-backbone"
if [[ "$PRETRAINED" == "0" || "$BACKBONE" == "ir100" ]]; then
    PRETRAINED_FLAG="--no-pretrained-backbone"
fi

echo "[cache] building bone-landmark cache for dataset=$DATASET (GPU landmarking)"

python -m research_v2.src.run_stage2 \
    --dataset "$DATASET" \
    --backbone "$BACKBONE" \
    $PRETRAINED_FLAG \
    --batch "$BATCH" \
    --ablation \
    --only-variant MDIE-full \
    --build-cache-only

echo "[cache] done at `date`"
ls -l research_v2/datasets_cache/bone_targets.npz 2>/dev/null || \
    echo "[cache] WARNING: bone_targets.npz not found — check the log above." >&2
