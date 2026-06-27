#!/bin/bash
# ============================================================================
#  MDIE — Stage 2 fan-out: train the 4 ablation variants in parallel, each on
#  its own single A100, as a SLURM array. This is the ROBUST multi-GPU path
#  (zero distributed-training code) — every task is an ordinary 1-GPU Stage-2
#  run selected with --only-variant, writing to an isolated results/figures/
#  checkpoints dir so the jobs never collide.
#
#  Submit all four at once:
#       sbatch hpc/slurm_fanout_train.sh
#  Or via the orchestrator (adds a dependency-chained merge):
#       bash hpc/submit_fanout.sh
#
#  Per-variant outputs land under research_v2/results/fanout/<variant>/ etc.
#  Merge them into one ablation table afterwards with:
#       python -m research_v2.src.merge_fanout            (see submit_fanout.sh)
# ============================================================================
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=08:00:00
#SBATCH --job-name=mdie-fan
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=dgxnp
#SBATCH --array=0-3

# shellcheck source=./_prelude.sh
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    source "$SLURM_SUBMIT_DIR/hpc/_prelude.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/_prelude.sh"
fi

VARIANTS=("MDIE-full" "MDIE-noRATA" "MDIE-noAMD" "MDIE-noICCL")
IDX="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$IDX]}"
SLUG="$(echo "$VARIANT" | tr 'A-Z-' 'a-z_')"

# Isolate each variant's outputs so the parallel jobs never overwrite each other.
export MDIE_RESULTS_DIR="$REPO_ROOT/research_v2/results/fanout/$SLUG"
export MDIE_FIGURES_DIR="$REPO_ROOT/research_v2/figures/fanout/$SLUG"
export MDIE_CKPT_DIR="$REPO_ROOT/research_v2/checkpoints/fanout/$SLUG"
mkdir -p "$MDIE_RESULTS_DIR" "$MDIE_FIGURES_DIR" "$MDIE_CKPT_DIR"
# Completion sentinel: cleared at the start of every attempt, re-created only
# when this variant finishes ALL epochs + eval. The merge guard waits on it, so
# a self-resubmitted (still-training) variant never triggers a premature merge.
rm -f "$MDIE_RESULTS_DIR/.complete"

DATASET="${DATASET:-casia}"
BACKBONE="${BACKBONE:-ir50}"
EPOCHS_S2="${EPOCHS_S2:-40}"
BATCH="${BATCH:-256}"
LR="${LR:-2e-3}"
VAL_PAIRS="${VAL_PAIRS:-6000}"
CPB="${CPB:-32}"
SPC="${SPC:-8}"
PRETRAINED="${PRETRAINED:-1}"

# --- 8 h walltime fail-safe ------------------------------------------------
# Training checkpoints every epoch (<variant>_last.pt, atomic write). We cap the
# wall-clock budget BELOW the SLURM walltime so a long run stops cleanly at an
# epoch boundary (never mid-write) and this job resubmits itself with --resume.
# Default 7 h leaves ~1 h margin under the 8 h walltime for the final epoch +
# eval. Tune with MDIE_TRAIN_MAX_SECONDS; cap the resubmit chain with
# MDIE_MAX_ATTEMPTS (default 8 -> up to ~56 h of cumulative training).
export MDIE_TRAIN_MAX_SECONDS="${MDIE_TRAIN_MAX_SECONDS:-25200}"
ATTEMPT="${MDIE_ATTEMPT:-1}"
MAX_ATTEMPTS="${MDIE_MAX_ATTEMPTS:-8}"
echo "[failsafe] attempt $ATTEMPT/$MAX_ATTEMPTS  budget=${MDIE_TRAIN_MAX_SECONDS}s  walltime=08:00:00"

# Realistic occlusion augmentation (mask/glasses/cap/structured occluder). ON by
# default for CASIA-scale training; set REALISTIC_AUG=0 to reproduce the flat
# synthetic baseline exactly.
export MDIE_REALISTIC_AUG="${REALISTIC_AUG:-1}"

echo "[fanout] task $IDX -> variant $VARIANT  (dataset=$DATASET backbone=$BACKBONE realistic_aug=$MDIE_REALISTIC_AUG)"
echo "[fanout] outputs -> $MDIE_RESULTS_DIR"

PRETRAINED_FLAG="--pretrained-backbone"
if [[ "$PRETRAINED" == "0" || "$BACKBONE" == "ir100" ]]; then
    PRETRAINED_FLAG="--no-pretrained-backbone"
fi

set +e
python -m research_v2.src.run_stage2 \
    --dataset "$DATASET" \
    --backbone "$BACKBONE" \
    $PRETRAINED_FLAG \
    --epochs "$EPOCHS_S2" \
    --batch "$BATCH" \
    --lr "$LR" \
    --workers 8 \
    --channels-last \
    --balanced-sampler \
    --classes-per-batch "$CPB" \
    --samples-per-class "$SPC" \
    --val-pairs "$VAL_PAIRS" \
    --ablation \
    --resume \
    --only-variant "$VARIANT"
RC=$?
set -e

# Exit 64 == clean stop on the wall-clock budget (not finished). Resubmit this
# one array index with --resume to continue from the last checkpoint. Any other
# non-zero code is a real failure -> do NOT resubmit (avoid an error loop).
SUBMIT_SCRIPT="${SLURM_SUBMIT_DIR:-$REPO_ROOT}/hpc/slurm_fanout_train.sh"
if [[ $RC -eq 64 ]]; then
    if [[ $ATTEMPT -lt $MAX_ATTEMPTS ]]; then
        echo "[failsafe] budget reached; resubmitting variant $VARIANT (attempt $((ATTEMPT+1))/$MAX_ATTEMPTS)"
        sbatch --array="$IDX" \
            --export=ALL,MDIE_ATTEMPT=$((ATTEMPT+1)),MDIE_MAX_ATTEMPTS=$MAX_ATTEMPTS,MDIE_TRAIN_MAX_SECONDS=$MDIE_TRAIN_MAX_SECONDS,DATASET=$DATASET,BACKBONE=$BACKBONE,EPOCHS_S2=$EPOCHS_S2,BATCH=$BATCH,LR=$LR,VAL_PAIRS=$VAL_PAIRS,CPB=$CPB,SPC=$SPC,PRETRAINED=$PRETRAINED,REALISTIC_AUG=$MDIE_REALISTIC_AUG \
            "$SUBMIT_SCRIPT"
        echo "[failsafe] resubmitted; this task exits 0 so the chain continues."
        exit 0
    fi
    echo "[failsafe] reached MAX_ATTEMPTS=$MAX_ATTEMPTS without finishing $VARIANT" >&2
    exit 1
elif [[ $RC -ne 0 ]]; then
    echo "[failsafe] run_stage2 failed with code $RC (not a budget stop); not resubmitting" >&2
    exit $RC
fi

echo "Finished $VARIANT at `date`"
touch "$MDIE_RESULTS_DIR/.complete"
