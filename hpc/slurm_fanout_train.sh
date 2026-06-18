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

DATASET="${DATASET:-casia}"
BACKBONE="${BACKBONE:-ir50}"
EPOCHS_S2="${EPOCHS_S2:-40}"
BATCH="${BATCH:-256}"
LR="${LR:-2e-3}"
VAL_PAIRS="${VAL_PAIRS:-6000}"
CPB="${CPB:-32}"
SPC="${SPC:-8}"
PRETRAINED="${PRETRAINED:-1}"

echo "[fanout] task $IDX -> variant $VARIANT  (dataset=$DATASET backbone=$BACKBONE)"
echo "[fanout] outputs -> $MDIE_RESULTS_DIR"

PRETRAINED_FLAG="--pretrained-backbone"
if [[ "$PRETRAINED" == "0" || "$BACKBONE" == "ir100" ]]; then
    PRETRAINED_FLAG="--no-pretrained-backbone"
fi

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
    --only-variant "$VARIANT"

echo "Finished $VARIANT at `date`"
