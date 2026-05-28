#!/bin/bash
# ============================================================================
#  MDIE — Stage 2: MDIE-full + ablation (-AMD, -RATA).
#  Expected walltime: 5-6 h on 1 A100-SXM4.
#
#  Submit:  sbatch hpc/slurm_stage2.sh
# ============================================================================
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=08:00:00
#SBATCH --job-name=mdie-s2
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=dgxnp

# shellcheck source=./_prelude.sh
source "$(dirname "$0")/_prelude.sh"

EPOCHS_S2="${EPOCHS_S2:-60}"
BATCH="${BATCH:-256}"
LR="${LR:-2e-3}"
VAL_PAIRS="${VAL_PAIRS:-3000}"
CPB="${CPB:-32}"
SPC="${SPC:-8}"

python -m research_v2.src.run_stage2 \
    --epochs "$EPOCHS_S2" \
    --batch "$BATCH" \
    --lr "$LR" \
    --workers 8 \
    --channels-last \
    --balanced-sampler \
    --classes-per-batch "$CPB" \
    --samples-per-class "$SPC" \
    --val-pairs "$VAL_PAIRS" \
    --ablation

echo "Finished at `date`"
