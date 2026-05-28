#!/bin/bash
# ============================================================================
#  MDIE — Stage 1 baselines (FaceNet, ArcFace, CosFace, MobileFaceNet).
#  Expected walltime: 3-4 h on 1 A100-SXM4.
#
#  Submit:  sbatch hpc/slurm_stage1.sh
# ============================================================================
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=06:00:00
#SBATCH --job-name=mdie-s1
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
##SBATCH --partition=nltmp    # uncomment if you have a project partition (advisory 1)

# shellcheck source=./_prelude.sh
source "$(dirname "$0")/_prelude.sh"

EPOCHS_S1="${EPOCHS_S1:-50}"
BATCH="${BATCH:-256}"
LR="${LR:-2e-3}"
VAL_PAIRS="${VAL_PAIRS:-3000}"

python -m research_v2.src.run_stage1 \
    --epochs "$EPOCHS_S1" \
    --batch "$BATCH" \
    --lr "$LR" \
    --workers 8 \
    --channels-last \
    --val-pairs "$VAL_PAIRS"

echo "Finished at `date`"
