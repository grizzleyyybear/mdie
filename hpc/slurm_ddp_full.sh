#!/bin/bash
# ============================================================================
#  MDIE — Stage 2 on multiple A100s via DistributedDataParallel (torchrun).
#  Trains MDIE-full (+ ablation) at CASIA-WebFace scale across N GPUs on one
#  node. Single-GPU stays the default elsewhere; this is the opt-in big run.
#
#  Submit (4 GPUs on one node, CASIA, IR-100):
#       sbatch hpc/slurm_ddp_full.sh
#  Override knobs from the environment, e.g.:
#       GPUS=8 DATASET=casia BACKBONE=ir100 EPOCHS_S2=40 sbatch hpc/slurm_ddp_full.sh
#
#  NOTE on SLURM advisories: max 8 GPUs/node, 16 CPU/GPU, no --exclusive.
# ============================================================================
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100-SXM4:4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=mdie-ddp
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=dgxnp

# shellcheck source=./_prelude.sh
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    source "$SLURM_SUBMIT_DIR/hpc/_prelude.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/_prelude.sh"
fi

# How many GPUs to use for DDP. Defaults to what SLURM allocated on this node;
# can be overridden with GPUS=N (must be <= allocated).
GPUS="${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
GPUS="${GPUS:-1}"

DATASET="${DATASET:-casia}"
BACKBONE="${BACKBONE:-ir100}"
EPOCHS_S2="${EPOCHS_S2:-40}"
BATCH="${BATCH:-256}"          # per-GPU batch
LR="${LR:-4e-3}"               # scaled up for the larger effective batch
VAL_PAIRS="${VAL_PAIRS:-6000}"
CPB="${CPB:-32}"
SPC="${SPC:-8}"
PRETRAINED="${PRETRAINED:-0}"  # IR-100 has no public w600k weights -> from scratch

# Realistic occlusion augmentation ON by default for scale-up; REALISTIC_AUG=0 to disable.
export MDIE_REALISTIC_AUG="${REALISTIC_AUG:-1}"

echo "[ddp] GPUS=$GPUS DATASET=$DATASET BACKBONE=$BACKBONE EPOCHS=$EPOCHS_S2 BATCH=$BATCH LR=$LR realistic_aug=$MDIE_REALISTIC_AUG"

# IR-100 must be trained from scratch (no pretrained w600k IR-100 weights exist).
PRETRAINED_FLAG="--no-pretrained-backbone"
if [[ "$PRETRAINED" == "1" ]]; then PRETRAINED_FLAG="--pretrained-backbone"; fi

# Single-node multi-GPU launch. --standalone picks a free rendezvous port.
torchrun --standalone --nnodes=1 --nproc_per_node="$GPUS" \
    -m research_v2.src.run_stage2 \
    --ddp \
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
    --ablation

echo "Finished at `date`"
