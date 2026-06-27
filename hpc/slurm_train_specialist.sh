#!/bin/bash
# ============================================================================
#  MDIE - SPECIALIST recipe: occlusion specialist that starts at PRODUCTION
#  parity and improves only on its niche.
#
#  Why this differs from the CASIA fan-out (and why it can actually beat the
#  production reference on occluded faces):
#    * --pretrained-backbone --freeze-backbone : the WebFace-12M w600k IResNet50
#      is kept FROZEN, so there is zero catastrophic forgetting. The native
#      embedding stays exactly the production 0.955-AUC encoder.
#    * --residual-fusion : the deployed 512-d vector is
#          norm(native + gate * delta),  gate initialised to 0,
#      so AT INIT the model emits the production embedding byte-for-byte. The
#      attention/residual pathway then learns occlusion-robust corrections ON
#      TOP, instead of the random-init concat head that overwrote production
#      quality in the earlier CASIA run.
#    * DATASET defaults to a bigger/more-diverse public set (ms1mv3) so the
#      learned corrections generalise far better than CASIA's 10.5k identities.
#
#  Honest ceiling: with a frozen 12M backbone, clean/aging benchmarks are
#  capped at production PARITY (you match, not beat). The real, defensible win
#  is on REAL occluded faces (masks/glasses): MFR2, MeGlass, RMFRD.
#
#  Stage a bigger ImageFolder first (per-identity subdirs) at
#       research_v2/datasets_cache/<DATASET>/<identity>/<img>.jpg
#  e.g. DATASET=ms1mv3 (see hpc/fetch_trainset.sh), then:
#       DATASET=ms1mv3 sbatch hpc/slurm_train_specialist.sh
#  Smoke test on the already-staged CASIA set:
#       DATASET=casia EPOCHS_S2=2 sbatch hpc/slurm_train_specialist.sh
# ============================================================================
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=08:00:00
#SBATCH --job-name=mdie-spec
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=dgxnp

# shellcheck source=./_prelude.sh
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    source "$SLURM_SUBMIT_DIR/hpc/_prelude.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/_prelude.sh"
fi

SLUG="mdie_specialist"
export MDIE_RESULTS_DIR="$REPO_ROOT/research_v2/results/specialist"
export MDIE_FIGURES_DIR="$REPO_ROOT/research_v2/figures/specialist"
export MDIE_CKPT_DIR="$REPO_ROOT/research_v2/checkpoints/$SLUG"
mkdir -p "$MDIE_RESULTS_DIR" "$MDIE_FIGURES_DIR" "$MDIE_CKPT_DIR"
rm -f "$MDIE_RESULTS_DIR/.complete"

DATASET="${DATASET:-ms1mv3}"
BACKBONE="${BACKBONE:-ir50}"
EPOCHS_S2="${EPOCHS_S2:-30}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-3}"
VAL_PAIRS="${VAL_PAIRS:-6000}"
CPB="${CPB:-32}"
SPC="${SPC:-8}"

# Heavy realistic worn-occlusion augmentation is the whole point of the
# specialist; keep it on.
export MDIE_REALISTIC_AUG="${REALISTIC_AUG:-1}"

# --- 8 h walltime fail-safe (resume across resubmits) ----------------------
export MDIE_TRAIN_MAX_SECONDS="${MDIE_TRAIN_MAX_SECONDS:-25200}"
ATTEMPT="${MDIE_ATTEMPT:-1}"
MAX_ATTEMPTS="${MDIE_MAX_ATTEMPTS:-8}"
echo "[failsafe] attempt $ATTEMPT/$MAX_ATTEMPTS  budget=${MDIE_TRAIN_MAX_SECONDS}s  walltime=08:00:00"

echo "[specialist] dataset=$DATASET backbone=$BACKBONE epochs=$EPOCHS_S2 batch=$BATCH lr=$LR realistic_aug=$MDIE_REALISTIC_AUG"

# Backbone policy:
#   default            : FROZEN pretrained w600k + identity-init residual fusion.
#                        Deployed embedding starts at production parity; clean
#                        benchmarks are capped at parity, occlusion improves.
#   UNFREEZE=1          : lightly fine-tune the backbone (small LR mult) so a
#                        large/diverse set (e.g. Glint360K) can push past
#                        production on clean too, while residual fusion keeps the
#                        occlusion gains. Only sound on a big set -- on a small
#                        set this forgets (which is exactly what hurt the CASIA
#                        run). Not guaranteed to beat production; it is the
#                        honest best shot at an overall win.
BACKBONE_FLAGS="--freeze-backbone"
BB_LR_MULT="${BB_LR_MULT:-0.05}"
if [[ "${UNFREEZE:-0}" == "1" ]]; then
    BACKBONE_FLAGS="--backbone-lr-mult $BB_LR_MULT"
    echo "[specialist] UNFREEZE=1 -> fine-tuning backbone at lr_mult=$BB_LR_MULT (ambitious overall-win run)"
else
    echo "[specialist] FROZEN pretrained w600k backbone + identity-init residual fusion"
fi
echo "[specialist] outputs -> $MDIE_RESULTS_DIR"

set +e
python -m research_v2.src.run_stage2 \
    --dataset "$DATASET" \
    --backbone "$BACKBONE" \
    --pretrained-backbone \
    $BACKBONE_FLAGS \
    --residual-fusion \
    --epochs "$EPOCHS_S2" \
    --batch "$BATCH" \
    --lr "$LR" \
    --workers 8 \
    --channels-last \
    --balanced-sampler \
    --classes-per-batch "$CPB" \
    --samples-per-class "$SPC" \
    --val-pairs "$VAL_PAIRS" \
    --resume \
    --only-variant MDIE-full
RC=$?
set -e

# Exit 64 == clean stop on the wall-clock budget (not finished). Resubmit to
# resume from <slug>_last.pt.
if [[ "$RC" == "64" ]]; then
    if [[ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]]; then
        SUBMIT_SCRIPT="${SLURM_SUBMIT_DIR:-$REPO_ROOT}/hpc/slurm_train_specialist.sh"
        echo "[failsafe] budget reached; resubmitting (attempt $((ATTEMPT+1))/$MAX_ATTEMPTS)"
        sbatch --export=ALL,MDIE_ATTEMPT=$((ATTEMPT+1)),MDIE_MAX_ATTEMPTS=$MAX_ATTEMPTS,MDIE_TRAIN_MAX_SECONDS=$MDIE_TRAIN_MAX_SECONDS,DATASET=$DATASET,BACKBONE=$BACKBONE,EPOCHS_S2=$EPOCHS_S2,BATCH=$BATCH,LR=$LR,VAL_PAIRS=$VAL_PAIRS,CPB=$CPB,SPC=$SPC,REALISTIC_AUG=$MDIE_REALISTIC_AUG,UNFREEZE=${UNFREEZE:-0},BB_LR_MULT=$BB_LR_MULT \
            "$SUBMIT_SCRIPT"
    else
        echo "[failsafe] max attempts reached; stopping."
    fi
    exit 0
elif [[ "$RC" != "0" ]]; then
    echo "[specialist] training failed with code $RC" >&2
    exit "$RC"
fi

touch "$MDIE_RESULTS_DIR/.complete"

# --- Evaluate the trained specialist against production on real benchmarks ---
echo "[specialist] training done; scoring against production on real benchmarks"
export MDIE_EVAL_VARIANT="mdie_full"
export MDIE_RESULTS_DIR="$REPO_ROOT/research_v2/results/specialist_real"
export MDIE_FIGURES_DIR="$REPO_ROOT/research_v2/figures/specialist_real"
mkdir -p "$MDIE_RESULTS_DIR" "$MDIE_FIGURES_DIR"
python -m research_v2.src.eval.run_real_benchmarks \
    --models mdie_full insightface_w600k_r50 \
    --benchmarks mfr2 meglass calfw agedb30 \
    --out "$MDIE_RESULTS_DIR/real_benchmarks_specialist.csv"

echo "[specialist] Finished at `date`"
