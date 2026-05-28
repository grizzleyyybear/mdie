#!/bin/bash
# ============================================================================
#  MDIE — full publication pipeline end-to-end on PARAM Siddhi-AI.
#    Phase 1: baselines (50 ep)
#    Phase 2: MDIE-full + ablation (60 ep)
#    Phase 3: MFR2 / CALFW / AgeDB-30 real-benchmark eval
#    Phase 4: Grad-CAM interpretability grid
#    Phase 5: rebuild research-paper & explainer PDFs with new numbers
#
#  Expected walltime: 9-12 h on 1 A100-SXM4 (well under 168 h cap).
#
#  Submit:  sbatch hpc/slurm_full_pipeline.sh
# ============================================================================
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=12:00:00
#SBATCH --job-name=mdie-full
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=dgxnp

# shellcheck source=./_prelude.sh
source "$(dirname "$0")/_prelude.sh"

EPOCHS_S1="${EPOCHS_S1:-50}"
EPOCHS_S2="${EPOCHS_S2:-60}"
BATCH="${BATCH:-256}"
LR="${LR:-2e-3}"
VAL_PAIRS="${VAL_PAIRS:-3000}"
CPB="${CPB:-32}"
SPC="${SPC:-8}"

echo
echo "============================================================"
echo "[phase 0] preflight"
echo "============================================================"
python -m research_v2.src.preflight

echo
echo "============================================================"
echo "[phase 1] Stage-1 baselines (epochs=$EPOCHS_S1 batch=$BATCH)"
echo "============================================================"
python -m research_v2.src.run_stage1 \
    --epochs "$EPOCHS_S1" --batch "$BATCH" --lr "$LR" \
    --workers 8 --channels-last --val-pairs "$VAL_PAIRS"

echo
echo "============================================================"
echo "[phase 2] Stage-2 MDIE + ablation (epochs=$EPOCHS_S2)"
echo "============================================================"
python -m research_v2.src.run_stage2 \
    --epochs "$EPOCHS_S2" --batch "$BATCH" --lr "$LR" \
    --workers 8 --channels-last \
    --balanced-sampler --classes-per-batch "$CPB" --samples-per-class "$SPC" \
    --val-pairs "$VAL_PAIRS" --ablation

echo
echo "============================================================"
echo "[phase 3] real-benchmark eval"
echo "============================================================"
python -m research_v2.src.eval.run_real_benchmarks || true

echo
echo "============================================================"
echo "[phase 4] Grad-CAM"
echo "============================================================"
python -m research_v2.src.eval.gradcam || true

echo
echo "============================================================"
echo "[phase 5] rebuild paper artefacts"
echo "============================================================"
python -m research_v2.src.paper.build_research_pdf || true
python -m research_v2.src.paper.build_explainer_pdf || true

echo
echo "[done] artefacts under research_v2/{results,figures,checkpoints}/"
echo "Finished at `date`"
