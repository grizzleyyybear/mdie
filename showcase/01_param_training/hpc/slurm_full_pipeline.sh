#!/bin/bash
# ============================================================================
#  MDIE - full publication pipeline end-to-end on PARAM Siddhi-AI.
#  Occlusion+lighting-robust, ArcFace-compatible single-512-d MDIE.
#
#    Phase 0: preflight
#    Phase 1: comparably-trained baselines (facenet/arcface/cosface/mobilefacenet)
#    Phase 2: MDIE-full + ablation (noRATA/noAMD/noICCL), fused 512-d head
#    Phase 3: real-benchmark eval (MFR2 masks, MeGlass glasses, CALFW, AgeDB-30)
#    Phase 4: attention-bone IoU interpretability (matched/mismatched/random)
#    Phase 5: ArcFace-compatibility inference proof (single 512-d, cosine==dot)
#    Phase 6: build the research paper PDF from paper/paper.tex (pdflatex)
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
EPOCHS_S2="${EPOCHS_S2:-30}"
BATCH="${BATCH:-256}"
LR="${LR:-2e-3}"
VAL_PAIRS="${VAL_PAIRS:-3000}"
CPB="${CPB:-32}"
SPC="${SPC:-8}"
BENCHMARKS="${BENCHMARKS:-mfr2 meglass calfw agedb30}"

echo
echo "============================================================"
echo "[phase 0] preflight"
echo "============================================================"
python -m research_v2.src.preflight

echo
echo "============================================================"
echo "[phase 1] comparably-trained baselines (epochs=$EPOCHS_S1 batch=$BATCH)"
echo "============================================================"
python -m research_v2.src.run_stage1 \
    --epochs "$EPOCHS_S1" --batch "$BATCH" --lr "$LR" \
    --workers 8 --channels-last --val-pairs "$VAL_PAIRS"

echo
echo "============================================================"
echo "[phase 2] MDIE-full + ablation (epochs=$EPOCHS_S2), fused 512-d head"
echo "============================================================"
python -m research_v2.src.run_stage2 \
    --epochs "$EPOCHS_S2" --batch "$BATCH" --lr "$LR" \
    --workers 8 --channels-last \
    --balanced-sampler --classes-per-batch "$CPB" --samples-per-class "$SPC" \
    --val-pairs "$VAL_PAIRS" --ablation

echo
echo "============================================================"
echo "[phase 3] real-benchmark eval ($BENCHMARKS)"
echo "============================================================"
# shellcheck disable=SC2086
python -m research_v2.src.eval.run_real_benchmarks \
    --models mdie arcface cosface mobilefacenet \
    --benchmarks $BENCHMARKS || true

echo
echo "============================================================"
echo "[phase 4] attention-bone IoU interpretability"
echo "============================================================"
python research_v2/scripts/attention_bone_iou.py 60 || true

echo
echo "============================================================"
echo "[phase 5] ArcFace-compatibility inference proof"
echo "============================================================"
python research_v2/scripts/inference_compat_proof.py || true

echo
echo "============================================================"
echo "[phase 6] build research paper PDF (paper/paper.tex)"
echo "============================================================"
if command -v pdflatex >/dev/null 2>&1; then
    ( cd research_v2/paper && \
      pdflatex -interaction=nonstopmode -halt-on-error paper.tex && \
      pdflatex -interaction=nonstopmode -halt-on-error paper.tex ) || \
      echo "[warn] pdflatex build failed; inspect research_v2/paper/paper.log"
    if [ -f research_v2/paper/paper.pdf ]; then
        cp -f research_v2/paper/paper.pdf research_v2/figures/mdie_research_paper.pdf
        echo "[ok] copied paper.pdf -> figures/mdie_research_paper.pdf"
    fi
else
    echo "[skip] pdflatex not on PATH; paper/paper.tex left for offline build"
fi

echo
echo "[done] artefacts under research_v2/{results,figures,checkpoints}/"
echo "  results/stage2_metrics.json            (per-mod + family AUC/EER)"
echo "  results/security_family_summary.json   (occlusion/lighting family table)"
echo "  results/real_benchmarks.{csv,json}     (MFR2/MeGlass/CALFW/AgeDB-30)"
echo "  results/inference_compat_proof.json    (ArcFace-compat deployment proof)"
echo "  figures/attention_bone_iou.png         (bone-IoU interpretability)"
echo "Finished at `date`"
