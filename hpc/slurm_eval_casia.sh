#!/bin/bash
# ============================================================================
#  MDIE - real-benchmark eval for the CASIA-trained fan-out checkpoints.
#
#  Scores the CASIA MDIE checkpoint(s) under research_v2/checkpoints/fanout/
#  against the REAL worn-occlusion / aging benchmarks (MFR2 real masks,
#  MeGlass real glasses, CALFW + AgeDB-30 aging) and, by default, the strong
#  production reference InsightFace-w600k_r50 (WebFace12M) on the same axes.
#  This is the synthetic->real transfer story: how the CASIA-scale MDIE holds
#  up on genuinely occluded faces vs a production model.
#
#  Inference-only. Walltime < 1 h on 1 A100. Batch-only (no srun needed):
#       sbatch hpc/slurm_eval_casia.sh
#
#  Knobs (all optional, via --export or env):
#    VARIANT             which fan-out checkpoint is the headline (default mdie_full)
#    EVAL_ALL_VARIANTS=1 also score the other 3 ablation variants (per-variant CSVs)
#    BENCHMARKS="..."    space list (default: mfr2 meglass calfw agedb30)
#    BASELINES="..."     external baselines for the headline plot
#                        (default: insightface_w600k_r50)
#
#  Outputs land under research_v2/results/casia_real/ :
#    real_benchmarks_casia.csv   headline table (MDIE-full + baselines)
#    roc_<bench>.pdf/png         combined ROC curves (figures/casia_real/)
#    <variant>/real_benchmarks_<variant>.csv   per-variant ablation (if enabled)
# ============================================================================
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=01:30:00
#SBATCH --job-name=mdie-evalc
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=dgxnp

# shellcheck source=./_prelude.sh
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    source "$SLURM_SUBMIT_DIR/hpc/_prelude.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/_prelude.sh"
fi

BENCHMARKS="${BENCHMARKS:-mfr2 meglass calfw agedb30}"
BASELINES="${BASELINES:-insightface_w600k_r50}"
VARIANT="${VARIANT:-mdie_full}"

FANOUT="$REPO_ROOT/research_v2/checkpoints/fanout"
OUTDIR="$REPO_ROOT/research_v2/results/casia_real"
mkdir -p "$OUTDIR"

if [[ ! -f "$FANOUT/$VARIANT/$VARIANT.pt" ]]; then
    echo "[eval-casia] ERROR: headline checkpoint not found:" >&2
    echo "             $FANOUT/$VARIANT/$VARIANT.pt" >&2
    echo "             (did the fan-out training finish? check research_v2/checkpoints/fanout/)" >&2
    exit 2
fi

echo "[eval-casia] headline variant : $VARIANT"
echo "[eval-casia] baselines        : $BASELINES"
echo "[eval-casia] benchmarks       : $BENCHMARKS"
echo "[eval-casia] outputs          : $OUTDIR"

# --- Headline run: CASIA MDIE-full + production baselines on the SAME axes ---
# MDIE_EVAL_VARIANT makes the mdie loader pick this variant's .pt and report it
# under its own column name; MDIE_CKPT_DIR points the loader at its fan-out dir.
export MDIE_CKPT_DIR="$FANOUT/$VARIANT"
export MDIE_EVAL_VARIANT="$VARIANT"
export MDIE_RESULTS_DIR="$OUTDIR"
export MDIE_FIGURES_DIR="$REPO_ROOT/research_v2/figures/casia_real"
mkdir -p "$MDIE_FIGURES_DIR"

# shellcheck disable=SC2086
python -m research_v2.src.eval.run_real_benchmarks \
    --models mdie $BASELINES \
    --benchmarks $BENCHMARKS \
    --out "$OUTDIR/real_benchmarks_casia.csv"

# --- Optional: the other 3 ablation variants (per-variant CSVs) --------------
if [[ "${EVAL_ALL_VARIANTS:-0}" == "1" ]]; then
    for V in mdie_norata mdie_noamd mdie_noiccl; do
        CK="$FANOUT/$V/$V.pt"
        if [[ ! -f "$CK" ]]; then
            echo "[eval-casia] skip $V (no checkpoint at $CK)"
            continue
        fi
        echo "[eval-casia] ablation variant: $V"
        export MDIE_CKPT_DIR="$FANOUT/$V"
        export MDIE_EVAL_VARIANT="$V"
        export MDIE_RESULTS_DIR="$OUTDIR/$V"
        export MDIE_FIGURES_DIR="$REPO_ROOT/research_v2/figures/casia_real/$V"
        mkdir -p "$MDIE_RESULTS_DIR" "$MDIE_FIGURES_DIR"
        # shellcheck disable=SC2086
        python -m research_v2.src.eval.run_real_benchmarks \
            --models mdie \
            --benchmarks $BENCHMARKS \
            --out "$OUTDIR/$V/real_benchmarks_$V.csv"
    done
fi

echo "[eval-casia] Finished at `date`"
