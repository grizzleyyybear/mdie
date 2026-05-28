#!/bin/bash
# ============================================================================
#  MDIE — real-benchmark eval (MFR2/CALFW/AgeDB-30) + Grad-CAM + paper PDFs.
#  Inference-only. Walltime: < 1 h on 1 A100-SXM4.
#
#  Submit:  sbatch hpc/slurm_eval.sh
# ============================================================================
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=01:00:00
#SBATCH --job-name=mdie-eval
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
##SBATCH --partition=nltmp    # uncomment if you have a project partition (advisory 1)

# shellcheck source=./_prelude.sh
source "$(dirname "$0")/_prelude.sh"

python -m research_v2.src.eval.run_real_benchmarks
python -m research_v2.src.eval.gradcam            || true
python -m research_v2.src.paper.build_research_pdf  || true
python -m research_v2.src.paper.build_explainer_pdf || true

echo "Finished at `date`"
