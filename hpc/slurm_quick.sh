#!/bin/bash
# ============================================================================
#  MDIE — 4-epoch smoke test on PARAM Siddhi-AI.
#  Use FIRST on a new account to verify env + datasets + GPU visibility.
#  Walltime budget: 1 hour (well under default 1 h cap).
#
#  Submit:  sbatch hpc/slurm_quick.sh
# ============================================================================
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=01:00:00
#SBATCH --job-name=mdie-quick
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
# If you have been allocated a project partition (e.g. nltmp, cpup),
# uncomment the next line and set it accordingly (advisory 1):
##SBATCH --partition=nltmp

# shellcheck source=./_prelude.sh
source "$(dirname "$0")/_prelude.sh"

python -m research_v2.src.preflight

python -m research_v2.src.run_stage1 \
    --epochs 4 --batch 128 --workers 8 \
    --channels-last --val-pairs 500

python -m research_v2.src.run_stage2 \
    --epochs 4 --batch 128 --workers 8 \
    --channels-last --val-pairs 500 --quick

python -m research_v2.src.eval.run_real_benchmarks \
    --benchmarks mfr2 calfw agedb30

echo "[smoke OK] env is healthy — submit slurm_full_pipeline.sh next."
echo "Finished at `date`"
