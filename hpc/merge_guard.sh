#!/bin/bash
# ============================================================================
#  MDIE fan-out merge guard.
#
#  The fan-out training tasks (hpc/slurm_fanout_train.sh) may self-resubmit to
#  survive the 8 h walltime, so a plain `afterok` dependency on the array job
#  can fire while a variant is still training in a follow-up job. This guard
#  instead WAITS for every variant's completion sentinel
#  (results/fanout/<slug>/.complete) before merging.
#
#  Submitted automatically by hpc/submit_fanout.sh. It re-queues itself every
#  ~30 min until all variants are done (capped by MERGE_MAX_ATTEMPTS), then runs
#  research_v2.src.merge_fanout once.
# ============================================================================
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=00:15:00
#SBATCH --job-name=mdie-merge
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=dgxnp

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    source "$SLURM_SUBMIT_DIR/hpc/_prelude.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/_prelude.sh"
fi

SLUGS=(mdie_full mdie_norata mdie_noamd mdie_noiccl)
FANOUT_DIR="$REPO_ROOT/research_v2/results/fanout"

missing=()
for s in "${SLUGS[@]}"; do
    [[ -f "$FANOUT_DIR/$s/.complete" ]] || missing+=("$s")
done

ATTEMPT="${MERGE_ATTEMPT:-1}"
MAX_ATTEMPTS="${MERGE_MAX_ATTEMPTS:-48}"   # 48 x 30 min = 24 h max wait
SUBMIT_SCRIPT="${SLURM_SUBMIT_DIR:-$REPO_ROOT}/hpc/merge_guard.sh"

if (( ${#missing[@]} > 0 )); then
    if (( ATTEMPT < MAX_ATTEMPTS )); then
        echo "[merge-guard] waiting on: ${missing[*]}  (attempt $ATTEMPT/$MAX_ATTEMPTS); recheck in 30 min"
        sbatch --begin=now+30minutes \
            --export=ALL,MERGE_ATTEMPT=$((ATTEMPT+1)),MERGE_MAX_ATTEMPTS=$MAX_ATTEMPTS \
            "$SUBMIT_SCRIPT"
        echo "[merge-guard] re-queued; exiting 0."
        exit 0
    fi
    echo "[merge-guard] timed out waiting for: ${missing[*]} — merging whatever completed." >&2
fi

echo "[merge-guard] all variants present; merging ..."
python -m research_v2.src.merge_fanout
echo "[merge-guard] done at `date`"
