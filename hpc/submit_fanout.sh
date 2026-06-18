#!/bin/bash
# ============================================================================
#  Orchestrate the MDIE fan-out: submit the 4-way variant array, then a
#  dependency-chained merge that gathers the per-variant results/fanout/<v>/
#  metrics into a single combined ablation summary.
#
#       bash hpc/submit_fanout.sh
#
#  Pass-through env knobs are forwarded to the array job, e.g.:
#       DATASET=casia BACKBONE=ir50 EPOCHS_S2=40 bash hpc/submit_fanout.sh
# ============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "[submit] launching 4-way variant fan-out array ..."
ARRAY_JID="$(sbatch --parsable hpc/slurm_fanout_train.sh)"
echo "[submit] array job id: $ARRAY_JID"

echo "[submit] queuing merge (runs after the whole array finishes ok) ..."
MERGE_JID="$(sbatch --parsable \
    --dependency="afterok:${ARRAY_JID}" \
    --job-name=mdie-merge \
    --partition=dgxnp \
    --time=00:20:00 \
    --output=job.%J.out --error=job.%J.err \
    --wrap="cd '$REPO_ROOT' && export PYTHONPATH='$REPO_ROOT' && python -m research_v2.src.merge_fanout")"
echo "[submit] merge job id: $MERGE_JID  (depends on $ARRAY_JID)"

cat <<MSG

[ok] Submitted:
   - fan-out array : $ARRAY_JID   (4 tasks, 1 GPU each)
   - merge         : $MERGE_JID   (afterok:$ARRAY_JID)
Watch with:  squeue --me
Per-variant outputs: research_v2/results/fanout/<variant>/
Merged summary:      research_v2/results/fanout/ablation_merged.json
MSG
