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

# The training tasks may self-resubmit to survive the 8 h walltime, so an
# `afterok` dependency on the array can fire while a variant is still training.
# Use the polling merge guard instead: it waits for every variant's .complete
# sentinel before merging (re-queues itself every 30 min until then).
echo "[submit] queuing merge guard (waits for all .complete sentinels) ..."
MERGE_JID="$(sbatch --parsable hpc/merge_guard.sh)"
echo "[submit] merge guard job id: $MERGE_JID"

cat <<MSG

[ok] Submitted:
   - fan-out array : $ARRAY_JID   (4 tasks, 1 GPU each; self-resubmit on 8 h walltime)
   - merge guard   : $MERGE_JID   (polls for results/fanout/<variant>/.complete)
Watch with:  squeue --me
Per-variant outputs: research_v2/results/fanout/<variant>/
Merged summary:      research_v2/results/fanout/ablation_merged.json
MSG
