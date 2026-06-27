#!/bin/bash
# ============================================================================
#  Orchestrate the MDIE fan-out (BATCH only — no interactive login needed):
#    1. (optional) build the shared bone-landmark cache once,
#    2. the 4-way variant array (depends on the cache job),
#    3. a polling merge guard that gathers the per-variant metrics.
#
#       bash hpc/submit_fanout.sh
#
#  Pass-through env knobs are forwarded to the jobs, e.g.:
#       DATASET=casia BACKBONE=ir50 EPOCHS_S2=40 bash hpc/submit_fanout.sh
#
#  Set BUILD_CACHE=0 to skip step 1 (e.g. the cache already exists). When the
#  cache exists the build job is a fast no-op anyway (verify + exit), so the
#  default is safe and idempotent.
# ============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_CACHE="${BUILD_CACHE:-1}"
ARRAY_DEP=""
if [[ "$BUILD_CACHE" != "0" ]]; then
    echo "[submit] queuing one-off bone-landmark cache build ..."
    CACHE_JID="$(sbatch --parsable hpc/slurm_build_cache.sh)"
    echo "[submit] cache job id: $CACHE_JID  (fan-out waits for it)"
    # afterok is correct here: the cache job has a fixed walltime and either
    # finishes (cache ready) or fails — it never self-resubmits.
    ARRAY_DEP="--dependency=afterok:${CACHE_JID}"
fi

echo "[submit] launching 4-way variant fan-out array ..."
ARRAY_JID="$(sbatch --parsable $ARRAY_DEP hpc/slurm_fanout_train.sh)"
echo "[submit] array job id: $ARRAY_JID"

# The training tasks may self-resubmit to survive the 8 h walltime, so an
# `afterok` dependency on the array can fire while a variant is still training.
# Use the polling merge guard instead: it waits for every variant's .complete
# sentinel before merging (re-queues itself every 30 min until then).
echo "[submit] queuing merge guard (waits for all .complete sentinels) ..."
MERGE_JID="$(sbatch --parsable hpc/merge_guard.sh)"
echo "[submit] merge guard job id: $MERGE_JID"

cat <<MSG

[ok] Submitted (all batch — no interactive login needed):
   - cache build   : ${CACHE_JID:-skipped (BUILD_CACHE=0)}
   - fan-out array : $ARRAY_JID   (4 tasks, 1 GPU each; self-resubmit on 8 h walltime)
   - merge guard   : $MERGE_JID   (polls for results/fanout/<variant>/.complete)
Watch with:  squeue --me
Per-variant outputs: research_v2/results/fanout/<variant>/
Merged summary:      research_v2/results/fanout/ablation_merged.json
MSG
