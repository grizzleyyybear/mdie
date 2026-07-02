#!/bin/bash
# ============================================================================
#  MDIE - stage a downloaded large training set into ImageFolder on a COMPUTE
#  node via SLURM. Do NOT run RecordIO/WebDataset extraction on the login node.
#
#  Normally you do not call this directly. Run on the login node:
#      bash hpc/fetch_trainset.sh ms1mv3
#  and fetch_trainset.sh downloads/locates the payload, then submits this job.
#
#  Direct usage (only if payload is already downloaded):
#      STAGE_KIND=REC STAGE_PAYLOAD=/path/to/recordio DATASET=ms1mv3 \
#          sbatch hpc/slurm_stage_trainset.sh
#
#  PARAM QOS requires a GPU for batch jobs, so this requests one A100 even
#  though extraction is mostly CPU/I/O. This is intentional: it keeps the work
#  off the login node and complies with site policy.
# ============================================================================
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=08:00:00
#SBATCH --job-name=mdie-stage
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=dgxnp

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "[stage] ERROR: do not run this script with bash on the login node." >&2
    echo "        Use: sbatch hpc/slurm_stage_trainset.sh" >&2
    echo "        Or:  bash hpc/fetch_trainset.sh <dataset>  (it submits this job)" >&2
    exit 2
fi

# shellcheck source=./_prelude.sh
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    source "$SLURM_SUBMIT_DIR/hpc/_prelude.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/_prelude.sh"
fi

DATASET="${DATASET:-}"
KIND="${STAGE_KIND:-}"
PAYLOAD="${STAGE_PAYLOAD:-}"
CACHE_ROOT="$REPO_ROOT/research_v2/datasets_cache"
DST="${STAGE_DST:-$CACHE_ROOT/$DATASET}"

if [[ -z "$DATASET" || -z "$KIND" || -z "$PAYLOAD" ]]; then
    echo "[stage] ERROR: DATASET, STAGE_KIND and STAGE_PAYLOAD are required." >&2
    exit 2
fi
if [[ "$KIND" != "REC" && "$KIND" != "WDS" ]]; then
    echo "[stage] ERROR: STAGE_KIND must be REC or WDS, got '$KIND'." >&2
    exit 2
fi
if [[ ! -d "$PAYLOAD" ]]; then
    echo "[stage] ERROR: STAGE_PAYLOAD is not a directory: $PAYLOAD" >&2
    exit 2
fi

mkdir -p "$CACHE_ROOT"
LOCK="$CACHE_ROOT/.stage_${DATASET}.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "[stage] ERROR: another staging job appears to be active for DATASET=$DATASET." >&2
    echo "        Lock: $LOCK" >&2
    echo "        If you are sure no job is running, remove it manually:" >&2
    echo "            rm -rf '$LOCK'" >&2
    exit 3
fi
echo "$SLURM_JOB_ID" > "$LOCK/jobid"
trap 'rm -rf "$LOCK"' EXIT

echo "[stage] dataset : $DATASET"
echo "[stage] kind    : $KIND"
echo "[stage] payload : $PAYLOAD"
echo "[stage] dst     : $DST"

if [[ -e "$DST" && ! -L "$DST" ]] && \
   find "$DST" -mindepth 2 \( -iname '*.jpg' -o -iname '*.png' \) -print -quit 2>/dev/null | grep -q .; then
    echo "[stage] $DST already contains images; validating loader and exiting."
else
    [[ -L "$DST" ]] && rm -f "$DST"
    TMP_OUT="$DST.partial"
    mkdir -p "$TMP_OUT"
    echo "[stage] using resumable partial dir: $TMP_OUT"
    if [[ "$KIND" == "REC" ]]; then
        PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/hpc/recordio_to_imagefolder.py" "$PAYLOAD" "$TMP_OUT"
    else
        PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/hpc/webdataset_to_imagefolder.py" "$PAYLOAD" "$TMP_OUT"
    fi

    if [[ -e "$DST" ]]; then
        rmdir "$DST" 2>/dev/null || {
            echo "[stage] ERROR: $DST exists and is not empty; refusing to overwrite." >&2
            echo "        Inspect it, then move/remove it if you want to finalize $TMP_OUT." >&2
            exit 4
        }
    fi
    mv "$TMP_OUT" "$DST"
    echo "[stage] finalized $DST"
fi

echo "[stage] verifying with the project loader ..."
PYTHONPATH="$REPO_ROOT" DATASET="$DATASET" python - <<'PY'
import os
from research_v2.src.config import DATA_DIR
from research_v2.src.data import build_train_dataset

ds = os.environ["DATASET"]
paths, labels, names = build_train_dataset(ds, DATA_DIR, min_imgs=4)
print(f"  [stage] loader sees {len(paths)} images across {len(names)} identities")
assert len(names) > 1000, "expected many identities -- check the staged dataset"
PY

cat <<MSG

[stage] OK. $DATASET is staged at $DST
[next] smoke test:
   BGI=1 VIS=1 DATASET=$DATASET EPOCHS_S2=2 sbatch hpc/slurm_train_specialist.sh
[next] full run:
   BGI=1 VIS=1 DATASET=$DATASET sbatch hpc/slurm_train_specialist.sh
MSG
