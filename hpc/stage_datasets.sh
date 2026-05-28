#!/bin/bash
# ============================================================================
#  Stage public face-verification benchmarks for MDIE on PARAM Siddhi-AI.
#
#  Run ONCE on the login node (compute nodes have no outbound internet):
#       bash hpc/stage_datasets.sh
#
#  Idempotent — re-runs only download what is missing.
# ============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE_ROOT="$REPO_ROOT/research_v2/datasets_cache"
mkdir -p "$CACHE_ROOT/benchmarks" "$CACHE_ROOT/lfw"

CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-$HOME/Conda}"
ENV_NAME="${ENV_NAME:-mdie}"
# shellcheck disable=SC1091
source "$CONDA_PREFIX_DIR/bin/activate" "$ENV_NAME"

echo "[stage] cache root: $CACHE_ROOT"
df -h "$CACHE_ROOT" | tail -1 || true

# ---- 1. LFW ---------------------------------------------------------------
LFW_DIR="$CACHE_ROOT/lfw/lfw"
if [[ ! -d "$LFW_DIR" ]]; then
    echo "[lfw] downloading..."
    LFW_TGZ="$CACHE_ROOT/lfw/lfw.tgz"
    wget -q -O "$LFW_TGZ" \
        https://vis-www.cs.umass.edu/lfw/lfw.tgz
    tar -xzf "$LFW_TGZ" -C "$CACHE_ROOT/lfw"
    rm -f "$LFW_TGZ"
fi
echo "[lfw] OK ($(find "$LFW_DIR" -name '*.jpg' | wc -l) images)"

# ---- 2. MFR2 / CALFW / AgeDB-30 via the project loaders -------------------
echo "[bench] triggering loader auto-download..."
cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT" python - <<'PY'
from research_v2.src.data.benchmarks import load_benchmark
for name in ["mfr2", "calfw", "agedb30"]:
    try:
        b = load_benchmark(name)
        print(f"  {name:10s}: {len(b.pairs)} pairs  ({b.notes})")
    except Exception as e:
        print(f"  {name:10s}: FAILED -> {e}")
PY

cat <<'MSG'

[note] Gated datasets are NOT downloaded automatically:
   - IIITD Plastic Surgery   -> add  export IIITD_ROOT=<path>  to your sbatch script
   - IJB-C occlusion         -> add  export IJBC_ROOT=<path>   to your sbatch script
   - MS1MV3 pretrain seed    -> drop  ir50_pretrained.pth      into research_v2/checkpoints/

[tip] If huggingface.co is blocked from the login node, retry with:
       HF_ENDPOINT=https://hf-mirror.com bash hpc/stage_datasets.sh
MSG

echo "[stage] done.  Total cache: $(du -sh "$CACHE_ROOT" | cut -f1)"
