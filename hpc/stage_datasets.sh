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

# Workspace-aware Conda (matches hpc/_prelude.sh)
_grand="$(cd "$REPO_ROOT/../.." 2>/dev/null && pwd || echo "")"
_parent="$(cd "$REPO_ROOT/.." 2>/dev/null && pwd || echo "")"
if [[ -n "$_grand" && "$_grand" == "$HOME"/* \
      && "$(basename "$_parent")" == "projects" ]]; then
    _PREFIX_DEFAULT="$_grand"
else
    _PREFIX_DEFAULT="$HOME"
fi
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-$_PREFIX_DEFAULT/Conda}"
ENV_NAME="${ENV_NAME:-mdie}"

# CDAC proxy: login node may need it for upstream downloads.
PARAM_PROXY="${PARAM_PROXY:-http://proxy-10g.10g.siddhi.param:9090}"
if [[ "${PARAM_USE_PROXY:-auto}" == "1" ]] || \
   ! curl -fsS --max-time 5 -o /dev/null https://huggingface.co 2>/dev/null; then
    echo "[proxy] enabling CDAC proxy: $PARAM_PROXY"
    export http_proxy="$PARAM_PROXY"
    export https_proxy="$PARAM_PROXY"
    export ftp_proxy="$PARAM_PROXY"
    export HTTP_PROXY="$PARAM_PROXY"
    export HTTPS_PROXY="$PARAM_PROXY"
fi

# shellcheck disable=SC1091
source "$CONDA_PREFIX_DIR/bin/activate" "$ENV_NAME"

echo "[stage] cache root: $CACHE_ROOT"
df -h "$CACHE_ROOT" | tail -1 || true

# ---- 1. LFW ---------------------------------------------------------------
LFW_DIR="$CACHE_ROOT/lfw/lfw"
LFW_URLS=(
    "https://vis-www.cs.umass.edu/lfw/lfw.tgz"
    "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
)

_lfw_ok() {
    # Tarball expanded? Need both the dir and at least one jpg inside.
    [[ -d "$LFW_DIR" ]] && [[ -n "$(find "$LFW_DIR" -name '*.jpg' -print -quit 2>/dev/null)" ]]
}

if ! _lfw_ok; then
    LFW_TGZ="$CACHE_ROOT/lfw/lfw.tgz"
    mkdir -p "$CACHE_ROOT/lfw"
    rm -f "$LFW_TGZ"

    fetched=0
    for url in "${LFW_URLS[@]}"; do
        for attempt in 1 2; do
            echo "[lfw] try $attempt: wget $url"
            # --show-progress so we *see* what's happening; --tries handles flaky links;
            # `|| true` so set -e doesn't kill us — we evaluate exit status manually.
            if wget --tries=3 --timeout=60 --show-progress -O "$LFW_TGZ" "$url"; then
                if [[ -s "$LFW_TGZ" ]]; then fetched=1; break; fi
            fi
            echo "[lfw] download failed; will retry"
            sleep 3
        done
        [[ "$fetched" == "1" ]] && break
    done

    if [[ "$fetched" != "1" ]]; then
        echo "[lfw] ERROR: couldn't fetch lfw.tgz from any mirror."
        echo "       Options:"
        echo "         1. Re-run with proxy forced on:"
        echo "              PARAM_USE_PROXY=1 bash hpc/stage_datasets.sh"
        echo "         2. Download lfw.tgz on your laptop and scp it to:"
        echo "              $LFW_TGZ"
        echo "            then re-run this script."
        exit 1
    fi

    echo "[lfw] extracting..."
    tar -xzf "$LFW_TGZ" -C "$CACHE_ROOT/lfw"
    rm -f "$LFW_TGZ"
fi

n_imgs=$(find "$LFW_DIR" -name '*.jpg' 2>/dev/null | wc -l)
echo "[lfw] OK ($n_imgs images at $LFW_DIR)"

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
