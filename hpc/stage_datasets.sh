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

# CDAC proxy: PARAM Siddhi-AI requires the proxy for arbitrary external
# downloads (pypi is whitelisted, vis-www.cs.umass.edu / figshare are not).
# Default behaviour: enable the proxy if PARAM_PROXY is reachable.
# Disable with: PARAM_USE_PROXY=0 bash hpc/stage_datasets.sh
PARAM_PROXY="${PARAM_PROXY:-http://proxy-10g.10g.siddhi.param:9090}"
_use_proxy="${PARAM_USE_PROXY:-auto}"
if [[ "$_use_proxy" == "0" ]]; then
    echo "[proxy] disabled (PARAM_USE_PROXY=0)"
elif [[ "$_use_proxy" == "1" ]] || \
     curl -fsS --max-time 5 --proxy "$PARAM_PROXY" -o /dev/null https://pypi.org/simple/ 2>/dev/null; then
    echo "[proxy] enabling CDAC proxy: $PARAM_PROXY"
    export http_proxy="$PARAM_PROXY"
    export https_proxy="$PARAM_PROXY"
    export ftp_proxy="$PARAM_PROXY"
    export HTTP_PROXY="$PARAM_PROXY"
    export HTTPS_PROXY="$PARAM_PROXY"
    # also forward to python/requests via env
    export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
else
    echo "[proxy] PARAM_PROXY unreachable; trying direct connections"
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

if _lfw_ok; then
    echo "[lfw] already staged at $LFW_DIR"
else
    LFW_TGZ="$CACHE_ROOT/lfw/lfw.tgz"
    mkdir -p "$CACHE_ROOT/lfw"

    # If user pre-uploaded the tarball, use it directly.
    if [[ ! -s "$LFW_TGZ" ]]; then
        fetched_url=""
        for url in "${LFW_URLS[@]}"; do
            for attempt in 1 2; do
                echo "[lfw] try $attempt: $url"
                if wget --tries=3 --timeout=120 --show-progress -O "$LFW_TGZ" "$url"; then
                    if [[ -s "$LFW_TGZ" ]]; then fetched_url="$url"; break; fi
                fi
                echo "[lfw] failed; retrying..."
                sleep 3
            done
            [[ -n "$fetched_url" ]] && break
        done

        if [[ -z "$fetched_url" ]]; then
            rm -f "$LFW_TGZ"
            echo ""
            echo "[lfw] ERROR: could not fetch LFW from any mirror."
            echo "       Force the CDAC proxy and retry:"
            echo "           PARAM_USE_PROXY=1 bash hpc/stage_datasets.sh"
            echo "       Or download lfw.tgz on a machine with internet and scp it to:"
            echo "           $LFW_TGZ"
            echo "       Then re-run:  bash hpc/stage_datasets.sh"
            exit 1
        fi
        echo "[lfw] downloaded from $fetched_url"
    else
        echo "[lfw] using pre-uploaded tarball at $LFW_TGZ"
    fi

    echo "[lfw] extracting..."
    tar -xzf "$LFW_TGZ" -C "$CACHE_ROOT/lfw"
    rm -f "$LFW_TGZ"
fi

n_imgs=$(find "$LFW_DIR" -name '*.jpg' 2>/dev/null | wc -l)
if [[ "$n_imgs" -lt 100 ]]; then
    echo "[lfw] ERROR: only $n_imgs jpgs found under $LFW_DIR (expected >13000)."
    exit 1
fi
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
