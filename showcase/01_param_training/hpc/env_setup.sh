#!/bin/bash
# ============================================================================
#  MDIE — one-shot environment bootstrap for PARAM Siddhi-AI (CDAC, Pune).
#  Follows the official user guide §7.1 (Miniconda in $HOME/Conda).
#
#  Run ONCE on the login node:
#       bash hpc/env_setup.sh
#
#  Subsequent jobs reactivate with:
#       source $HOME/Conda/bin/activate mdie
# ============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Workspace-aware Conda location (see hpc/_prelude.sh for the full explanation).
# Repo at $HOME/<sub>/projects/mdie  -> Conda at $HOME/<sub>/Conda
# Otherwise                          -> Conda at $HOME/Conda
# Always overridable via:  CONDA_PREFIX_DIR=/path  bash hpc/env_setup.sh
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
PY_VER="${PY_VER:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

# --- CDAC proxy (compute nodes have no outbound internet; login node *might*
#     also need this for pip/wget — set it whenever PARAM_USE_PROXY=1, or
#     auto-detect when wget can't reach pypi). ----------------------------
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
else
    echo "[proxy] PARAM_PROXY unreachable; using direct connections"
fi

echo "============================================================"
echo "  MDIE env_setup on PARAM Siddhi-AI"
echo "    repo  : $REPO_ROOT"
echo "    conda : $CONDA_PREFIX_DIR"
echo "    env   : $ENV_NAME (python $PY_VER)"
echo "============================================================"

# ---- 1. Install Miniconda if not already present --------------------------
if [[ ! -d "$CONDA_PREFIX_DIR" ]]; then
    echo "[conda] installing Miniconda to $CONDA_PREFIX_DIR ..."
    mkdir -p "$HOME/tmp"
    export TMPDIR="$HOME/tmp"
    INSTALLER="$HOME/tmp/Miniconda3-latest-Linux-x86_64.sh"
    wget -q -O "$INSTALLER" \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "$INSTALLER" -b -p "$CONDA_PREFIX_DIR" -u
    rm -f "$INSTALLER"
fi

# shellcheck disable=SC1091
source "$CONDA_PREFIX_DIR/bin/activate"

# Use conda-forge only — Anaconda's defaults channel now requires an
# interactive ToS click which breaks unattended HPC installs.
conda config --remove-key channels 2>/dev/null || true
conda config --add channels conda-forge
conda config --set channel_priority strict
conda config --set always_yes true >/dev/null

# Best-effort: accept defaults ToS so existing installs that already added
# the 'main'/'r' channels keep working. Ignored silently if conda-tos
# isn't supported on this conda version.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true

# ---- 2. Create / reuse the mdie env ---------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[conda] creating env '$ENV_NAME' (python=$PY_VER) from conda-forge ..."
    conda create -n "$ENV_NAME" -c conda-forge --override-channels \
        "python=$PY_VER" pip
fi
# shellcheck disable=SC1091
source "$CONDA_PREFIX_DIR/bin/activate" "$ENV_NAME"

python -m pip install --upgrade pip wheel setuptools

# ---- 3. PyTorch wheel that talks to A100-SXM4 -----------------------------
echo "[torch] installing from $TORCH_INDEX_URL"
pip install --index-url "$TORCH_INDEX_URL" \
    "torch>=2.1.0" "torchvision>=0.16.0"

# ---- 4. Rest of the project requirements ----------------------------------
pip install -r "$REPO_ROOT/requirements.txt"
# stage_datasets.sh needs gdown for the MFR2 Google Drive download
pip install gdown

# ---- 5. CUDA sanity check (runs on login node — torch.cuda may be False) --
python - <<'PY'
import torch
print(f"[verify] torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[verify] device 0 = {torch.cuda.get_device_name(0)}")
PY

# ---- 6. Project preflight (CPU side only) ---------------------------------
#  Skip the LFW-touching part if data isn't staged yet — otherwise sklearn
#  would try to download LFW from figshare (often blocked from PARAM) and hang.
cd "$REPO_ROOT"
export MDIE_SKIP_DATASET_PREFLIGHT="${MDIE_SKIP_DATASET_PREFLIGHT:-auto}"
PYTHONPATH="$REPO_ROOT" python -m research_v2.src.preflight || true

cat <<EOF

============================================================
  env_setup complete.

  To reactivate the env later:
      source $CONDA_PREFIX_DIR/bin/activate $ENV_NAME

  Next steps:
      bash hpc/stage_datasets.sh
      sbatch hpc/slurm_quick.sh
============================================================
EOF
