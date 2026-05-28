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
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-$HOME/Conda}"
ENV_NAME="${ENV_NAME:-mdie}"
PY_VER="${PY_VER:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

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
conda config --set always_yes true --set channel_priority strict >/dev/null

# ---- 2. Create / reuse the mdie env ---------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[conda] creating env '$ENV_NAME' (python=$PY_VER) ..."
    conda create -n "$ENV_NAME" "python=$PY_VER" pip
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
cd "$REPO_ROOT"
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
