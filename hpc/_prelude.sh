# ---- Common prelude for every PARAM Siddhi-AI SLURM job in this folder ----
# Sourced from each slurm_*.sh after the SBATCH header.
# Echoes the standard SLURM env vars (per the user-guide §8.3.1 template),
# activates the Conda env, sets PYTHONPATH, and chdirs into the repo.

echo "============================================================"
echo "Starting at `date`"
echo "Running on hosts: ${SLURM_NODELIST:-?}"
echo "Running on ${SLURM_NNODES:-?} nodes."
echo "Running ${SLURM_NTASKS:-?} tasks."
echo "Job id is ${SLURM_JOBID:-?}"
echo "Job submission directory is  : ${SLURM_SUBMIT_DIR:-$PWD}"
echo "============================================================"

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(pwd)"

# Activate the Miniconda env created by hpc/env_setup.sh.
# Conda's activate script references unset vars internally, so we
# temporarily lift `set -u` (re-enabled afterwards is intentionally OFF —
# keep `set -e` + pipefail; `-u` is too aggressive for HPC env scripts).
#
# Workspace-aware Conda location:
#   - If repo lives at  $HOME/<sub>/projects/mdie  (e.g. ~/mrinal/projects/mdie)
#     Conda goes to     $HOME/<sub>/Conda          (e.g. ~/mrinal/Conda)
#     so multiple users sharing one account stay isolated.
#   - Otherwise         $HOME/Conda                (single-user default)
#   - Always overridable via  CONDA_PREFIX_DIR  env var.
_mdie_grand="$(cd "$REPO_ROOT/../.." 2>/dev/null && pwd || echo "")"
_mdie_parent="$(cd "$REPO_ROOT/.." 2>/dev/null && pwd || echo "")"
if [[ -n "$_mdie_grand" && "$_mdie_grand" == "$HOME"/* \
      && "$(basename "$_mdie_parent")" == "projects" ]]; then
    _MDIE_PREFIX_DEFAULT="$_mdie_grand"
else
    _MDIE_PREFIX_DEFAULT="$HOME"
fi
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-$_MDIE_PREFIX_DEFAULT/Conda}"
ENV_NAME="${ENV_NAME:-mdie}"
echo "[prelude] CONDA_PREFIX_DIR=$CONDA_PREFIX_DIR  ENV_NAME=$ENV_NAME"
# shellcheck disable=SC1091
source "$CONDA_PREFIX_DIR/bin/activate" "$ENV_NAME"

export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Match dataloader/openMP fan-out to allocated cores.
# Advisory 6 (PARAM Siddhi-AI): 16 CPUs per GPU (was 32).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS_PER_NODE:-16}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# onnxruntime-gpu (used to build the one-off CASIA landmark cache) needs cuDNN 9
# + cuBLAS at runtime. The torch CUDA wheels already bundle them under
# site-packages/nvidia/*/lib, but they're not on LD_LIBRARY_PATH by default, so
# ORT fails to load libonnxruntime_providers_cuda.so and SILENTLY falls back to
# CPU — turning the ~390k-image cache from minutes on the A100 into ~16 h (which
# blows past the walltime). Prepend the bundled NVIDIA lib dirs so the GPU
# provider loads. Harmless when the libs are absent (block is a no-op).
_nvidia_libs="$(python - <<'PY' 2>/dev/null
import glob, importlib.util, os
spec = importlib.util.find_spec("nvidia")
locs = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
dirs = sorted({os.path.dirname(p) for base in locs
               for p in glob.glob(os.path.join(base, "*", "lib", "*.so*"))})
print(":".join(dirs))
PY
)"
if [[ -n "$_nvidia_libs" ]]; then
    export LD_LIBRARY_PATH="$_nvidia_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[prelude] added bundled NVIDIA libs (cuDNN/cuBLAS) to LD_LIBRARY_PATH for onnxruntime-gpu"
fi

# CDAC proxy for outbound HTTP from compute nodes. Enabled by default unless
# explicitly disabled with PARAM_USE_PROXY=0.
PARAM_PROXY="${PARAM_PROXY:-http://proxy-10g.10g.siddhi.param:9090}"
_use_proxy="${PARAM_USE_PROXY:-auto}"
if [[ "$_use_proxy" == "0" ]]; then
    echo "[prelude] proxy disabled (PARAM_USE_PROXY=0)"
elif [[ "$_use_proxy" == "1" ]] || \
     curl -fsS --max-time 5 --proxy "$PARAM_PROXY" -o /dev/null https://pypi.org/simple/ 2>/dev/null; then
    echo "[prelude] enabling CDAC proxy: $PARAM_PROXY"
    export http_proxy="$PARAM_PROXY"
    export https_proxy="$PARAM_PROXY"
    export ftp_proxy="$PARAM_PROXY"
    export HTTP_PROXY="$PARAM_PROXY"
    export HTTPS_PROXY="$PARAM_PROXY"
fi

mkdir -p research_v2/logs

nvidia-smi || true
python -c 'import torch; ok = torch.cuda.is_available(); dev = torch.cuda.get_device_name(0) if ok else "-"; print(f"[torch] {torch.__version__}  cuda={ok}  device={dev}")'
