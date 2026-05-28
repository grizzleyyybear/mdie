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
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-$HOME/Conda}"
ENV_NAME="${ENV_NAME:-mdie}"
# shellcheck disable=SC1091
source "$CONDA_PREFIX_DIR/bin/activate" "$ENV_NAME"

export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Match dataloader/openMP fan-out to allocated cores.
# Advisory 6 (PARAM Siddhi-AI): 16 CPUs per GPU (was 32).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS_PER_NODE:-16}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

mkdir -p research_v2/logs

nvidia-smi || true
python -c 'import torch; ok = torch.cuda.is_available(); dev = torch.cuda.get_device_name(0) if ok else "-"; print(f"[torch] {torch.__version__}  cuda={ok}  device={dev}")'
