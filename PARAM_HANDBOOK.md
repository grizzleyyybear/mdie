# MDIE on PARAM Siddhi-AI — Operator Handbook

Single-file reference for running this codebase on the
**CDAC PARAM Siddhi-AI** A100 cluster. Bookmark this file.

Last updated to follow:
- *PARAM Siddhi-AI User Guide* (88 pp, official)
- *Advisory 1–7* (NLTM partition, CPU partition, walltime cap, job
  limits, `--exclusive` ban, **16 cpus/GPU**, `quota_check`)
- *Fortinet PARAM VPN Manual*

If the cluster operators publish a newer advisory that contradicts this
file, the advisory wins — update both this file and the scripts in
`hpc/` accordingly.

---

## 0 · One-page cheat sheet

```bash
# (one-time, from your laptop)
# install Fortinet SSL-VPN client, connect to gateway:  npsf-gw.cdac.in:443

# (one-time, on the cluster)
ssh <user>@login-siddhi.pune.cdac.in
git clone <repo-url> $HOME/mdie
cd $HOME/mdie
bash hpc/env_setup.sh              # installs Miniconda + creates `mdie` env
bash hpc/stage_datasets.sh         # downloads LFW + MFR2 + CALFW + AgeDB-30

# (every run)
sbatch hpc/slurm_quick.sh          # 4-epoch smoke (1 h walltime)
squeue -u $USER                    # monitor your queue
sbatch hpc/slurm_full_pipeline.sh  # full publication run (~12 h)
scancel <jobid>                    # kill a job
quota_check                        # check storage quota
```

---

## 1 · Cluster facts (PARAM Siddhi-AI)

| Item | Value |
|---|---|
| Login host | `login-siddhi.pune.cdac.in` (or just `login` from inside the cluster) |
| Architecture | DGX-A100 nodes (31 in `dgxnp`), 8 × **A100-SXM4-40GB** per node |
| Scheduler | SLURM — partition `dgxnp` is default for GPU, `cpup` for CPU-only |
| GRES name | `gpu:A100-SXM4:N` (NOT `gpu:A100`). Generic `gpu:N` also works. |
| Storage | Lustre PFS at `$HOME`, e.g. `/nlsasfs/home/<group>/<user>` |
| Quota check | `quota_check` (run on login node) |
| Frameworks | Miniconda in `$HOME/Conda` (no module subsystem for ML) |
| Driver / CUDA | NVIDIA driver **550.90**, CUDA **12.4** runtime (PyTorch cu121 wheels work) |
| Partition QoS | `nodeallocgpu` (auto-applied to dgxnp) |
| Default CPU/GPU | `DefCpuPerGPU=16` (matches advisory 6) |
| Internet on compute nodes | **NOT available** — pre-stage data from login node |

## 2 · Hard rules from the advisories (must follow)

| # | Rule | Where applied |
|---|---|---|
| **A1** | Partition: this cluster uses `dgxnp` for A100 jobs (verified via `sinfo`). CPU-only jobs go to `cpup`. | All `slurm_*.sh` set `#SBATCH --partition=dgxnp`. |
| **A3** | Interactive job walltime is **max 4 h** (was 7 d). | `hpc/interactive.sh` uses 1 h by default. |
| **A4** | **8 running + 8 queued** jobs per user, max 16 total submitted. | Don't bulk-submit ablations — chain via `--dependency=afterok:<jobid>`. |
| **A5** | **Do not use `--exclusive`.** Reserve GPUs only via `--gres`. | None of our scripts pass `--exclusive`. |
| **A6** | **16 CPUs per GPU** (reduced from 32). For 2 GPUs → 32, full node (8) → 128. | All `slurm_*.sh` use `--ntasks-per-node=16` for 1 A100. |
| **A7** | `quota_check` shows project occupancy (soft/hard limits, 1-week grace). | Run periodically — checkpoints can be large. |

Default batch-job walltime cap remains **168 h** (7 days).

## 3 · Login + VPN

External access requires the Fortinet SSL-VPN.

| Field | Value |
|---|---|
| VPN type | SSL-VPN |
| Remote gateway | `npsf-gw.cdac.in` |
| Port | `443` |
| Client | FortiClient VPN (Windows / Linux / macOS) |
| Credentials | Supplied separately by CDAC support |

Once the VPN tunnel is up, `ssh <user>@login-siddhi.pune.cdac.in`.

For files larger than a few MB, use `scp` / `rsync` (Linux/macOS) or
WinSCP / MobaXterm (Windows) — see user-guide §5.3.

## 4 · Repository layout (what runs where)

```
aicte/                                  ← repo root, clone to $HOME/mdie/
├── hpc/                                ← PARAM-specific scripts (this folder)
│   ├── env_setup.sh                    ← once, login node
│   ├── stage_datasets.sh               ← once, login node
│   ├── interactive.sh                  ← srun helper (debug)
│   ├── _prelude.sh                     ← sourced by all SBATCH scripts
│   ├── slurm_quick.sh                  ← sbatch, smoke test
│   ├── slurm_stage1.sh                 ← sbatch, baselines
│   ├── slurm_stage2.sh                 ← sbatch, MDIE + ablation
│   ├── slurm_eval.sh                   ← sbatch, real-bench eval
│   ├── slurm_full_pipeline.sh          ← sbatch, everything end-to-end
│   └── README.md                       ← short index of this folder
├── research_v2/
│   ├── src/                            ← all training, eval, paper code
│   ├── datasets_cache/                 ← populated by stage_datasets.sh
│   ├── checkpoints/                    ← trainers write *.pt here
│   ├── results/                        ← metrics CSV/JSON
│   ├── figures/                        ← ROC PDFs + paper PDFs
│   └── logs/                           ← per-job log files
├── requirements.txt                    ← pip dependencies
└── PARAM_HANDBOOK.md                   ← this file
```

## 5 · Provided SBATCH scripts — verbatim conventions

Every script in `hpc/` mirrors the official user-guide §8.3.1 template:

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=16        # advisory 6: 16 CPUs / GPU
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=HH:MM:SS
#SBATCH --job-name=mdie-<phase>
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=dgxnp           # A100 partition on PARAM Siddhi-AI

source "$(dirname "$0")/_prelude.sh" # activates Conda env, echoes SLURM vars

python -m research_v2.src.<module> [args...]
```

`_prelude.sh` prints the standard SLURM diagnostic block (NODELIST,
NNODES, NTASKS, JOBID, SUBMIT_DIR), `cd`s into `$SLURM_SUBMIT_DIR`,
activates the `mdie` conda env, sets `PYTHONPATH`, and runs
`nvidia-smi` to confirm GPU visibility.

### Tuneable env vars (override on the sbatch command line)

```bash
EPOCHS_S1=80 BATCH=192 sbatch hpc/slurm_stage1.sh
```

> **Memory note**: This cluster has **A100-SXM4 40 GB** GPUs (not 80 GB).
> Defaults (`BATCH=256`, IR-50 @ 112², AMP) fit in ~22 GB and are safe.
> If you ever hit `CUDA out of memory`, drop `BATCH` to 192 or 128.

| Var | Default | Used by |
|---|---|---|
| `EPOCHS_S1` | 50 | stage1, full |
| `EPOCHS_S2` | 60 | stage2, full |
| `BATCH` | 256 | stage1, stage2, full |
| `LR` | 2e-3 | stage1, stage2, full |
| `VAL_PAIRS` | 3000 | stage1, stage2, full |
| `CPB` / `SPC` | 32 / 8 | stage2, full (balanced sampler) |
| `CONDA_PREFIX_DIR` | `$HOME/Conda` | env_setup, prelude |
| `ENV_NAME` | `mdie` | env_setup, prelude |
| `IIITD_ROOT` | unset | eval (gated dataset) |
| `IJBC_ROOT` | unset | eval (gated dataset) |
| `HF_ENDPOINT` | upstream | stage_datasets (use mirror if blocked) |

## 6 · End-to-end workflow

### 6.1 First time on the cluster

```bash
ssh <user>@login-siddhi.pune.cdac.in
git clone <repo-url> $HOME/mdie
cd $HOME/mdie
bash hpc/env_setup.sh           # ~5 min  (Miniconda + PyTorch wheels)
bash hpc/stage_datasets.sh      # ~30 min (LFW + MFR2 + CALFW + AgeDB-30)
quota_check                     # confirm you have headroom
```

### 6.2 Sanity check (always do this on a fresh account)

```bash
sbatch hpc/slurm_quick.sh
# 4-epoch smoke: stage1 + stage2 + real-bench eval in ~30 min walltime
# Outputs land in research_v2/{results,figures,checkpoints}/ as usual.
```

If `job.<JOBID>.err` is empty and `job.<JOBID>.out` ends with
`[smoke OK]`, you're cleared for the long run.

### 6.3 Publication run

```bash
sbatch hpc/slurm_full_pipeline.sh
# ~9-12 h walltime, single A100. Produces:
#   research_v2/results/{stage1,stage2,real_benchmarks}.{json,csv}
#   research_v2/figures/{roc_*,methodology,mdie_explainer,mdie_research_paper}.pdf
#   research_v2/checkpoints/*.pt
```

### 6.4 Chaining jobs (respect 8+8 limit, advisory A4)

```bash
JID1=$(sbatch --parsable hpc/slurm_stage1.sh)
JID2=$(sbatch --parsable --dependency=afterok:$JID1 hpc/slurm_stage2.sh)
sbatch        --dependency=afterok:$JID2 hpc/slurm_eval.sh
```

### 6.5 Interactive debugging shell

```bash
bash hpc/interactive.sh           # srun --pty /bin/bash, 1 A100, 1 h
# inside the allocated node:
source $HOME/Conda/bin/activate mdie
cd $HOME/mdie
python -m research_v2.src.preflight
```

## 7 · Job control commands

| Task | Command |
|---|---|
| Submit | `sbatch hpc/slurm_*.sh` |
| List my jobs | `squeue -u $USER` |
| Inspect 1 job | `squeue --job <jobid>` |
| Cancel | `scancel <jobid>` |
| Show why pending | `squeue -u $USER -o "%.18i %.8j %.10T %.20R"` |
| Disk quota | `quota_check` |
| GPU on node | `nvidia-smi` (inside an allocation) |

## 8 · Pulling artefacts back to your laptop

```bash
# from a Linux/macOS terminal on your laptop
rsync -avh --include='*/' \
            --include='research_v2/results/***' \
            --include='research_v2/figures/***' \
            --exclude='*' \
    <user>@login-siddhi.pune.cdac.in:~/mdie/  ./mdie/

# include checkpoints too (large)
rsync -avh <user>@login-siddhi.pune.cdac.in:~/mdie/research_v2/checkpoints/ \
            ./mdie/research_v2/checkpoints/
```

Windows: use **WinSCP** (drag & drop) or **MobaXterm** (upload/download
buttons), per user-guide §5.3.

## 9 · Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `sbatch: error: Batch script contains DOS line breaks` | CRLF endings | `dos2unix hpc/*.sh` (we set `.gitattributes` to LF) |
| `Invalid generic resource (gres) specification` | Wrong GRES name | Use `gpu:A100-SXM4:N`, not `gpu:A100:N` |
| `Invalid partition name specified` | Wrong partition | Confirm with `sinfo` then set `#SBATCH --partition=<name>` (this cluster uses `dgxnp` for A100, `cpup` for CPU). |
| Job stuck in `PD` (pending) with reason `QOSMaxJobsPerUserLimit` | Hit advisory A4 (8+8) | `scancel` queued jobs you don't need, or wait |
| Job stuck in `PD` with reason `Resources` | Cluster busy | Just wait — Siddhi can be heavily loaded |
| `CUDA out of memory` | batch too large | `BATCH=128 sbatch hpc/slurm_*.sh` |
| `disk quota exceeded` | hit soft/hard limit | `quota_check`, delete old `checkpoints/*.pt`, or ask CDAC to extend |
| HuggingFace timeout in `stage_datasets.sh` | upstream blocked | `HF_ENDPOINT=https://hf-mirror.com bash hpc/stage_datasets.sh` |
| `torch.cuda.is_available() == False` inside job | env not activated | confirm `_prelude.sh` ran; check `which python` is the conda one |
| `bash -n` of script fails | edited on Windows in Notepad | re-save as UTF-8 LF, or `sed -i 's/\r$//' hpc/*.sh` |
| Compute node has no internet | known, by design | always pre-stage data from login node |

## 10 · Where to ask for help

| Channel | Use for |
|---|---|
| `helpdesk-siddhi@cdac.in` | Account, quota, partition assignment, hardware issues |
| User-guide PDF (88 pp) | SLURM details, MPI, CUDA-aware MPI, NGC containers |
| Fortinet VPN PDF | VPN client install + connection problems |
| Advisory PDF | Latest policy changes (re-read after every CDAC announcement) |
| Repo `README.md` | Codebase questions (training, eval, paper rebuild) |

## 11 · Provenance — exact mapping of advisories to code

| Advisory | Effect on scripts |
|---|---|
| A1 (partition) | `#SBATCH --partition=dgxnp` in every `slurm_*.sh` (A100 partition, verified via `sinfo`). |
| A2 (`cpup` for CPU-only) | If you want to rebuild PDFs without a GPU, edit `slurm_eval.sh`: remove `--gres`, drop `cpup`, drop `--gres` flag entirely; the eval code falls back to CPU. |
| A3 (4 h interactive cap) | `hpc/interactive.sh` requests 1 h; user may raise up to 4 h. |
| A4 (8+8 jobs per user) | Documented; chain with `--dependency=afterok:<jobid>`. |
| A5 (no `--exclusive`) | None of our scripts pass it. |
| A6 (16 CPUs/GPU) | All scripts use `--ntasks-per-node=16` for 1 A100. `OMP_NUM_THREADS` derived from it in `_prelude.sh`. |
| A7 (`quota_check`) | Documented in §0, §1, §6.1, §9. |
