# MDIE on PARAM Siddhi-AI (CDAC, Pune)

> **For complete operator instructions, see `../PARAM_HANDBOOK.md`.**
> This file is just the directory index.

Runnable scripts here follow the official
**PARAM Siddhi-AI user guide** + **advisories 1–7**:

* Resource manager: SLURM
* GRES syntax: `--gres=gpu:A100-SXM4:N`
* CPU rule (advisory 6): **16 cores per GPU** →
  `--ntasks-per-node=16` for 1 A100, `32` for 2, `128` for a full
  8-GPU node
* Walltime: max **168 h** for batch jobs; max **4 h** for interactive
  (advisory 3); default 1 h
* Job limits (advisory 4): max **8 running + 8 queued** per user
* Do **not** pass `--exclusive` (advisory 5); reserve GPUs via `--gres` only
* Output/error: `job.%J.out` / `job.%J.err`
* No environment-modules subsystem — Python frameworks live in a
  **Miniconda** install under `$HOME/Conda/`
* Login node: `login-siddhi.pune.cdac.in`
* Storage: `/home/<user>` on a 10.5 PiB lustre PFS (subject to quota)

```
hpc/
├── README.md                  this file
├── env_setup.sh               once: install Miniconda + create mdie env
├── stage_datasets.sh          once: download LFW + MFR2 + CALFW + AgeDB-30
├── slurm_quick.sh             4-epoch smoke run on 1 A100 (~30 min)
├── slurm_stage1.sh            Stage-1 baselines on 1 A100  (~3–4 h)
├── slurm_stage2.sh            Stage-2 MDIE + ablation on 1 A100 (~5–6 h)
├── slurm_eval.sh              real-bench eval + bone-IoU + compat proof (<1.5 h)
├── slurm_full_pipeline.sh     all of the above end-to-end (~12 h)
└── interactive.sh             srun helper for an A100 debug shell
```

---

## 0. Connect

From inside the C-DAC network (or via the institutional VPN — see
*Fortinet PARAM VPN* doc):

```
ssh <username>@login-siddhi.pune.cdac.in
```

Clone the repo into your home directory:

```bash
cd $HOME
git clone <this_repo> mdie
cd mdie
```

## 1. One-time environment setup (login node)

```bash
bash hpc/env_setup.sh
```

This will:

1. Download and run Miniconda installer into `$HOME/Conda/` (per the
   PARAM user guide §7.1).
2. Create a conda env named **`mdie`** with Python 3.11.
3. `pip install` PyTorch (CUDA 12.1 wheels — these run fine against the
   PARAM A100 driver) and the rest of `requirements.txt`.
4. Run `python -m research_v2.src.preflight` to verify the env.

Re-activating the env later:

```bash
source $HOME/Conda/bin/activate mdie
```

(All SLURM scripts in this folder do this for you.)

## 2. Stage the datasets (login node, once)

```bash
bash hpc/stage_datasets.sh
```

Downloads the public benchmarks (MFR2 ~13 MB, CALFW ~185 MB,
AgeDB-30 ~200 MB, LFW ~180 MB) into
`$HOME/mdie/research_v2/datasets_cache/`. Idempotent.

Gated datasets are NOT downloaded — set the following env vars in your
sbatch script if you have license access:

| Variable | Maps to |
|---|---|
| `IIITD_ROOT` | IIITD Plastic Surgery root dir |
| `IJBC_ROOT`  | IJB-C images + protocol root dir |

For MS1MV3 backbone pretraining, drop a torch-format
`ir50_pretrained.pth` into `research_v2/checkpoints/` (if you have it).

## 3. Submit jobs

```bash
sbatch hpc/slurm_quick.sh           # do this FIRST on a new account
squeue -u $USER                     # monitor
tail -f job.<JOBID>.out             # follow log (written to cwd by SLURM)
```

Once the smoke run finishes cleanly:

```bash
sbatch hpc/slurm_full_pipeline.sh
```

## 4. Interactive A100 shell (debug only)

```bash
bash hpc/interactive.sh             # 1 A100, 1 hour
```

Drops you onto a compute node. Inside the shell, remember to
`source $HOME/Conda/bin/activate mdie` and `cd $HOME/mdie`.

## 5. Pull artefacts back

From your local machine:

```bash
# only the small artefacts
rsync -avh --include='*/' \
    --include='results/***' --include='figures/***' \
    --exclude='*' \
    <user>@login-siddhi.pune.cdac.in:~/mdie/research_v2/  ./research_v2/

# or include checkpoints (large)
rsync -avh \
    <user>@login-siddhi.pune.cdac.in:~/mdie/research_v2/  ./research_v2/
```

---

## Notes specific to PARAM Siddhi-AI

* Reserving 2 GPUs requires 64 cores, 3 → 96, 8 (full node) → 256.
* Compute nodes generally have no outbound internet — that is why
  datasets must be staged from the login node.
* HuggingFace mirrors: if `huggingface.co` is blocked from the login
  node, export `HF_ENDPOINT=https://hf-mirror.com` before
  `stage_datasets.sh`.
* `nvidia-smi` confirms allocated GPUs inside any SLURM job.
* `scancel <jobid>` to kill a job; `squeue --job <jobid>` to inspect it.
