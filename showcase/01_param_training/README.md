# MDIE PARAM Training Package

This is the clean PARAM Siddhi-AI version of the MDIE codebase. It is designed for A100-SXM4 training under SLURM and keeps the source, dataset staging, job scripts, and operator instructions in one folder.

## Layout

```text
01_param_training\
|-- hpc\                         SLURM scripts and PARAM bootstrap scripts
|-- research_v2\src\              MDIE source code
|-- research_v2\datasets_cache\   dataset staging target
|-- research_v2\checkpoints\      model outputs
|-- research_v2\results\          metrics outputs
|-- research_v2\figures\          generated paper figures
|-- PARAM_HANDBOOK.md             complete PARAM operating notes
`-- requirements.txt              Python dependencies
```

## First run on PARAM

From the PARAM login node:

```bash
cd ~/mrinal/projects/mdie/showcase/01_param_training
bash hpc/env_setup.sh
bash hpc/stage_datasets.sh
sbatch hpc/slurm_quick.sh
```

If the quick job finishes cleanly:

```bash
sbatch hpc/slurm_full_pipeline.sh
```

## Job choices

| Script | Use case |
|---|---|
| `hpc/slurm_quick.sh` | 4-epoch smoke test. Run this first. |
| `hpc/slurm_stage1.sh` | Baseline failure quantification. |
| `hpc/slurm_stage2.sh` | MDIE + ablation training. |
| `hpc/slurm_eval.sh` | Public benchmark evaluation and Grad-CAM. |
| `hpc/slurm_full_pipeline.sh` | Full publication pipeline. |
| `hpc/interactive.sh` | A100 debug shell. |

## PARAM defaults

- Partition: `dgxnp`
- GPU request: `--gres=gpu:A100-SXM4:1`
- CPU rule: `--ntasks-per-node=16` for 1 A100
- Environment: conda env `mdie`
- Code import root: current folder via `PYTHONPATH`

## Pull results back to laptop

From Windows PowerShell after the job completes:

```powershell
scp -r bhatib@login-siddhi.pune.cdac.in:~/mrinal/projects/mdie/showcase/01_param_training/research_v2/results .\param_results
scp -r bhatib@login-siddhi.pune.cdac.in:~/mrinal/projects/mdie/showcase/01_param_training/research_v2/figures .\param_figures
```

Use `PARAM_HANDBOOK.md` for proxy, dataset, upload, and troubleshooting details.

