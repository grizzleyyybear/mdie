# MDIE Showcase Codebase

Modification-Disentangled Identity Encoder (MDIE) is a training and evaluation
pipeline for **security / access-control face verification under worn occlusions
and adverse lighting**: masks, caps/hats, glasses, partial occluders, low light,
over-exposure, harsh directional shadow, plus aging and small adversarial
perturbations. The deployed model is an **ArcFace drop-in** — a single
L2-normalised 512-d embedding produced in one forward pass and matched by plain
cosine similarity.

This repository is now organized for review and demonstration first. Open
`showcase\` when you need the clean handoff package.

## Showcase layout

| Folder | Purpose |
|---|---|
| `showcase\01_param_training` | PARAM Siddhi-AI package for A100 training and evaluation. Includes SLURM scripts, environment setup, dataset staging, source code, and PARAM instructions. |
| `showcase\02_laptop_rtx3050` | Laptop package for RTX 3050 4 GB demos. Includes the same source code, conservative launch scripts, cached results, and quick validation commands. |
| `showcase\03_theory_and_diagrams` | Presentation package with the paper, methodology notes, figures, diagrams, benchmark tables, and speaking notes. |

Both runnable packages use the same implementation under `research_v2\src`, so
the PARAM and laptop paths stay logically identical.

## Fast laptop demo

```powershell
cd showcase\02_laptop_rtx3050
.\show_cached_results.ps1
.\run_smoke.ps1
```

Use quick training only when PyTorch/CUDA and the dataset cache are available:

```powershell
.\run_stage1_quick.ps1
.\run_stage2_quick.ps1
.\run_real_benchmarks.ps1
```

## PARAM run

From the PARAM login node:

```bash
cd ~/mrinal/projects/mdie/showcase/01_param_training
bash hpc/env_setup.sh
bash hpc/stage_datasets.sh
sbatch hpc/slurm_quick.sh
sbatch hpc/slurm_full_pipeline.sh
```

The PARAM package follows Siddhi-AI SLURM conventions:
`--partition=dgxnp`, `--gres=gpu:A100-SXM4:1`, and 16 CPU tasks per GPU.
Detailed operating notes live in `showcase\01_param_training\PARAM_HANDBOOK.md`.

## Quality checks

The repository includes a lightweight test suite that does not need datasets or
GPU access. It protects the handoff layout, validates the source-copy sync
between root/PARAM/laptop packages, catches stale imports from removed helper
files, and compiles the Python sources.

```powershell
python -m unittest discover -s tests
```

## Presentation order

1. `showcase\03_theory_and_diagrams\documents\PRESENTATION_NOTES.md`
2. `showcase\03_theory_and_diagrams\figures\methodology.pdf`
3. `showcase\03_theory_and_diagrams\figures\stage1_per_mod_eer.png`
4. `showcase\03_theory_and_diagrams\figures\stage2_attention_examples.png`
5. `showcase\03_theory_and_diagrams\results\real_benchmarks.csv`

## Maintained source

| Path | Role |
|---|---|
| `research_v2\src` | Core MDIE code: data loaders, modifications, models, losses, training, evaluation, and paper figure builders. |
| `hpc` | Source PARAM SLURM scripts mirrored into the showcase package. |
| `requirements.txt` | Minimal Python dependency set for the maintained implementation. |
| `PARAM_HANDBOOK.md` | Full PARAM operating handbook. |
| `PRESENTATION_NOTES.md` | Theory-focused review and viva notes. |

Generated datasets, checkpoints, logs, and local virtual environments are not
part of the handoff. They are recreated through the laptop and PARAM scripts.
