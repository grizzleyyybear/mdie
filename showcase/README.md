# MDIE Showcase Codebase

This folder is the clean handoff version of the MDIE project. It keeps the
useful code, PARAM training scripts, laptop-safe runners, latest paper assets,
diagrams, and result tables without old logs, cache files, failed attempts, or
notebook clutter.

## Folder map

| Folder | Purpose |
|---|---|
| `01_param_training` | Production training/evaluation package for PARAM Siddhi-AI A100 systems. Includes SLURM scripts, environment setup, dataset staging, source code, and PARAM instructions. |
| `02_laptop_rtx3050` | Laptop-safe runnable package for an RTX 3050 4 GB machine. Uses smaller batches, AMP, quick runs, and offline result files for demos. |
| `03_theory_and_diagrams` | Presentation material: paper PDFs, methodology diagrams, latest metrics, benchmark tables, and talking notes. |

## What to show

1. Open `03_theory_and_diagrams\documents\PRESENTATION_NOTES.md` for the verbal explanation.
2. Open `03_theory_and_diagrams\figures\methodology.pdf` or `methodology_simple.pdf` for the architecture.
3. Show `03_theory_and_diagrams\figures\stage1_per_mod_eer.png` to prove the failure modes.
4. Show `03_theory_and_diagrams\figures\stage2_attention_examples.png` to explain RATA.
5. Show `03_theory_and_diagrams\results\real_benchmarks.csv` for public benchmark evaluation.

## Codebase story

MDIE has two targets:

- **PARAM target:** full A100 training and benchmark reproduction.
- **Laptop target:** quick verification, demos, and code walkthrough on RTX 3050 4 GB.

Both targets use the same `research_v2\src` implementation so the logic stays consistent.

## Sanity check

From the repository root:

```powershell
python -m unittest discover -s tests
```

That check is intentionally lightweight: it verifies the three-folder handoff,
confirms the PARAM and laptop source copies match the maintained source, catches
stale imports, and compiles the Python files without requiring datasets or a GPU.

