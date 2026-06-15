# MDIE Theory, Diagrams, and Results

This folder contains the presentation material only: latest documents, diagrams, tables, and benchmark outputs. It is the folder to open during a viva, project review, or research discussion.

## Documents

| File | Use |
|---|---|
| `documents\PRESENTATION_NOTES.md` | Main speaking notes and Q&A. |
| `documents\METHODOLOGY.md` | Full technical method explanation. |
| `documents\METHODOLOGY_SIMPLE.md` | Short four-block methodology explanation. |
| `documents\paper.tex` | Paper source for the current draft. |
| `documents\PROJECT_README.md` | Original project-level README. |
| `documents\RESEARCH_README.md` | Research code README. |
| `documents\PARAM_HANDBOOK.md` | PARAM execution and troubleshooting notes. |

## Figures to present

| Figure | Message |
|---|---|
| `figures\methodology.pdf` | MDIE architecture: RATA + AMD + consistency training. |
| `figures\methodology_simple.pdf` | Simple visual explanation for non-specialists. |
| `figures\stage1_per_mod_eer.png` | Baselines fail hardest under masks/glasses/low light. |
| `figures\stage1_roc_pooled.png` | Controlled failure-mode ROC evidence. |
| `figures\stage2_per_mod_eer.png` | MDIE and ablation comparison. |
| `figures\stage2_attention_examples.png` | Attention visualization for the region-aware argument. |
| `figures\roc_mfr2.png` | Real masked-face benchmark. |
| `figures\roc_calfw.png` | Cross-age benchmark. |
| `figures\roc_agedb30.png` | AgeDB-30 benchmark. |

## Result files

| File | Contains |
|---|---|
| `results\stage1_metrics.json` | Baseline failure-mode numbers. |
| `results\stage2_metrics.json` | MDIE and ablation numbers. |
| `results\real_benchmarks.csv` | MFR2, CALFW, AgeDB-30 benchmark table. |
| `results\stage1_tables.tex` | Paper-ready Stage-1 tables. |
| `results\stage2_tables.tex` | Paper-ready Stage-2 tables. |

## One-line project framing

MDIE is an occlusion- and lighting-robust face verification framework for security / access-control use. It combines bone-anchored region-aware attention (RATA) with adversarial disentanglement (AMD) and an inter-condition consistency loss (ICCL) so identity embeddings preserve identity while suppressing modification cues such as masks, caps, glasses, partial occluders, and lighting shifts (low-light, over-exposure, harsh shadow). The deployed embedding is a single 512-d ArcFace-compatible vector.

