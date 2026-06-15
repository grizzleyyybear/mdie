# research_v2 — Occlusion+Lighting-Robust Face Recognition (ArcFace-compatible)

Publishable research codebase. Niche: **security / access-control face
verification under worn occlusions (mask, cap/hat, glasses, partial occluder)
and adverse lighting (low-light, over-exposure, harsh shadow)**. The deployed
embedding is a **single L2-normalised 512-d vector** (one forward pass, plain
cosine match) — an **ArcFace drop-in** for edge devices. Three layers:

| Layer | Script | Output |
|-------|--------|--------|
| 1. Baselines + failure modes (controlled LFW) | `python -m src.run_stage1` | `figures/stage1_*`, `results/stage1_*` |
| 2. Novel MDIE + ablation                       | `python -m src.run_stage2 --ablation` | `figures/stage2_*`, `results/stage2_*` |
| 3. Real-benchmark eval + interpretability + deploy proof | `python -m src.eval.run_real_benchmarks` <br> `python scripts/attention_bone_iou.py` <br> `python scripts/inference_compat_proof.py` | `results/real_benchmarks.csv`, `figures/roc_*.png`, `figures/attention_bone_iou.png`, `results/inference_compat_proof.json` |

## Quick smoke-test
```cmd
cd research_v2
python -m src.run_stage1 --quick           REM ~10 min on RTX 3050
python -m src.run_stage2 --quick --ablation
```

## Full run
```cmd
research_v2\run_full_v3.bat               REM end-to-end ~4 hours on RTX 3050
```
or manually:
```cmd
python -m src.run_stage1 --epochs 50 --batch 32
python -m src.run_stage2 --epochs 40 --batch 24 --ablation
python -m src.eval.run_real_benchmarks
python scripts/attention_bone_iou.py
python scripts/inference_compat_proof.py
```

## Layout
```
research_v2/
├── src/
│   ├── config.py
│   ├── pretrained.py        face.evoLVe IR-50 local seed loader
│   ├── run_stage1.py        baseline failure-mode runner
│   ├── run_stage2.py        MDIE training + comparison runner
│   ├── data/                LFW + modifications + 5 real-bench loaders
│   ├── models/              IR-50, MobileFaceNet, InceptionResnetV1, IResNet50
│   ├── baselines/           4 SOTA baseline models + trainer
│   ├── novel/               MDIE — RATA (ablation) + AMD + ICCL trainer
│   ├── train/               pretrain_backbone entry point
│   ├── eval/                embeddings, ROC/EER/TAR@FAR, occlusion sensitivity,
│   │                        run_real_benchmarks (real benchmarks), occlusion sensitivity
│   └── paper/               publication figures + LaTeX tables + methodology docs
├── checkpoints/             best weights per model (+ drop-in instructions)
├── results/                 JSON + CSV + LaTeX tables
├── figures/                 300-dpi PNG + PDF figures for the paper
├── datasets_cache/          LFW + benchmark .bin files
└── paper/                   IEEE-conference draft + methodology docs
```

## Models registered in the eval harness
`facenet`, `arcface`, `cosface`, `mobilefacenet`, `mdie`, `ir50_pretrained`,
**`insightface_w600k_r50`** (production IR-50 trained on WebFace12M,
auto-downloaded from HuggingFace `Icar/buffalo_l-torch`).

## Benchmarks registered
`mfr2` (masks), `meglass` (glasses), `calfw` (cross-age), `agedb30` (aging),
`iiitd_surgery` (gated — set `IIITD_ROOT`), `ijbc_occ` (gated — set `IJBC_ROOT`).

For real benchmarks the loader expects InsightFace `.bin` files dropped into
`datasets_cache/benchmarks/<name>/`. See
`datasets_cache/benchmarks/README.md` for the exact path conventions.

## Read more
- `paper/METHODOLOGY.md`        — full methodology in prose
- `paper/METHODOLOGY_SIMPLE.md` — four-box version + plan mapping
- `paper/paper.tex`             — IEEEtran draft
- `figures/methodology_combined.{docx,pdf}` — combined methodology document
