# MDIE — Modification-Invariant Face Recognition

*A research project on making face recognition robust to real-world face modifications — masks, glasses, plastic surgery, aging, low light, adversarial perturbations.*

> **One sentence.** When ArcFace meets a masked face on LFW its verification AUC
> drops by **0.198**; the method in this repo — MDIE — drops by **0.027**,
> roughly **7× more robust**, with no extra parameters at inference, trained
> end-to-end on a single laptop GPU (RTX 3050, 4 GB).

---

## 1. The approach

### 1.1 The problem
Production face recognition models (FaceNet, ArcFace, CosFace, MobileFaceNet)
work beautifully on clean, frontal, well-lit photographs. They start to break
the moment the world intervenes — a surgical mask, a new pair of glasses, a
post-surgery jawline, an aging gap, or a 4-pixel adversarial nudge.

The *proposed execution plan* this work serves asks us to first quantify that
failure, then design a model that does not have it. We do both, on standard
public data, in one runnable repository.

### 1.2 The idea — MDIE in three lines
- Keep the deployed inference path *exactly* the same as ArcFace
  (IR-50 backbone, 512-D embedding, cosine compare).
- At **training time**, attach two extra signals:
  - **AMD** — *Adversarial Modification Disentanglement.* A small classifier
    head guesses which modification is on the input (mask / glasses / FGSM…).
    A **Gradient-Reversal Layer** sits between it and the encoder, so the
    encoder is pushed to make the modification *unreadable*.
  - **ICCL** — *Identity-Consistency Contrastive Loss.* For every clean
    image we synthesise the *same* identity with one of nine modifications
    and pull their embeddings together, while pushing negatives apart with
    **same-modification negatives weighted 2×** (so the model cannot cheat
    by using the modification itself as the similarity signal).

AMD removes the modification cue; ICCL guarantees identity survives that
removal. Neither alone works — the ablation table in the paper shows this.

### 1.3 Why this is publishable
1. A **fair failure-mode benchmark** across 9 modifications on a standard
   dataset (LFW, identity-disjoint split, deterministic seeds).
2. Two **new training-only losses** (AMD via GRL on a modification-id head,
   ICCL with modification-aware hard negatives) that together cut worst-case
   AUC drop by **~7×** vs ArcFace at the same inference cost.
3. **Honest ablations**, including a published negative result on a
   region-attention transformer (RATA) that needs more data than a 4 GB
   laptop GPU can give it.
4. A **strong external baseline** — InsightFace's production IR-50 trained
   on WebFace12M (`w600k_r50`) — auto-downloaded by the eval harness so
   the headline comparison is against a real production model, not just our
   own re-trained baselines.

### 1.4 Headline number

| Model                       | AUC clean | AUC worst-mod | Δ (lower = more robust) |
|-----------------------------|-----------|---------------|--------------------------|
| FaceNet                     | 0.672     | 0.533         | 0.139                    |
| MobileFaceNet               | 0.904     | 0.733         | 0.171                    |
| CosFace                     | 0.903     | 0.730         | 0.173                    |
| ArcFace                     | 0.902     | 0.704         | 0.198                    |
| **MDIE (ours)**             | **0.858** | **0.831**     | **0.027 (≈ 7× lower)**   |

Worst cell — `disguise_mask`: ArcFace 0.704 → **MDIE 0.831 (+12.7 pp)**.

---

## 2. How to run it

### 2.1 Install

```cmd
pip install -r requirements.txt
```

Python 3.10+. A CUDA GPU is recommended for training; everything else
(downloads, eval, paper figures, methodology DOCX) works on CPU.

### 2.2 Reproduce the full pipeline

```cmd
research_v2\run_full_v3.bat
```

This runs, in order:

| Phase | What it does                                                | Needs GPU? |
|-------|-------------------------------------------------------------|------------|
| 0     | Pre-fetches the InsightFace `w600k_r50` baseline (~174 MB)  | no         |
| A     | Seeds the IR-50 backbone from pretrained weights (best-effort) | no      |
| B     | Materialises the real benchmarks (MFR2 / CALFW / AgeDB-30)  | no         |
| 1     | Trains the 4 SOTA baselines (FaceNet, ArcFace, CosFace, MobileFaceNet) | yes |
| 2     | Trains MDIE + the ablation variants                         | yes        |
| C     | Runs the unified real-benchmark eval (all models × all benches) | inf only |
| E     | Generates the Grad-CAM grid and the CAM-IoU eye-region bar  | inf only   |

The whole pipeline takes roughly four hours on an RTX 3050.

### 2.3 Run pieces individually

```cmd
REM Stage 1: baselines + failure modes
python -m research_v2.src.run_stage1 --epochs 50 --batch 32

REM Stage 2: MDIE + ablation
python -m research_v2.src.run_stage2 --epochs 40 --batch 24 --ablation

REM Real-benchmark eval (auto-downloads InsightFace w600k_r50 on first use)
python -m research_v2.src.eval.run_real_benchmarks

REM Grad-CAM interpretability grid
python -m research_v2.src.eval.gradcam

REM Quick smoke test (~10 min, CPU-friendly)
python -m research_v2.src.run_stage1 --quick
python -m research_v2.src.run_stage2 --quick --ablation
```

### 2.3b Preflight + A100 launch

```cmd
REM 8-check preflight (versions, dirs, disk, dataset, fwd+bwd for all models)
python -m research_v2.src.preflight
```

A100 / dedicated-GPU long run — trainers auto-detect bf16, TF32, channels-last:
```bash
# convenience launcher (calls preflight + stage1 + stage2 + real benches + gradcam)
bash scripts/launch_a100.sh

# CDAC PARAM (PARAM Siddhi-AI / PAAM / Rudra) — full SLURM workflow:
#   bash hpc/env_setup.sh           # once: modules + venv + pip
#   bash hpc/stage_datasets.sh      # once: download benches to $SCRATCH
#   sbatch hpc/slurm_quick.sh       # 4-epoch smoke
#   sbatch hpc/slurm_full_pipeline.sh
# See hpc/README.md for full details.

# or invoke a single phase directly:
python -m research_v2.src.run_stage2 \
    --epochs 50 --batch 256 --lr 2e-3 --workers 8 \
    --channels-last --balanced-sampler \
    --classes-per-batch 32 --samples-per-class 8 \
    --val-pairs 1000 --compile --ablation
```
Optional flags worth knowing:
- `--compile` wraps the model with `torch.compile(mode="reduce-overhead")`,
  silently falls back to eager if Triton isn't available.
- `--balanced-sampler` swaps the random `DataLoader` for
  `IdentityBalancedSampler` (classes×samples-per-class) so ICCL gets real
  in-batch positives and modification-aware hard negatives.
- `--val-pairs N` enables a per-epoch verification-AUC check on a disjoint
  pair pool and writes a `<name>_best_auc.pt` snapshot whenever it improves.

Each variant writes 3 checkpoints: `<name>.pt` (best train loss),
`<name>_best_auc.pt` (best per-epoch val-AUC), `<name>_last.pt` (resumable via `--resume`).
A run manifest is written to `research_v2/results/run_<timestamp>_stage{1,2}.json`.
Per-epoch CSV (`research_v2/results/history_<name>.csv`) now also logs
`imgs_per_sec` so you can spot a stalled DataLoader within one epoch.

#### Re-evaluate a single checkpoint without retraining
```bash
python -m research_v2.src.eval.eval_from_ckpt \
    --ckpt research_v2/checkpoints/mdie_full_best_auc.pt --model mdie
```

### 2.4 Gated datasets (optional)

Two benchmarks need agreements / sign-up — they are silently skipped if
their env vars are unset:

```cmd
set IIITD_ROOT=D:\datasets\iiitd_plastic_surgery
set IJBC_ROOT=D:\datasets\ijbc
```

Layout expected by the loaders is documented in
`research_v2/src/data/benchmarks/{iiitd_surgery,ijbc_occ}.py`.

### 2.5 Rebuild the methodology document

```cmd
python -m research_v2.src.paper.build_methodology_docx
```

Produces `research_v2/paper/methodology_combined.docx` with the simple
diagram, full training graph, and first-person prose description.

---

## 3. What lives where

```
aicte/
├── README.md                       ← this file (approach + how-to)
├── requirements.txt
├── execution_plan (1).docx         ← sponsor brief, kept verbatim
└── research_v2/
    ├── README.md                   ← short technical entry point
    ├── run_full_v3.bat             ← one-shot reproduce
    ├── src/
    │   ├── config.py               ← paths, seeds, hyperparameters
    │   ├── pretrained.py           ← IR-50 seed loader (HF + local + insightface)
    │   ├── run_stage1.py           ← entry: baselines
    │   ├── run_stage2.py           ← entry: MDIE + ablation
    │   ├── data/
    │   │   ├── lfw.py, pairs.py, torch_dataset.py, modifications.py
    │   │   └── benchmarks/         ← MFR2, CALFW, AgeDB-30, IIITD, IJB-C loaders
    │   ├── models/
    │   │   ├── backbones.py        ← face.evoLVe IR-50, MobileFaceNet, InceptionResnetV1
    │   │   ├── heads.py            ← ArcFace, CosFace, triplet
    │   │   ├── losses.py
    │   │   └── iresnet.py          ← InsightFace IResNet50 (loads w600k_r50.pth)
    │   ├── baselines/              ← Stage-1 trainer + 4 baseline wrappers
    │   ├── novel/
    │   │   ├── mdie.py             ← MDIE model (backbone + AMD head + ICCL projector)
    │   │   ├── region_attention.py ← RATA (ablation; honest negative result)
    │   │   └── train_mdie.py
    │   ├── train/pretrain_backbone.py  ← --use-pretrained / --from-scratch
    │   ├── eval/
    │   │   ├── embeddings.py, metrics.py
    │   │   ├── occlusion_sensitivity.py
    │   │   ├── gradcam.py          ← grid + CAM-IoU eye-region bar
    │   │   └── run_real_benchmarks.py  ← unified harness
    │   └── paper/
    │       ├── figures.py, latex_tables.py
    │       ├── methodology_diagram.py, methodology_simple.py
    │       └── build_methodology_{docx,pdf}.py
    ├── paper/
    │   ├── paper.tex               ← IEEEtran draft (builds with pdflatex)
    │   ├── METHODOLOGY.md          ← detailed methodology
    │   ├── METHODOLOGY_SIMPLE.md   ← simplified four-box methodology
    │   └── README.md
    ├── checkpoints/                ← trained .pt files (+ drop-in instructions)
    ├── datasets_cache/             ← LFW + real benchmarks (auto-downloaded)
    ├── figures/                    ← publication-ready PDFs & PNGs
    └── results/                    ← JSON / CSV / .tex tables
```

---

## 4. Datasets

| Dataset                | Use                  | How obtained                                    |
|------------------------|----------------------|-------------------------------------------------|
| LFW                    | Train + ablation     | auto-downloaded by `src/data/lfw.py`            |
| MFR2 (masks)           | Real-bench eval      | drop `.bin` in `datasets_cache/benchmarks/mfr2` |
| CALFW (cross-age)      | Real-bench eval      | drop `.bin` in `datasets_cache/benchmarks/calfw`|
| AgeDB-30 (aging)       | Real-bench eval      | drop `.bin` in `datasets_cache/benchmarks/agedb30` |
| IIITD Plastic Surgery  | Real-bench eval      | gated — set `IIITD_ROOT`                        |
| IJB-C (occlusion)      | Real-bench eval      | gated — set `IJBC_ROOT`                         |
| InsightFace `w600k_r50`| External baseline    | auto-downloaded from HuggingFace (`Icar/buffalo_l-torch`) |

The InsightFace `.bin` format is what the standard
`face.evoLVe / insightface / arcface_torch` releases ship. The loader in
`src/data/benchmarks/_bin_parser.py` handles it.

---

## 5. Algorithms used

- **Backbones:** IR-50 (face.evoLVe), MobileFaceNet, InceptionResnetV1; plus
  IResNet50 for the InsightFace baseline.
- **Identity supervision:** ArcFace (m = 0.5, s = 64), CosFace, triplet (FaceNet).
- **Novel losses:**
  - AMD — Gradient-Reversal Layer (Ganin & Lempitsky 2015) into a
    modification-id MLP head.
  - ICCL — InfoNCE-style contrastive loss with **modification-aware
    hard-negative mining** (same-mod negatives weighted 2×).
- **Modification engine** (deterministic, 9 modes): surgery-style warp
  (nose / jaw), mask, glasses, random occlusion, aging filter, low-light
  γ + noise, FGSM (ε = 4/255).
- **Training stack:** PyTorch ≥ 2.1, AMP mixed precision, AdamW + cosine
  LR with warmup, MTCNN-style 5-point alignment → 112×112 crop.
- **Evaluation:** cosine-similarity verification → ROC / AUC, EER,
  TAR @ fixed FAR, occlusion-sensitivity heatmaps, Grad-CAM + CAM-IoU.

---

## 6. Mapping to the proposed execution plan

| Plan stage                                    | This repo                                |
|-----------------------------------------------|-------------------------------------------|
| **Stage 1** — quantify SOTA failure modes     | `run_stage1.py` + Phase C real benches    |
| **Stage 3A** — lightweight edge backbone      | IR-50 + ArcFace head (deployable footprint) |
| **Stages 3C / 3D** — region-stable rep + embedding geometry | AMD + ICCL (training-only)      |
| **Stage 4** — laptop demonstrator             | full pipeline on RTX 3050 4 GB            |
| **Stages 2 / 3B / 5–7** (GAN aug, depth+IR, edge / federated / field) | architecture plug-in points preserved; no inference-time changes needed |

Detailed mapping and longer prose live in
`research_v2/paper/METHODOLOGY.md` and `research_v2/paper/paper.tex`.

---

## 7. Status

| Item                                          | State    |
|-----------------------------------------------|----------|
| Stage 1 baselines (4 SOTA models, LFW × 9 mods) | done   |
| Stage 2 MDIE + ablations                      | done     |
| Real-benchmark loaders (MFR2 / CALFW / AgeDB-30 / IIITD / IJB-C) | done (awaiting data drops on gated sets) |
| InsightFace `w600k_r50` baseline integration  | done     |
| Grad-CAM grid + CAM-IoU                       | code done; needs trained MDIE + ArcFace checkpoints |
| Pretrained IR-50 seed for MDIE init           | blocked on a public face.evoLVe-layout torch checkpoint; workaround is `--from-scratch` on a real GPU server |
| IEEE-conference paper draft                   | done; updated as v3 results land          |

---

## 8. Citation

If this work is useful to you, please cite the paper draft in
`research_v2/paper/paper.tex`.
