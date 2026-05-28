# MDIE — Methodology Summary

> Companion text to **`figures/methodology.png`** / **`methodology.pdf`**.
>
> *Voice: first-person. Naming: "research project" / "proposed execution plan".
> v3 of this document — covers the real-benchmark and interpretability
> upgrades on top of the original synthetic-LFW protocol.*

---

## 1. The one-paragraph version

MDIE keeps the standard ArcFace face-recognition pipeline and adds two
training-only objectives that, together, make the embedding *invariant
to how the photo has been corrupted* without ever telling the model — at
inference time — that a corruption happened. The first new objective
(**AMD**, Adversarial Modification Disentanglement) attaches a small
classifier to the embedding that tries to predict *which* corruption
was applied, with a gradient-reversal layer in front of it; the encoder
is therefore trained to *erase* the corruption signal from the
embedding. The second (**ICCL**, Identity-Consistency Contrastive
Loss) explicitly pulls together the embeddings of (clean, corrupted)
versions of the same person while pushing apart
same-corruption-different-person hard negatives. The two losses are
added on top of the standard ArcFace identity loss; nothing else about
the model changes. At test time the inference path is bit-for-bit
identical to ArcFace — same backbone, same 512-D embedding, same
cosine similarity, **no modification label required**.

---

## 2. How it works, step by step

Refer to the diagram (left → right):

### Stage A — Pair construction (blue boxes)
For every training identity $y_i$ I have a clean image $x_i$. A
**modification engine** $\mathcal{M}_{m_i}$ samples a corruption type
$m_i$ uniformly from nine choices — clean control, surgical nasal warp,
surgical jaw warp, opaque glasses, mouth-and-nose mask, random rectangular
occlusion, age proxy, low-light γ-shift, and FGSM adversarial — and
produces $\tilde{x}_i$. Both images go
into the same minibatch.

### Stage B — Shared backbone (green box)
A standard **IR-50** convolutional backbone $\mathbf{f}_\theta$ embeds
both images into 512-D vectors. The two passes share weights exactly —
there is one network, not two. I therefore obtain
$z_i = \mathbf{f}_\theta(x_i)$ and
$\tilde{z}_i = \mathbf{f}_\theta(\tilde{x}_i)$.

### Stage C — Three heads, three signals (right column)

1. **Identity head $H_{\text{id}}$ (orange — standard ArcFace).**
   ArcFace additive angular margin ($m=0.5,\,s=64$) on $z_i$ for the
   cross-entropy identity loss $\mathcal{L}_{\text{arc}}$.

2. **AMD head $H_{\text{mod}}$ (red — *novel*).**
   A 2-layer MLP that takes $z_i$ (and $\tilde{z}_i$) through a
   **Gradient-Reversal Layer (GRL)** and predicts the modification class
   $m_i$. The loss $\mathcal{L}_{\text{amd}}$ is ordinary cross-entropy
   on $m_i$, but the GRL reverses and rescales the gradient by
   $-\lambda_{\text{amd}}$ before it reaches the encoder. The encoder is
   therefore optimised to *fool* the classifier — i.e. to *remove*
   modification information from the embedding. This is the same
   technique as Ganin & Lempitsky's domain-adversarial training, applied
   here for the first time (to my knowledge) to face modifications
   rather than dataset domains.

3. **ICCL loss (violet — *novel*).**
   For each $i$, $\tilde{z}_i$ is the *only* positive of $z_i$ in the
   batch; every other identity is a negative. I compute a
   temperature-softmaxed log-likelihood of the positive,
   $-\log\frac{\exp(z_i^\top\tilde{z}_i/\tau)}
                {\sum_j w_{ij}\exp(z_i^\top\tilde{z}_j/\tau)}$,
   with one twist: negatives that share the same modification get a
   double weight ($w_{ij}=2$ when $m_j = m_i$). The intuition is sharp:
   *the hardest negative for "person A in a mask" is "person B in a
   mask"* — superficial shape cues line up. Doubling that weight forces
   the model to look past the disguise.

### Stage D — Total objective (yellow box)
$$\mathcal{L} \;=\; \mathcal{L}_{\text{arc}}
              \;+\;\lambda_{\text{iccl}}\,\mathcal{L}_{\text{iccl}}
              \;+\;\lambda_{\text{amd}}\,\mathcal{L}_{\text{amd}}^{\text{GRL}}$$
with $\lambda_{\text{iccl}} = 0.50$ and $\lambda_{\text{amd}} = 0.10$.

### Stage E — Inference (green strip at the bottom)
At test time, drop the two new heads. Feed an image through
$\mathbf{f}_\theta$, L2-normalise, compare with cosine similarity. The
verifier is **structurally indistinguishable from ArcFace** — same
parameter count at inference, same FLOPs, same I/O. **No modification
label is ever needed.**

---

## 3. Why it is novel

| # | Piece                                                              | Status before MDIE |
|---|--------------------------------------------------------------------|--------------------|
| 1 | Adversarial removal of *modification class* from a face embedding  | Adversarial training had been used to remove *dataset domains* (DANN) and *demographic attributes* (debiasing). Removing **modification type** — mask, glasses, occlusion, low light, FGSM — has not been published to my knowledge. |
| 2 | Modification-aware hard-negative weighting in a contrastive loss   | Supervised contrastive losses treat negatives uniformly or mine by embedding hardness. Mining by *which corruption matches the anchor's corruption* is new and does most of the heavy lifting in the ablation. |
| 3 | Joint training of both losses on a single shared backbone          | Mask-invariant FR methods either retrain on masked data, do feature inpainting, or use multi-stream networks that need a mask-detector at inference. MDIE needs neither. |

In simple terms: the literature previously split the problem — either
(i) generate more masked / disguised data and re-train, or (ii) use a
modification-detector to *route* to a specialised model. MDIE shows that
a single network can be **trained to be modification-blind** by
combining a known idea from a different field (adversarial domain
confusion) with a new contrastive-mining heuristic, and that the
combination produces robustness gains that neither component achieves
alone.

---

## 4. Why it is promising

### 4.1 Controlled-LFW protocol (my synthetic 9-modification benchmark)

LFW, identity-disjoint 80 / 20 split, 217 identities, 6 000 verification
pairs per modification, 50 / 40 epoch training on one RTX 3050 4 GB.

| Property                                          | ArcFace (best baseline) | **MDIE (ours)**       |
|---------------------------------------------------|--------------------------|------------------------|
| AUC on clean images                               | 0.902                    | 0.858 (−4 pp)          |
| AUC on the worst modification (`disguise_mask`)   | 0.704                    | **0.831 (+12.7 pp)**   |
| **Worst-case AUC drop (clean − worst-mod)**       | 0.198                    | **0.027 (≈ 7× lower)** |
| Pooled AUC across all 9 modifications             | 0.861                    | 0.850 (−1 pp)          |
| Modification label needed at inference?           | n/a                      | **No**                 |
| Inference cost vs ArcFace                         | 1×                       | **1×** (identical)     |
| Trainable on a 4 GB laptop GPU?                   | Yes                      | **Yes**                |

### 4.2 Real-benchmark protocol (v3 — `eval/run_real_benchmarks.py`)

Five standard public benchmarks, plus a strong external baseline:

| Benchmark                          | What it stresses          | How obtained |
|------------------------------------|---------------------------|--------------|
| **MFR2**                           | Real masked faces         | drop `.bin` in `datasets_cache/benchmarks/mfr2` |
| **CALFW**                          | Cross-age (10-fold)       | drop `.bin` in `datasets_cache/benchmarks/calfw` |
| **AgeDB-30**                       | Aging gap (10-fold)       | drop `.bin` in `datasets_cache/benchmarks/agedb30` |
| **IIITD Plastic Surgery**          | Post-surgical identity    | gated — set `IIITD_ROOT` env var |
| **IJB-C** (occlusion protocol)     | Wild occlusion at scale   | gated — set `IJBC_ROOT` env var |
| **InsightFace `w600k_r50`** (baseline) | Production IR-50 trained on WebFace12M | auto-downloaded from HuggingFace (`Icar/buffalo_l-torch`) |

The headline comparison in v3 is **MDIE vs InsightFace's production
`w600k_r50`** under degradation. Winning against a baseline I trained
on a small dataset is easy to dismiss; winning against a production
model trained on 12 million faces is the right standard.

### 4.3 Interpretability (v3 — `eval/gradcam.py`)

For every (model × benchmark) cell I compute a **Grad-CAM grid**
showing where each model is looking on representative pairs, plus a
**CAM-IoU bar chart** measuring the overlap between the CAM and the
periocular eye-region box (the band that survives a surgical mask).
The hypothesis the paper tests is that MDIE keeps a high
CAM-IoU on the eye region across modifications while ArcFace's CAM
drifts toward the now-occluded lower face — the geometric reason masks
hurt ArcFace so much.

Three reasons MDIE is more than a one-off result:

1. **The novelty is small, the effect is large.** AMD is one extra MLP
   plus a gradient-reversal layer; ICCL is one extra loss term.
   Combined they cut worst-case AUC drop by **~7×** with a sub-2-pp
   pooled-AUC cost.

2. **The trade-off is visibly favourable.** A 1 pp drop in average
   accuracy for a 17 pp drop in worst-case failure is a trivial sell
   for anything safety-critical (border control, masked authentication,
   surveillance after a haircut).

3. **It scales down, not just up.** MDIE trains end-to-end on a 4 GB
   laptop GPU in about four hours. That makes the result reproducible
   by undergraduates, hobbyists and small labs.

---

## 5. Honest caveats

- **RATA** (the third originally-proposed component) does not work in
  my small-data regime (173 training identities). I report it as a
  candid negative result; it almost certainly needs MS1MV3-scale
  training to show its design intent, which does not fit on a 4 GB card.
- **Synthetic modifications.** My nine corruptions are
  protocol-faithful but synthetic. The v3 evaluation harness adds the
  real-data benchmarks above to address this directly; absolute numbers
  will shift, but the *worst-case AUC drop* metric should remain
  faithful.
- **Pretrained seed.** I could not find a public PyTorch IR-50
  checkpoint whose key layout matches face.evoLVe's IR-50, so MDIE is
  currently trained from random init. For the eval comparison I use the
  InsightFace `w600k_r50` baseline (a different architecture,
  `iresnet50`) which I do download as a production reference point. A
  matched-layout pretrained seed remains future work.
- **Statistical power.** 6 000 pairs per cell give tight ROCs but do
  not probe the very-low-FAR tail (FAR ≤ 10⁻⁴). The protocol scales
  trivially; only the storage budget changes.

---

## 6. Where to read more

- **Diagrams:** `figures/methodology.png` (full graph),
  `figures/methodology_simple.png` (four boxes).
- **Combined DOCX / PDF:** `figures/methodology_combined.{docx,pdf}` —
  rebuild with `python -m research_v2.src.paper.build_methodology_docx`.
- **Paper draft:** `paper/paper.tex` (IEEEtran, full sections).
- **Source code for every step in the diagram:**
  - Modification engine: `src/data/modifications.py`
  - IR-50 backbone:       `src/models/backbones.py`
  - ArcFace head:         `src/models/heads.py`
  - **AMD head + GRL:**   `src/novel/mdie.py`
  - **ICCL loss:**        `src/novel/train_mdie.py`
    (`identity_consistency_contrastive_loss`)
  - **Real-benchmark eval harness:** `src/eval/run_real_benchmarks.py`
  - **Grad-CAM grid + CAM-IoU:**      `src/eval/gradcam.py`
  - **InsightFace IResNet50 baseline:** `src/models/iresnet.py`
- **Numbers behind the tables above:** `results/stage{1,2}_metrics.json`
  (controlled LFW) and `results/real_benchmarks.{csv,json}` (v3 real
  benchmarks, once the harness runs against dropped-in data).
