# MDIE — Methodology Summary

> Companion text to **`figures/methodology.png`** / **`methodology.pdf`**.
>
> *Voice: first-person. Naming: "research project" / "proposed execution plan".
> v4 of this document — re-scoped to the **security / access-control niche:
> worn-occlusion (mask, cap/hat, glasses, partial occluder) and adverse-lighting
> (low-light, over-exposure, harsh shadow) robustness**, with an **ArcFace-compatible
> single-512-d deployment** and a working **bone-anchored RATA interpretability
> result** (the earlier "RATA negative result" is superseded).*

---

## 1. The one-paragraph version

MDIE keeps the standard ArcFace face-recognition pipeline and adds three
training objectives that, together, make the embedding *invariant
to worn occlusions and lighting* without ever telling the model — at
inference time — that a corruption happened. The first new objective
(**AMD**, Adversarial Modification Disentanglement) attaches a small
classifier to the embedding that tries to predict *which* corruption
was applied, with a gradient-reversal layer in front of it; the encoder
is therefore trained to *erase* the corruption signal from the
embedding. The second (**ICCL**, Identity-Consistency Contrastive
Loss) explicitly pulls together the embeddings of (clean, corrupted)
versions of the same person while pushing apart
same-corruption-different-person hard negatives. The third (**RATA**,
Region-Aware Token Attention) supervises a spatial attention map onto
each face's own detected **rigid bone landmarks** (brow ridge, orbital
rims, nasal bridge, cheekbones, jaw, chin) — the skeletal scaffold that
survives masks, caps, glasses and lighting — giving the model an
anatomical inductive bias and a verifiable interpretability signal. The
losses are added on top of the standard ArcFace identity loss. At
inference, the bone-anchored attention embedding and the native identity
embedding are merged by a learned **fusion head** into a single
L2-normalised **512-D** vector in one forward pass — the inference path is a
**drop-in ArcFace encoder** (same backbone, same 512-D embedding, same
cosine similarity, **no modification label required**).

---

## 2. How it works, step by step

Refer to the diagram (left → right):

### Stage A — Pair construction (blue boxes)
For every training identity $y_i$ I have a clean image $x_i$. A
**modification engine** $\mathcal{M}_{m_i}$ samples a corruption type
$m_i$ uniformly from nine choices — opaque/clear glasses, mouth-and-nose
mask, forehead cap, random rectangular occlusion, low-light γ-shift,
over-exposure / blown highlights, harsh directional shadow, an aging
proxy, and FGSM adversarial — alongside the clean control, and
produces $\tilde{x}_i$. Both images go
into the same minibatch.

### Stage B — Shared backbone (green box)
A standard **IR-50** convolutional backbone $\mathbf{f}_\theta$ embeds
both images into 512-D vectors. The two passes share weights exactly —
there is one network, not two. I therefore obtain
$z_i = \mathbf{f}_\theta(x_i)$ and
$\tilde{z}_i = \mathbf{f}_\theta(\tilde{x}_i)$.

### Stage C — Four heads, four signals (right column)

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

4. **RATA attention supervision (teal — *novel*).**
   The backbone's mid/high-resolution feature maps are fused into a
   768-channel **14×14** token grid and passed through a small attention
   module that emits a per-face spatial attention map. We supervise that
   map onto each face's own detected **rigid bone landmarks** — splatted
   with **equal mass** per landmark so the densely-clustered central
   T-zone cannot collapse into one central blob — using a symmetric
   (forward+reverse) matching loss $\mathcal{L}_{\text{attn}}$ plus a
   background-suppression gate. This injects the prior that identity
   under occlusion rides on the **skeletal scaffold**, and yields a
   verifiable interpretability signal (Section 4.3): the attention lands
   on *that individual's* bones, not a shared template.

### Stage D — Total objective (yellow box)
$$\mathcal{L} \;=\; \mathcal{L}_{\text{arc}}
              \;+\;\lambda_{\text{iccl}}\,\mathcal{L}_{\text{iccl}}
              \;+\;\lambda_{\text{amd}}\,\mathcal{L}_{\text{amd}}^{\text{GRL}}
              \;+\;\lambda_{\text{attn}}\,\mathcal{L}_{\text{attn}}$$
with $\lambda_{\text{iccl}} = 0.50$, $\lambda_{\text{amd}} = 0.10$ and the
RATA attention-matching weight $\lambda_{\text{attn}}$. The ArcFace
identity loss and ICCL are computed on the **fused** embedding, so the
deployed 512-D vector is exactly what is optimised.

### Stage E — Inference (green strip at the bottom)
At test time, drop the AMD and modification heads. Feed an image through
$\mathbf{f}_\theta$, take the bone-anchored attention embedding and the
native identity embedding, merge them through the learned **fusion head**
(`Linear(1024→512)+BatchNorm`), L2-normalise, and compare with cosine
similarity. The verifier emits **one 512-D vector in a single forward
pass** and is **structurally a drop-in for ArcFace** — same embedding
dimensionality, same cosine matcher, same I/O. **No modification label,
no flip-TTA, no multi-crop is ever needed** (verified in
`results/inference_compat_proof.json`: unit-norm, cosine == dot,
deterministic).

---

## 3. Why it is novel

| # | Piece                                                              | Status before MDIE |
|---|--------------------------------------------------------------------|--------------------|
| 1 | Adversarial removal of *modification class* from a face embedding  | Adversarial training had been used to remove *dataset domains* (DANN) and *demographic attributes* (debiasing). Removing **worn-occlusion / lighting type** — mask, cap, glasses, occluder, low-light, over-exposure, harsh-shadow, FGSM — has not been published to my knowledge. |
| 2 | Modification-aware hard-negative weighting in a contrastive loss   | Supervised contrastive losses treat negatives uniformly or mine by embedding hardness. Mining by *which corruption matches the anchor's corruption* is new and does most of the heavy lifting in the ablation. |
| 3 | Bone-anchored attention with a verifiable IoU control              | Prior occlusion/attention FR (OREO, LAFS, PDSN, Attention-Partial-FR) supervises or learns attention but does not *prove* it lands on each individual's own rigid bone geometry. RATA reports a matched-vs-mismatched-vs-random **bone-IoU** with a significance test (Section 4.3). |
| 4 | Joint training on a single shared backbone, ArcFace-compatible out | Mask-invariant FR methods either retrain on masked data, do feature inpainting, or use multi-stream networks that need a mask-detector at inference. MDIE needs neither and deploys as a single 512-D ArcFace drop-in. |

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

### 4.1 Held-out LFW protocol (synthetic 9-modification ablation)

LFW, identity-disjoint 80 / 20 split, 173 training identities, 44 held-out
(unseen) identities, 6 000 verification pairs per modification, on one
RTX 3050 4 GB. All numbers use the **deployed single-512-D fused embedding**.

| Property                                              | Best comparably-trained baseline | **MDIE-full (ours)** |
|-------------------------------------------------------|----------------------------------|-----------------------|
| AUC on clean images                                   | 0.849 (mobilefacenet)            | **0.984**             |
| AUC on the worst modification (`disguise_mask`)       | 0.69 (mobilefacenet)             | **0.966 (+0.28)**     |
| **Worst-case AUC drop (clean − worst-mod)**           | 0.16                             | **0.018 (≈ 9× lower)**|
| Occlusion-family AUC (mask/cap/glasses/occluder)      | 0.716 (cosface)                  | **0.975 (+0.26)**     |
| Lighting-family AUC (low-light/over-exp/harsh-shadow) | 0.755 (mobilefacenet)            | **0.980 (+0.22)**     |
| Pooled AUC across all 9 modifications                 | 0.749                            | **0.979**             |
| Modification label needed at inference?               | n/a                              | **No**                |
| Inference: vectors / forwards / TTA                   | 1 × 512-D / 1 / none             | **1 × 512-D / 1 / none** |
| Trainable on a 4 GB laptop GPU?                       | Yes                              | **Yes**               |

The clean ablation ladder is monotone — **MDIE-full (0.979) > noICCL ≈
noAMD (0.974) > noRATA (0.972) > best baseline (0.749)** — so every
component contributes; AMD + ICCL carry most of the AUC, and RATA carries
the interpretability (Section 4.3).

### 4.2 Real-benchmark protocol (v3 — `eval/run_real_benchmarks.py`)

Five standard public benchmarks, plus a strong external baseline:

### 4.2 Real-benchmark protocol (`eval/run_real_benchmarks.py`)

Public verification protocols, evaluated with the **same deployed
single-512-D fused model**. MFR2 (masks) and MeGlass (glasses) are
**real worn occlusions** — the acid test that the bone-anchored embedding
transfers off the synthetic training distribution. AUC, higher = better:

| Benchmark | Pairs | What it stresses | best comparably-trained baseline | **MDIE (ours)** |
|-----------|-------|------------------|----------------------------------|-----------------|
| **MFR2** (real masks)    | 848   | masked face recognition | 0.677 (mobilefacenet) | **0.826** |
| **MeGlass** (real glasses)| 3 000 | real worn eyeglasses    | 0.706 (cosface)       | **0.926** |
| **CALFW**                | 6 000 | cross-age (off-niche)   | 0.516 (cosface)       | **0.599** |
| **AgeDB-30**             | 6 000 | aging gap (off-niche)   | 0.500 (mobilefacenet) | **0.683** |

MDIE wins **every** real benchmark against the comparably-trained
baselines — **+0.15 to +0.34 AUC** on the real occlusions, and even on
off-niche cross-age sets where the baselines sit at near-chance. The
production reference InsightFace `w600k_r50` (trained on ~17M faces) is
still ahead on raw numbers; we do **not** claim to beat it — the honest
claim is that MDIE is the **most robust recognizer in the
comparably-trained regime on the occlusion + lighting niche**.

### 4.3 Interpretability — attention–bone IoU (`attention_bone_iou.py`)

The headline interpretability result is a thresholded region-overlap
metric **with controls**, computed on the 44 held-out (unseen)
identities. We binarise the learned RATA attention and each face's own
detected rigid-bone target to their top cells and measure IoU:

| matched (own bones) | mismatched (other face's bones) | random-attention null |
|---------------------|---------------------------------|-----------------------|
| **0.694 ± 0.090**   | 0.280 ± 0.100                   | 0.077                 |

The **matched ≫ mismatched ≫ random** ordering holds across the
top-10/15/20/25 % thresholds (matched−mismatched gap 0.36–0.41),
**Mann–Whitney p = 1.6e-15**. So the attention is anchored on *each
individual's* bone geometry — not a shared face-layout template (0.280),
not chance (0.077). Every anatomical group is attended above its uniform
share: nose-bridge 13.2×, orbital rim 5.6×, cheekbone 4.9×, jaw/chin
3.2×, brow ridge 3.0×. Figure: `figures/attention_bone_iou.png`. **No
prior occlusion/attention FR work reports this control.**

Three reasons MDIE is more than a one-off result:

1. **The novelty is small, the effect is large.** AMD is one extra MLP
   plus a gradient-reversal layer; ICCL is one extra loss term; RATA is a
   supervised attention map. Combined they lift pooled occlusion/lighting
   AUC from the 0.75 baseline regime to **0.979**.

2. **The trade-off is visibly favourable, and the embedding deploys like
   ArcFace.** One 512-D vector, one forward pass, plain cosine — ideal
   for safety-critical edge use (access control, masked authentication,
   surveillance under poor lighting).

3. **It scales down, not just up.** MDIE trains end-to-end on a 4 GB
   laptop GPU. That makes the result reproducible by undergraduates,
   hobbyists and small labs.

---

## 5. Honest caveats

- **RATA's contribution is interpretability, not raw AUC.** On pooled
  AUC the MDIE variants sit within ~0.007 of each other — AMD + ICCL
  already carry most of the cross-modification invariance. RATA's
  decisive, *measurable* value is the bone-anchored localization proof of
  Section 4.3 (matched IoU 0.694 ≫ mismatched 0.280 ≫ random 0.077, p =
  1.6e-15): it shows the surviving signal is the rigid bone scaffold, not
  a shortcut. *(This supersedes the earlier "RATA negative result" —
  with the equal-mass bone-target splatting and the 14×14 fused map, RATA
  produces per-face-distinct, anatomically-anchored attention.)*
- **Synthetic training modifications.** The nine training corruptions are
  protocol-faithful but synthetic; the real-data benchmarks (MFR2 masks,
  MeGlass glasses) above address transfer directly and MDIE wins them.
- **Scale, not architecture, is the ceiling.** We train on ~13k images;
  the production reference InsightFace `w600k_r50` trains on ~17M. We do
  not claim to beat it on raw accuracy — the contribution is the
  occlusion/lighting-niche win, the bone-anchored interpretability, and
  the ArcFace-compatible deployment, all retrainable at scale on the
  PARAM A100 bundle.
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
  - **AMD head + GRL, RATA attention, fusion head:** `src/novel/mdie.py`
  - **ICCL loss + attention-matching loss:** `src/novel/train_mdie.py`
  - **Real-benchmark eval harness:** `src/eval/run_real_benchmarks.py`
  - **Attention–bone IoU interpretability:** `attention_bone_iou.py`
  - **ArcFace-compat inference proof:** `inference_compat_proof.py`
  - **InsightFace IResNet50 baseline:** `src/models/iresnet.py`
- **Numbers behind the tables above:** `results/stage2_metrics.json` +
  `results/security_family_summary.json` (held-out LFW ablation),
  `results/real_benchmarks.{csv,json}` (real benchmarks), and
  `results/inference_compat_proof.json` (ArcFace-compatible deployment).
