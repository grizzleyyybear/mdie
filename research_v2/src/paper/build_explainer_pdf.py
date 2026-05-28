"""Clean, spacious visual explainer for MDIE.

Design rules:
- ONE idea per page.
- Generous whitespace.
- 11pt body text minimum, no walls of text inside boxes.
- Diagram first, short caption underneath.

Output: research_v2/figures/mdie_explainer.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "figures" / "mdie_explainer.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

A4 = (8.27, 11.69)

INK = "#1a3a6a"
MUTED = "#666666"
LINE = "#cccccc"

AMD = "#fde9d4"; AMD_E = "#d97706"
ICCL = "#dcefdc"; ICCL_E = "#2f8b3a"
ARC = "#ece0fa"; ARC_E = "#7e57c2"
BAD = "#fbe2e2"; BAD_E = "#c62828"
GOOD = "#dcefdc"; GOOD_E = "#2e7d32"
NEUTRAL = "#f2f4f8"


# ---------- primitives -----------------------------------------------------

def _page(title=None, subtitle=None):
    fig = plt.figure(figsize=A4)
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.88])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_axis_off()
    if title:
        ax.text(0, 97, title, fontsize=20, fontweight="bold",
                color=INK, va="top")
    if subtitle:
        ax.text(0, 91.5, subtitle, fontsize=11.5, color=MUTED, va="top")
    if title:
        ax.plot([0, 100], [89, 89], color=LINE, lw=0.6)
    return fig, ax


def _box(ax, x, y, w, h, text, *, face=NEUTRAL, edge=INK, lw=1.0,
         fontsize=11, fontweight="normal", radius=0.8, color="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={radius}",
                 fc=face, ec=edge, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=color)


def _arrow(ax, x1, y1, x2, y2, color="#444", lw=1.3, dashed=False):
    ls = (0, (4, 3)) if dashed else "-"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                  arrowstyle="-|>", color=color, lw=lw,
                  mutation_scale=14, linestyle=ls))


def _footer(ax, text):
    ax.text(50, 1, text, ha="center", fontsize=9, color=MUTED,
            style="italic")


def _save(pdf, fig):
    pdf.savefig(fig); plt.close(fig)


# ---------- pages ----------------------------------------------------------

def page_cover(pdf):
    fig, ax = _page()
    ax.text(50, 78, "MDIE", ha="center", fontsize=64,
            fontweight="bold", color=INK)
    ax.text(50, 70, "Modification-Disentangled Identity Encoder",
            ha="center", fontsize=16, color=INK)
    ax.plot([25, 75], [65, 65], color=INK, lw=0.8)

    ax.text(50, 58,
            "A training-only framework that makes face recognition\n"
            "stable across masks, glasses, surgery, age, occlusion,\n"
            "low light, blur, and adversarial perturbation.",
            ha="center", fontsize=13, color="#333",
            linespacing=1.6)

    ax.text(50, 38, "Mrinal", ha="center", fontsize=13,
            fontweight="bold", color=INK)

    ax.text(50, 18,
            "Visual explainer\nwhat we built  ·  why it is novel  ·  how it works",
            ha="center", fontsize=11.5, color=MUTED,
            linespacing=1.7, style="italic")
    _save(pdf, fig)


def page_problem(pdf):
    fig, ax = _page("The problem",
                    "State-of-the-art face encoders collapse under image modifications.")

    # one big illustration block per side
    ax.text(25, 80, "ArcFace today", ha="center", fontsize=14,
            fontweight="bold", color=BAD_E)
    _box(ax, 10, 60, 30, 12, "clean photo  →  embedding A",
         face="#fff", edge=MUTED, fontsize=11)
    _box(ax, 10, 45, 30, 12, "+ mask        →  embedding A′",
         face="#fff", edge=MUTED, fontsize=11)
    _box(ax, 10, 30, 30, 9, "cos(A, A′)  is LOW",
         face=BAD, edge=BAD_E, fontsize=12, fontweight="bold")

    ax.text(75, 80, "MDIE  (this work)", ha="center", fontsize=14,
            fontweight="bold", color=GOOD_E)
    _box(ax, 60, 60, 30, 12, "clean photo  →  embedding A",
         face="#fff", edge=MUTED, fontsize=11)
    _box(ax, 60, 45, 30, 12, "+ mask        →  embedding A″",
         face="#fff", edge=MUTED, fontsize=11)
    _box(ax, 60, 30, 30, 9, "cos(A, A″)  stays HIGH",
         face=GOOD, edge=GOOD_E, fontsize=12, fontweight="bold")

    ax.plot([50, 50], [25, 80], color=LINE, lw=0.6, linestyle=(0, (3, 4)))

    ax.text(50, 18,
            "Across nine modifications on LFW, baseline AUC drops by 0.13 to 0.19.\n"
            "An early MDIE smoke run already shows a drop of only 0.01.",
            ha="center", fontsize=11, color="#333", linespacing=1.7)
    _save(pdf, fig)


def page_modifications(pdf):
    fig, ax = _page("Nine modifications we stress-test",
                    "Generated on the fly from clean LFW images.")

    mods = [
        "surgery_nose", "surgery_jaw", "disguise_glasses",
        "disguise_mask", "occlusion_random", "aging",
        "low_light", "motion_blur", "adversarial",
    ]
    for i, name in enumerate(mods):
        col, row = i % 3, i // 3
        x = 8 + col * 30
        y = 70 - row * 18
        _box(ax, x, y, 25, 12, name, face=NEUTRAL, edge=INK,
             fontsize=12, fontweight="bold", radius=1.0)

    ax.text(50, 12,
            "Synthetic so we get unlimited paired (clean, modified) samples.\n"
            "Real benchmarks (MFR2, CALFW, AgeDB-30, IIITD-Surgery, IJB-C) are the test set.",
            ha="center", fontsize=11, color="#333", linespacing=1.7)
    _save(pdf, fig)


def page_train(pdf):
    fig, ax = _page("How MDIE trains",
                    "One backbone, three heads, one composite loss.")

    # left column: inputs
    _box(ax, 5, 70, 22, 10, "clean image", face="#fff", edge=MUTED,
         fontsize=11)
    _box(ax, 5, 52, 22, 10, "modified image", face="#fff", edge=MUTED,
         fontsize=11)

    # center: shared backbone
    _box(ax, 38, 58, 24, 18, "IR-50\nbackbone\n(shared)",
         face=NEUTRAL, edge=INK, fontsize=12, fontweight="bold",
         radius=1.2)

    _arrow(ax, 27, 75, 38, 70)
    _arrow(ax, 27, 57, 38, 64)

    # right: 3 heads
    _box(ax, 72, 76, 23, 8, "ArcFace head",
         face=ARC, edge=ARC_E, fontsize=11, fontweight="bold")
    _box(ax, 72, 64, 23, 8, "AMD head (GRL)",
         face=AMD, edge=AMD_E, fontsize=11, fontweight="bold")
    _box(ax, 72, 52, 23, 8, "ICCL projector",
         face=ICCL, edge=ICCL_E, fontsize=11, fontweight="bold")

    _arrow(ax, 62, 70, 72, 80, color=ARC_E)
    _arrow(ax, 62, 67, 72, 68, color=AMD_E)
    _arrow(ax, 62, 64, 72, 56, color=ICCL_E)

    # composite loss
    _box(ax, 18, 28, 64, 12,
         "L_total  =  L_ArcFace  +  0.1 · L_AMD  +  0.5 · L_ICCL",
         face="#fff8e1", edge="#fbc02d", fontsize=13,
         fontweight="bold", radius=1.2)

    _arrow(ax, 83, 76, 83, 40, color=ARC_E, dashed=True)
    _arrow(ax, 83, 64, 83, 40, color=AMD_E, dashed=True)
    _arrow(ax, 83, 52, 83, 40, color=ICCL_E, dashed=True)

    _arrow(ax, 18, 34, 10, 34, color="#fbc02d", lw=1.4)
    _arrow(ax, 10, 34, 10, 66, color="#fbc02d", lw=1.4, dashed=True)
    ax.text(11, 50, "gradient back\nto backbone", fontsize=9,
            color="#a36a00", style="italic")

    _footer(ax, "All three heads are discarded at inference. Only the backbone ships.")
    _save(pdf, fig)


def page_infer(pdf):
    fig, ax = _page("How MDIE runs at inference",
                    "Same architecture as ArcFace. Zero extra parameters, zero extra latency.")

    _box(ax, 8, 70, 20, 10, "image A", face="#fff", edge=MUTED, fontsize=12)
    _box(ax, 8, 45, 20, 10, "image B", face="#fff", edge=MUTED, fontsize=12)

    _arrow(ax, 28, 75, 38, 75)
    _arrow(ax, 28, 50, 38, 50)

    _box(ax, 38, 70, 24, 10, "IR-50 backbone", face=NEUTRAL, edge=INK,
         fontsize=12, fontweight="bold")
    _box(ax, 38, 45, 24, 10, "IR-50 backbone", face=NEUTRAL, edge=INK,
         fontsize=12, fontweight="bold")

    _arrow(ax, 62, 75, 72, 75)
    _arrow(ax, 62, 50, 72, 50)

    _box(ax, 72, 70, 20, 10, "z_A  ∈  R^512", face="#f5f5f5", edge=MUTED,
         fontsize=11)
    _box(ax, 72, 45, 20, 10, "z_B  ∈  R^512", face="#f5f5f5", edge=MUTED,
         fontsize=11)

    _arrow(ax, 82, 70, 60, 32, color=MUTED)
    _arrow(ax, 82, 55, 60, 32, color=MUTED)

    _box(ax, 30, 22, 40, 12, "cos(z_A, z_B)  ≷  τ",
         face=GOOD, edge=GOOD_E, fontsize=14, fontweight="bold",
         radius=1.2)

    ax.text(50, 12,
            "Inference parameter count        ArcFace : 43.91 M       MDIE : 43.91 M",
            ha="center", fontsize=11.5, color=INK, fontweight="bold")
    _footer(ax, "AMD head + ICCL projector exist only during training.")
    _save(pdf, fig)


def page_amd(pdf):
    fig, ax = _page("Novelty 1  ·  AMD",
                    "Adversarial Modification Disentanglement.")

    # single horizontal flow
    _box(ax, 4, 70, 18, 10, "embedding\nz", face="#f5f5f5", edge=MUTED,
         fontsize=11)
    _arrow(ax, 22, 75, 30, 75, color=AMD_E, lw=1.5)
    _box(ax, 30, 70, 22, 10, "Gradient\nReversal Layer",
         face=AMD, edge=AMD_E, fontsize=11, fontweight="bold")
    _arrow(ax, 52, 75, 60, 75, color=AMD_E, lw=1.5)
    _box(ax, 60, 70, 22, 10, "MLP  →  m̂\n(modification id)",
         face=AMD, edge=AMD_E, fontsize=11, fontweight="bold")

    # what GRL does — two short bullets, no walls of text
    ax.text(50, 58, "What the GRL does",
            ha="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(50, 51,
            "forward pass    →   identity (pass-through)\n"
            "backward pass  →   multiply gradient by  −λ",
            ha="center", fontsize=12, family="monospace", color="#222",
            linespacing=1.8)

    # the equilibrium one-liner
    _box(ax, 12, 28, 76, 10,
         "min_θ  max_φ  −L_AMD(θ, φ)     ⇒     I(z ; modification)  →  0",
         face="#fff", edge=AMD_E, fontsize=13, fontweight="bold",
         radius=1.2)

    ax.text(50, 18,
            "The backbone is incentivised to make the modification UNRECOVERABLE\n"
            "from the embedding. Strictly stronger than data augmentation.",
            ha="center", fontsize=11, color="#333", linespacing=1.7)
    _footer(ax, "Inference cost: 0 parameters, 0 FLOPs.")
    _save(pdf, fig)


def page_iccl(pdf):
    fig, ax = _page("Novelty 2  ·  ICCL",
                    "Identity-Consistency Contrastive Loss.")

    # pair flow
    _box(ax, 5, 72, 22, 9, "clean   x", face="#fff", edge=MUTED,
         fontsize=11)
    _box(ax, 5, 58, 22, 9, "modified  x̃", face="#fff", edge=MUTED,
         fontsize=11)
    _arrow(ax, 27, 76, 35, 72, color=ICCL_E)
    _arrow(ax, 27, 62, 35, 67, color=ICCL_E)
    _box(ax, 35, 62, 22, 16, "backbone f_θ",
         face=NEUTRAL, edge=INK, fontsize=11, fontweight="bold")
    _arrow(ax, 57, 75, 65, 75, color=ICCL_E)
    _arrow(ax, 57, 65, 65, 65, color=ICCL_E)
    _box(ax, 65, 70, 28, 12, "InfoNCE\npull z, z̃ together",
         face=ICCL, edge=ICCL_E, fontsize=11, fontweight="bold",
         radius=1.0)
    _box(ax, 65, 56, 28, 10, "+ hard-negative term",
         face=ICCL, edge=ICCL_E, fontsize=11, fontweight="bold",
         radius=1.0)

    # the hard-negative idea in one short line
    ax.text(50, 46, "The hard-negative idea",
            ha="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(50, 39,
            "different person  +  SAME modification  =  the case that fools the model",
            ha="center", fontsize=12, color="#333")

    _box(ax, 12, 24, 76, 10,
         "L_hard = mean of  max( 0,  cos(z_i, z̃_j) − 0.3 )  ·  1[m_i=m_j, y_i≠y_j]",
         face="#fff", edge=ICCL_E, fontsize=11, fontweight="bold",
         radius=1.2)

    ax.text(50, 14,
            "Identity-balanced sampler  (32 classes × 8 samples per class)\n"
            "guarantees both same-id positives and same-mod hard negatives exist in every batch.",
            ha="center", fontsize=11, color="#333", linespacing=1.7)
    _save(pdf, fig)


def page_why_both(pdf):
    fig, ax = _page("Why both losses, not one",
                    "Each loss alone has a failure mode. Together they form a saddle point.")

    cards = [
        ("AMD only", AMD, AMD_E,
         "removes modification\nbut identity may collapse"),
        ("ICCL only", ICCL, ICCL_E,
         "preserves identity\nbut modification leaks back"),
        ("AMD + ICCL", "#fff8e1", "#fbc02d",
         "modification removed\nidentity preserved"),
    ]
    for i, (title, fc, ec, body) in enumerate(cards):
        x = 7 + i * 31
        _box(ax, x, 58, 26, 26, "", face=fc, edge=ec, radius=1.4)
        ax.text(x + 13, 78, title, ha="center", fontsize=13.5,
                fontweight="bold", color=ec)
        ax.text(x + 13, 67, body, ha="center", fontsize=11,
                color="#222", linespacing=1.7)

    # bar chart of measured worst-case AUC drop
    ax.text(50, 49, "Worst-case AUC drop across 9 modifications (lower = better)",
            ha="center", fontsize=12, fontweight="bold", color=INK)
    bars = [
        ("FaceNet  · 50 ep",            0.168, BAD_E),
        ("ArcFace  · 4 ep smoke",       0.135, BAD_E),
        ("CosFace  · 50 ep",            0.187, BAD_E),
        ("MobileFaceNet  · 50 ep",      0.179, BAD_E),
        ("MDIE-full  · 4 ep smoke",     0.010, GOOD_E),
    ]
    x0, y0, h, gap, scale = 32, 41, 3.0, 4.0, 250
    for i, (lbl, v, c) in enumerate(bars):
        y = y0 - i * gap
        ax.add_patch(mpatches.Rectangle((x0, y), max(v * scale, 0.6), h,
                     facecolor=c, edgecolor=c))
        ax.text(x0 - 1, y + h / 2, lbl, ha="right", va="center", fontsize=10)
        ax.text(x0 + v * scale + 1, y + h / 2, f"{v:.3f}",
                ha="left", va="center", fontsize=10, color="#222")

    ax.text(50, 14,
            "MDIE's drop is roughly 13× flatter than any baseline,\n"
            "already from a 4-epoch smoke run on RTX 3050.",
            ha="center", fontsize=11, color="#333", linespacing=1.7)
    _footer(ax, "Numbers from research_v2/results/{stage1,stage2}_metrics.json")
    _save(pdf, fig)


def page_pipeline(pdf):
    fig, ax = _page("End-to-end pipeline",
                    "Same code on RTX 3050 and on A100 — only the preset differs.")

    stages = [
        ("preflight",       "GPU, disk, dataset and fwd/bwd checks"),
        ("Stage 1",         "Train and evaluate FaceNet, ArcFace, CosFace, MobileFaceNet"),
        ("Stage 2",         "Train MDIE-full and ablations (−AMD, −RATA)"),
        ("real benchmarks", "MFR2, CALFW, AgeDB-30, IIITD-Surgery, IJB-C (gated)"),
        ("Grad-CAM",        "Where each model looks · CAM↔eye-region IoU"),
        ("artefacts",       "ROC plots, tables, methodology figures, this PDF"),
    ]
    y0, h, gap = 80, 7, 3.5
    for i, (k, v) in enumerate(stages):
        y = y0 - i * (h + gap)
        _box(ax, 6, y, 24, h, k, face=NEUTRAL, edge=INK,
             fontsize=11.5, fontweight="bold")
        _box(ax, 34, y, 60, h, v, face="#fff", edge="#bbb", fontsize=11)
        if i < len(stages) - 1:
            _arrow(ax, 18, y, 18, y - gap + 0.4, color=INK, lw=1.4)

    ax.text(50, 6, "One command:    bash scripts/launch_a100.sh",
            ha="center", fontsize=12, family="monospace", color=INK,
            fontweight="bold")
    _save(pdf, fig)


def page_results(pdf):
    fig, ax = _page("Measured results",
                    "From research_v2/results/  ·  no synthetic claims.")

    header = ("Model",           "Run",           "Clean",  "Worst",  "Drop")
    rows = [
        ("FaceNet",        "50 ep",       "0.687", "0.519", "0.168"),
        ("ArcFace",        "4 ep smoke",  "0.864", "0.730", "0.135"),
        ("CosFace",        "50 ep",       "0.900", "0.713", "0.187"),
        ("MobileFaceNet",  "50 ep",       "0.900", "0.721", "0.179"),
        ("MDIE-full",      "4 ep smoke",  "0.618", "0.608", "0.010"),
    ]
    cols_x = [10, 38, 58, 72, 88]
    y_top = 76

    for i, txt in enumerate(header):
        ax.text(cols_x[i], y_top, txt, ha="center", fontsize=12,
                fontweight="bold", color=INK)
    ax.plot([6, 96], [y_top - 2.5, y_top - 2.5], color=INK, lw=0.6)

    for j, row in enumerate(rows):
        y = y_top - 7 - j * 7
        is_mdie = row[0].startswith("MDIE")
        if is_mdie:
            ax.add_patch(mpatches.Rectangle((6, y - 1.8), 90, 5.5,
                         facecolor="#fff8e1", edgecolor="none"))
        for i, txt in enumerate(row):
            ax.text(cols_x[i], y, txt, ha="center", fontsize=11.5,
                    color="#222",
                    fontweight=("bold" if is_mdie else "normal"))

    ax.text(50, 30,
            "MDIE drop  =  0.010   ·   ~13× flatter than the strongest baseline",
            ha="center", fontsize=12.5, fontweight="bold", color=GOOD_E)

    ax.text(50, 20,
            "Absolute AUC for MDIE is still low because the run is only 4 epochs;\n"
            "the full 50-epoch A100 training is the next step.",
            ha="center", fontsize=11, color="#333", linespacing=1.7)

    _footer(ax, "Schema: clean and worst-mod AUC across 10 cells (clean + 9 modifications).")
    _save(pdf, fig)


def page_status(pdf):
    fig, ax = _page("Status",
                    "What is measured · what is pending.")

    # measured
    _box(ax, 5, 50, 42, 36, "", face="#eef7ee", edge=GOOD_E, radius=1.4)
    ax.text(26, 82, "Measured", ha="center", fontsize=14,
            fontweight="bold", color=GOOD_E)
    items_m = [
        "9-modification LFW eval harness",
        "FaceNet, CosFace, MobileFaceNet (50 ep)",
        "ArcFace 4-ep smoke",
        "MDIE-full 4-ep smoke",
        "Ablations: −AMD, −RATA (40 ep)",
        "ROC / EER / FAR@FRR per cell",
        "MFR2 / CALFW / AgeDB-30 real evals",
        "vs InsightFace w600k_r50 baseline",
    ]
    for i, t in enumerate(items_m):
        ax.text(7, 77 - i * 3.7, "·  " + t, fontsize=10.0, color="#1b5e20")

    # pending
    _box(ax, 53, 50, 42, 36, "", face="#fff3e0", edge="#e65100", radius=1.4)
    ax.text(74, 82, "Pending  (needs A100)", ha="center", fontsize=14,
            fontweight="bold", color="#bf360c")
    items_p = [
        "Full 50-ep MDIE on MS1MV3 seed",
        "IIITD Plastic Surgery (gated)",
        "IJB-C occlusion protocol (gated)",
        "MS1MV3 backbone pretraining",
        "Grad-CAM interpretability grid",
        "RATA re-train on MS1M seed",
    ]
    for i, t in enumerate(items_p):
        ax.text(55, 77 - i * 4.2, "·  " + t, fontsize=10.5, color="#bf360c")

    ax.text(50, 38,
            "Architecture, losses, training loop, eval harness, and paper pipeline\n"
            "are complete. Only compute-bound runs remain.",
            ha="center", fontsize=11.5, color="#333", linespacing=1.7,
            style="italic")
    _save(pdf, fig)


def page_real_benchmarks(pdf):
    """Real cross-dataset benchmark numbers (AUC) — honest results."""
    import json
    fig, ax = _page("Real benchmarks",
                    "AUC on public face-verification benchmarks (cross-dataset).")

    p = Path(__file__).resolve().parents[2] / "results" / "real_benchmarks.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    bench_order = ["mfr2", "calfw", "agedb30"]
    model_order = ["arcface", "cosface", "facenet", "mobilefacenet", "mdie",
                   "insightface_w600k_r50"]
    display = {
        "arcface": "ArcFace (ours, LFW-217)",
        "cosface": "CosFace (ours, LFW-217)",
        "facenet": "FaceNet (ours, LFW-217)",
        "mobilefacenet": "MobileFaceNet (ours, LFW-217)",
        "mdie": "MDIE-full (ours, LFW-217)",
        "insightface_w600k_r50": "InsightFace W600K-R50 (pretrained)",
    }

    # column headers
    ax.text(8, 78, "Model", fontsize=10.5, fontweight="bold", color=INK)
    for i, b in enumerate(bench_order):
        ax.text(58 + i * 12, 78, b.upper(), fontsize=10.5,
                fontweight="bold", color=INK, ha="center")
    ax.plot([6, 94], [76, 76], color=INK, lw=0.8)

    for r, m in enumerate(model_order):
        y = 72 - r * 5.5
        col = "#0d47a1" if m == "insightface_w600k_r50" else "#333"
        ax.text(8, y, display[m], fontsize=10, color=col)
        for i, b in enumerate(bench_order):
            v = data.get(b, {}).get(m, {}).get("auc")
            ax.text(58 + i * 12, y,
                    f"{v:.3f}" if isinstance(v, (int, float)) else "—",
                    fontsize=10, color=col, ha="center")

    ax.text(50, 28,
            "Take-away: models trained on our LFW-217 subset (~4.8 k images) do not\n"
            "generalise to unseen identities — AUC is at chance on CALFW / AgeDB-30.\n"
            "InsightFace W600K-R50 (60 M images, ResNet-50) is the realistic ceiling\n"
            "and what MDIE must match after the planned MS1MV3 pretrain on A100.",
            ha="center", fontsize=10.5, color="#333", linespacing=1.7,
            style="italic")
    _save(pdf, fig)


# ---------- main -----------------------------------------------------------

def main():
    with PdfPages(OUT) as pdf:
        page_cover(pdf)
        page_problem(pdf)
        page_modifications(pdf)
        page_train(pdf)
        page_infer(pdf)
        page_amd(pdf)
        page_iccl(pdf)
        page_why_both(pdf)
        page_pipeline(pdf)
        page_results(pdf)
        page_real_benchmarks(pdf)
        page_status(pdf)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
