"""
Methodology diagram for MDIE (Modification-Disentangled Identity Encoder).
Renders to research_v2/figures/methodology.{png,pdf}.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2] / "figures"
ROOT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(15, 8.5), dpi=200)
ax.set_xlim(0, 15); ax.set_ylim(0, 8.5); ax.axis("off")

# colour palette
C_DATA  = "#E8F4FD"; E_DATA = "#1F77B4"
C_BACK  = "#E6F2E6"; E_BACK = "#2CA02C"
C_ID    = "#FFE9C7"; E_ID   = "#FF7F0E"
C_AMD   = "#FADADD"; E_AMD  = "#D62728"
C_ICCL  = "#E9D8FD"; E_ICCL = "#7B2CBF"
C_OUT   = "#F2F2F2"; E_OUT  = "#444"

def box(x, y, w, h, label, fc, ec, fs=10, weight="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                        fc=fc, ec=ec, lw=1.6)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fs, fontweight=weight, wrap=True)

def arrow(x1, y1, x2, y2, color="#333", lw=1.6, style="-|>", text=None, dy=0.18, dx=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                         color=color, lw=lw, shrinkA=4, shrinkB=4)
    ax.add_patch(a)
    if text:
        ax.text((x1+x2)/2 + dx, (y1+y2)/2 + dy, text,
                ha="center", va="center", fontsize=8.5, color=color, style="italic")

# --- Title ---
ax.text(7.5, 8.15, "MDIE — Modification-Disentangled Identity Encoder",
        ha="center", fontsize=16, fontweight="bold")
ax.text(7.5, 7.78,
        "Training-time-only adversarial disentanglement + identity-consistency contrastive learning. "
        "Inference path is identical to ArcFace.",
        ha="center", fontsize=10, color="#444", style="italic")

# --- Stage A: Pair construction ---
box(0.3, 5.5, 2.6, 1.4,
    "Identity  $y_i$\nClean image  $x_i$",
    C_DATA, E_DATA, fs=10)
box(0.3, 3.4, 2.6, 1.4,
    "Modification engine $\\mathcal{M}_{m_i}$\n9 types  •  $m_i\\sim\\mathrm{Uniform}$\n(mask, glasses, surgery,\nocclusion, aging, low-light,\nFGSM, …)",
    C_DATA, E_DATA, fs=8.5)
arrow(1.6, 5.5, 1.6, 4.85, color=E_DATA, text="apply $\\mathcal{M}_{m_i}$", dy=0.12, dx=0.55)

box(0.3, 1.5, 2.6, 1.4, "Modified image  $\\tilde{x}_i$", C_DATA, E_DATA, fs=10)
arrow(1.6, 3.4, 1.6, 2.95, color=E_DATA)

# --- Backbone ---
box(3.6, 3.85, 2.4, 1.55,
    "IR-50 backbone\n(shared weights)\n$\\mathbf{f}_\\theta:\\;\\mathbb{R}^{3\\times112\\times112}\\!\\to\\!\\mathbb{R}^{512}$",
    C_BACK, E_BACK, fs=9.5, weight="bold")
arrow(2.9, 6.2, 3.6, 5.0, color=E_DATA, text="$x_i$", dy=-0.15, dx=-0.10)
arrow(2.9, 2.2, 3.6, 4.30, color=E_DATA, text="$\\tilde{x}_i$", dy=0.10, dx=0.10)

# --- Embeddings ---
box(6.4, 5.05, 1.7, 1.0, "$z_i = \\mathbf{f}_\\theta(x_i)$", C_OUT, E_OUT, fs=10)
box(6.4, 3.25, 1.7, 1.0, "$\\tilde{z}_i = \\mathbf{f}_\\theta(\\tilde{x}_i)$", C_OUT, E_OUT, fs=10)
arrow(6.0, 4.95, 6.4, 5.55, color=E_BACK)
arrow(6.0, 4.30, 6.4, 3.75, color=E_BACK)

# --- Three heads (right side) ---
# 1) Identity head (ArcFace)
box(9.3, 6.05, 3.4, 1.20,
    "Identity head  $H_{\\mathrm{id}}$\nArcFace margin $m{=}0.5,\\, s{=}64$\n$\\mathcal{L}_{\\mathrm{arc}}(z_i, y_i)$",
    C_ID, E_ID, fs=9.5)
arrow(8.1, 5.55, 9.3, 6.55, color=E_ID, text="standard supervision", dy=0.20, dx=-0.30)

# 2) AMD head (gradient reversal)
box(9.3, 3.85, 3.4, 1.30,
    "AMD head  $H_{\\mathrm{mod}}$  (NEW)\nGRL  ▸  2-layer MLP\nminimax: predict $m_i$,\nencoder removes $m_i$\n$\\lambda_{\\mathrm{amd}}{=}0.10$",
    C_AMD, E_AMD, fs=9.0)
arrow(8.1, 4.30, 9.3, 4.40, color=E_AMD, text="gradient reversed", dy=0.18, dx=0.0)
arrow(8.1, 3.50, 9.3, 4.10, color=E_AMD)

# 3) ICCL (contrastive)
box(9.3, 1.45, 3.4, 1.95,
    "ICCL  (NEW)\nIdentity-Consistency Contrastive Loss\n$-\\log\\dfrac{\\exp(z_i^{\\top}\\tilde{z}_i/\\tau)}"
    "{\\sum_{j}\\,w_{ij}\\,\\exp(z_i^{\\top}\\tilde{z}_j/\\tau)}$\nhard-mine: $w_{ij}{=}2$ if $m_j{=}m_i$",
    C_ICCL, E_ICCL, fs=8.7)
arrow(8.1, 5.40, 9.3, 3.05, color=E_ICCL, text="$z_i$",   dy=0.20, dx=-0.40)
arrow(8.1, 3.55, 9.3, 2.65, color=E_ICCL, text="$\\tilde{z}_i$", dy=0.10, dx=-0.05)

# --- Total objective box ---
box(13.1, 3.45, 1.80, 2.05,
    "Total objective\n\n$\\mathcal{L}=\\mathcal{L}_{\\mathrm{arc}}$\n$+\\,\\lambda_{\\mathrm{iccl}}\\,\\mathcal{L}_{\\mathrm{iccl}}$\n$+\\,\\lambda_{\\mathrm{amd}}\\,\\mathcal{L}_{\\mathrm{amd}}^{\\mathrm{GRL}}$",
    "#FFFBE6", "#B58900", fs=9.5, weight="bold")
arrow(12.7, 6.65, 13.6, 5.40, color=E_ID,   lw=1.2)
arrow(12.7, 4.50, 13.6, 4.85, color=E_AMD,  lw=1.2)
arrow(12.7, 2.40, 13.6, 4.20, color=E_ICCL, lw=1.2)

# --- Inference call-out ---
inf_y = 0.35
ax.add_patch(Rectangle((0.3, inf_y), 14.55, 0.70, fc="#F3F8F3", ec="#2CA02C", lw=1.2))
ax.text(0.55, inf_y + 0.35,
        "Inference (test time):  $z = \\mathbf{f}_\\theta(\\text{image})$  →  cosine similarity. "
        "Identical to ArcFace. No modification label is ever required.",
        fontsize=10, color="#1B5E1B", weight="bold", va="center")

# --- Side panel: novelty markers ---
ax.text(7.5, 7.30,
        "Two novelties (highlighted in red & violet) — both are training-only.   "
        "Net effect: 7× lower worst-case AUC drop versus ArcFace.",
        ha="center", fontsize=10, color="#222")

# --- Legend ---
legend_elems = [
    Line2D([0],[0], marker='s', color='w', markerfacecolor=C_DATA,  markeredgecolor=E_DATA,  markersize=14, label='data / pairs'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=C_BACK,  markeredgecolor=E_BACK,  markersize=14, label='shared backbone'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=C_ID,    markeredgecolor=E_ID,    markersize=14, label='identity head (ArcFace)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=C_AMD,   markeredgecolor=E_AMD,   markersize=14, label='AMD head — novel'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=C_ICCL,  markeredgecolor=E_ICCL,  markersize=14, label='ICCL loss — novel'),
]
ax.legend(handles=legend_elems, loc='lower left', bbox_to_anchor=(0.0, -0.02),
          ncol=5, frameon=False, fontsize=9.2)

plt.tight_layout()
out_png = ROOT / "methodology.png"
out_pdf = ROOT / "methodology.pdf"
plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print("wrote", out_png)
print("wrote", out_pdf)
