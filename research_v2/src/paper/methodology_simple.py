"""
Simple methodology diagram mapped to all seven stages of the proposed execution plan.
Text is sized + boxes scaled so nothing overflows.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2] / "figures"
ROOT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

C_PROBE = "#FFE0B2"; E_PROBE = "#E65100"
C_BACK  = "#C8E6C9"; E_BACK  = "#1B5E20"
C_NOVEL = "#FFCDD2"; E_NOVEL = "#B71C1C"
C_OUT   = "#BBDEFB"; E_OUT   = "#0D47A1"
C_TAG   = "#ECEFF1"; E_TAG   = "#37474F"
C_FUT   = "#F5F5F5"; E_FUT   = "#9E9E9E"

def box(x, y, w, h, lines, fc, ec, fs=10, weight="normal", color="black"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                        fc=fc, ec=ec, lw=1.6)
    ax.add_patch(p)
    txt = lines if isinstance(lines, str) else "\n".join(lines)
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=color)

def arrow(x1, y1, x2, y2, lw=1.8, color="#333"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                         color=color, lw=lw, shrinkA=3, shrinkB=3)
    ax.add_patch(a)

ax.text(8.0, 8.55, "MDIE  —  Methodology mapped to the proposed execution plan",
        ha="center", fontsize=17, fontweight="bold")
ax.text(8.0, 8.18,
        "Modification-invariant face recognition: surgery, mask, glasses, occlusion, aging, low-light, adversarial.",
        ha="center", fontsize=10.5, color="#444", style="italic")

y = 4.4; h = 1.95; w = 3.55
gap = 0.30
xs = [0.4, 0.4 + (w + gap), 0.4 + 2*(w + gap), 0.4 + 3*(w + gap)]

box(xs[0], y, w, h,
    ["1. INPUT",
     "",
     "Clean face + same face",
     "with one of 9 modifications",
     "(mask, glasses, surgery,",
     "occlusion, aging, low-light,",
     "adversarial)"],
    C_PROBE, E_PROBE, fs=10.0, weight="bold")

box(xs[1], y, w, h,
    ["2. SHARED ENCODER",
     "",
     "IR-50 backbone",
     "(same as ArcFace)",
     "",
     "produces a 512-D embedding",
     "for both views"],
    C_BACK, E_BACK, fs=10.0, weight="bold")

box(xs[2], y, w, h,
    ["3. TWO TRAINING-ONLY SIGNALS",
     "",
     "AMD: gradient-reversal head",
     "         erases  which-modification  cue",
     "ICCL: contrastive loss pulls",
     "         clean and modified together",
     "(novel — used only at training)"],
    C_NOVEL, E_NOVEL, fs=9.2, weight="bold")

box(xs[3], y, w, h,
    ["4. INFERENCE",
     "",
     "Identical to ArcFace:",
     "embed  ->  cosine similarity",
     "",
     "No modification label",
     "ever required at runtime"],
    C_OUT, E_OUT, fs=10.0, weight="bold")

for i in range(3):
    arrow(xs[i] + w, y + h/2, xs[i+1], y + h/2)

tag_y = 6.75; tag_h = 0.95
def tag(x, w, lines, fc=C_TAG, ec=E_TAG):
    box(x, tag_y, w, tag_h, lines, fc, ec, fs=8.8, color="#222")
tag(xs[0], w, ["Plan Stage 1", "Problem validation +", "failure-mode benchmark"])
tag(xs[1], w, ["Plan Stage 3A", "Lightweight edge backbone", "(IR-50 / ArcFace)"])
tag(xs[2], w, ["Plan Stages 3C + 3D", "Region-stable representation +", "embedding optimisation"])
tag(xs[3], w, ["Plan Stage 4", "Laptop end-to-end", "demonstrator"])
for x in xs:
    arrow(x + w/2, tag_y, x + w/2, y + h + 0.05, lw=1.0, color="#777")

fy = 2.55; fh = 1.45
def futurebox(x, w, title, lines):
    box(x, fy, w, fh,
        [title, ""] + lines,
        C_FUT, E_FUT, fs=9.2, color="#555")
futurebox(0.4, 4.85, "Plan Stage 2  -  GAN augmentation",
          ["Synthetic before/after surgery pairs.",
           "MDIE today uses protocol-faithful synthetic mods;",
           "GAN pipeline plugs into INPUT (box 1) when ready."])
futurebox(5.55, 4.85, "Plan Stage 3B  -  Depth + IR fusion",
          ["RGB + depth + IR at face-capture distance.",
           "Hooks into ENCODER (box 2) once",
           "sensor procurement completes."])
futurebox(10.7, 4.85, "Plan Stages 5 - 7  -  Edge, federated, field",
          ["Inference is identical to ArcFace,",
           "so MDIE ports to edge (5) unchanged.",
           "Federated (6) + field deploy (7) wrap around it."])

ax.text(8.0, 4.15, "Future stages of the plan  -  MDIE is built so each plugs in cleanly:",
        ha="center", fontsize=10, style="italic", color="#555")

res_y = 0.55; res_h = 1.55
ax.add_patch(FancyBboxPatch((0.4, res_y), 15.2, res_h,
              boxstyle="round,pad=0.05,rounding_size=0.10",
              fc="#FFF9C4", ec="#F57F17", lw=1.6))
ax.text(8.0, res_y + 1.20,
        "RESULT  -  LFW + 9 modifications,  217 identities,  6000 verification pairs / modification,  RTX 3050 4GB",
        ha="center", fontsize=11.5, weight="bold", color="#5D4037")
ax.text(8.0, res_y + 0.70,
        "ArcFace worst-case AUC drop:  0.198      ->      MDIE worst-case AUC drop:  0.027      (~ 7x more robust)",
        ha="center", fontsize=12, color="#222")
ax.text(8.0, res_y + 0.25,
        "Pooled AUC: ArcFace 0.861  vs  MDIE 0.850 (within 1 pp).   Inference cost: identical to ArcFace.",
        ha="center", fontsize=10.2, color="#444", style="italic")

plt.tight_layout()
out_png = ROOT / "methodology_simple.png"
out_pdf = ROOT / "methodology_simple.pdf"
plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print("wrote", out_png)
print("wrote", out_pdf)
