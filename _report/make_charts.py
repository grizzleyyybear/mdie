"""Generate charts for the MDIE status report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))
NAVY = "#1f3a5f"
ACCENT = "#c0392b"
GREY = "#95a5a6"
GREEN = "#27ae60"

plt.rcParams.update({"font.size": 10})

# 1. Pooled AUC: MDIE vs comparably-trained baselines
models = ["FaceNet", "ArcFace", "CosFace", "MobileFN", "MDIE-full"]
pooled = [0.603, 0.641, 0.749, 0.749, 0.979]
colors = [GREY, GREY, GREY, GREY, ACCENT]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
bars = ax.bar(models, pooled, color=colors, edgecolor="black", linewidth=0.6)
ax.set_ylabel("Pooled AUC (held-out IDs)")
ax.set_ylim(0, 1.05)
ax.set_title("Robustness on occlusion+lighting niche (comparably-trained)", fontweight="bold")
for b, v in zip(bars, pooled):
    ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.3f}", ha="center", fontsize=9)
ax.axhline(0.5, ls="--", lw=0.7, color="black", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_pooled.png"), dpi=200)
plt.close()

# 2. Ablation ladder (occlusion / lighting)
variants = ["MDIE-noRATA", "MDIE-noAMD", "MDIE-noICCL", "MDIE-full"]
occ = [0.9688, 0.9696, 0.9700, 0.9745]
light = [0.9725, 0.9752, 0.9751, 0.9799]
x = np.arange(len(variants)); w = 0.38
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.bar(x-w/2, occ, w, label="occlusion", color=NAVY, edgecolor="black", linewidth=0.5)
ax.bar(x+w/2, light, w, label="lighting", color=ACCENT, edgecolor="black", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(variants, fontsize=9)
ax.set_ylim(0.96, 0.985)
ax.set_ylabel("AUC")
ax.set_title("Ablation: each component helps (monotone full > rest)", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_ablation.png"), dpi=200)
plt.close()

# 3. Real benchmarks (AUC) grouped
benches = ["MFR2\n(real masks)", "MeGlass\n(real glasses)", "CALFW\n(age)", "AgeDB-30\n(age)"]
arc = [0.659, 0.724, 0.514, 0.528]
cos = [0.667, 0.726, 0.518, 0.504]
mob = [0.632, 0.681, 0.523, 0.536]
mdie = [0.734, 0.824, 0.557, 0.594]
x = np.arange(len(benches)); w = 0.2
fig, ax = plt.subplots(figsize=(6.6, 3.3))
ax.bar(x-1.5*w, arc, w, label="ArcFace", color="#aab7c4", edgecolor="black", linewidth=0.4)
ax.bar(x-0.5*w, cos, w, label="CosFace", color="#7f8fa6", edgecolor="black", linewidth=0.4)
ax.bar(x+0.5*w, mob, w, label="MobileFN", color="#4b6584", edgecolor="black", linewidth=0.4)
ax.bar(x+1.5*w, mdie, w, label="MDIE", color=ACCENT, edgecolor="black", linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels(benches, fontsize=8.5)
ax.set_ylabel("AUC"); ax.set_ylim(0, 0.95)
ax.set_title("Real-world transfer: MDIE wins every benchmark", fontweight="bold")
ax.legend(fontsize=8, ncol=4, loc="upper center")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_real.png"), dpi=200)
plt.close()

# 4. Attention-bone IoU interpretability
cats = ["matched\n(own bones)", "mismatched\n(other face)", "random\nnull"]
iou = [0.694, 0.280, 0.077]
err = [0.090, 0.100, 0.0]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
bars = ax.bar(cats, iou, yerr=err, capsize=4, color=[GREEN, GREY, "#bdc3c7"],
              edgecolor="black", linewidth=0.6)
ax.set_ylabel("IoU @ top-15%")
ax.set_ylim(0, 0.85)
ax.set_title("Attention anchors on each face's own bones\n(Mann-Whitney p=1.6e-15)", fontweight="bold", fontsize=10)
for b, v in zip(bars, iou):
    ax.text(b.get_x()+b.get_width()/2, v+0.04, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_iou.png"), dpi=200)
plt.close()

print("charts written")
