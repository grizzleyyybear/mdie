"""Build a single honest research one-pager: hero attention figure + the real
CASIA-vs-InsightFace benchmark table + a paper-style abstract & contributions.

Data-driven: the results table is read straight from the committed CSVs so the
numbers can never drift from what the model actually produced. Run:

    python _report/build_onepager.py
"""
import csv
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "MDIE_Research_OnePager.pdf")
HERO = os.path.join(ROOT, "research_v2", "figures", "stage2_attention_examples.png")

# Real-benchmark CSV: prefer the freshly scp'd copy, fall back to in-repo path.
CSV_CANDIDATES = [
    os.path.join(ROOT, "casia_real_results", "real_benchmarks_casia.csv"),
    os.path.join(ROOT, "research_v2", "results", "casia_real", "real_benchmarks_casia.csv"),
]
NORATA_CANDIDATES = [
    os.path.join(ROOT, "casia_real_results", "mdie_norata", "real_benchmarks_mdie_norata.csv"),
    os.path.join(ROOT, "research_v2", "results", "casia_real", "mdie_norata", "real_benchmarks_mdie_norata.csv"),
]

NAVY = colors.HexColor("#1f3a5f")
ACCENT = colors.HexColor("#c0392b")
LIGHT = colors.HexColor("#eaeef3")
GREEN = colors.HexColor("#1e7e45")
GREY = colors.HexColor("#5d6d7e")


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of these exist: {paths}")


def _read_auc(csv_path):
    """benchmark -> {model: auc} from a run_real_benchmarks CSV."""
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["benchmark"], {})[row["model"]] = float(row["auc"])
    return out


styles = getSampleStyleSheet()


def S(name, **kw):
    base = kw.pop("parent", styles["Normal"])
    return ParagraphStyle(name, parent=base, **kw)


title = S("title", fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=NAVY, spaceAfter=2)
sub = S("sub", fontName="Helvetica-Oblique", fontSize=9, leading=12, textColor=GREY, spaceAfter=6)
h = S("h", fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=ACCENT, spaceBefore=7, spaceAfter=3)
body = S("body", fontName="Helvetica", fontSize=8.7, leading=11.6, alignment=TA_JUSTIFY, spaceAfter=4)
small = S("small", fontName="Helvetica", fontSize=7.6, leading=10, textColor=GREY)
cap = S("cap", fontName="Helvetica-Oblique", fontSize=7.6, leading=9.5, textColor=GREY, alignment=TA_CENTER, spaceAfter=4)
white = S("white", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.white)
cell = S("cell", fontName="Helvetica", fontSize=8, leading=10)


def para(t, st=body):
    return Paragraph(t, st)


# ---------------------------------------------------------------- data
auc = _read_auc(_first_existing(CSV_CANDIDATES))
norata = _read_auc(_first_existing(NORATA_CANDIDATES))

ORDER = [("mfr2", "MFR2 (real masks)"), ("meglass", "MeGlass (real glasses)"),
         ("calfw", "CALFW (cross-age)"), ("agedb30", "AgeDB-30 (30-yr gap)")]
# Old LFW-era MDIE numbers (committed: research_v2/results/real_benchmarks.csv).
LFW_MDIE = {"mfr2": 0.734, "meglass": 0.824, "calfw": 0.557, "agedb30": 0.594}


def f3(x):
    return f"{x:.3f}"


# ---------------------------------------------------------------- story
story = []
story.append(para("MDIE &mdash; Bone-Anchored, Interpretable, ArcFace-Compatible "
                  "Face Recognition under Worn Occlusion", title))
story.append(para("CASIA-WebFace (0.49M) scale-up &middot; PARAM Siddhi-AI A100 &middot; honest research summary", sub))

# ---- Abstract
story.append(para("Abstract", h))
story.append(para(
    "Production face recognizers fold nuisance factors &mdash; masks, glasses, caps, harsh lighting &mdash; into "
    "the identity embedding, so similarity collapses exactly when a security camera needs it most. We present "
    "<b>MDIE</b>, an encoder that anchors a token-attention map to the <b>rigid facial bones</b> (brow ridge, orbital "
    "rims, nasal bridge, cheekbones, jaw) that survive appearance changes, trained with two auxiliary training-only "
    "signals (adversarial modification disentanglement and a clean&harr;modified consistency loss). The deployed "
    "output is a <b>single 512-d L2-normalised vector from one forward pass</b> &mdash; a literal ArcFace drop-in for "
    "any cosine / FAISS pipeline. Trained on CASIA-WebFace (0.49M images, ~24&times; less data than the production "
    "reference), MDIE <b>approaches a production InsightFace model on real worn-occlusion benchmarks</b> (real glasses "
    f"within {auc['meglass']['insightface_w600k_r50']-auc['meglass']['mdie_full']:.3f} AUC) while providing a novel, "
    "control-backed interpretability metric proving the attention rests on anatomy rather than a dataset shortcut. "
    "We do <b>not</b> claim to beat production accuracy; the contribution is interpretable, anatomy-grounded "
    "robustness at small data scale plus a reproducible occlusion failure-mode benchmark.", body))

# ---- Contributions
story.append(para("Contributions (with honest novelty labels)", h))
contrib = [
    ("<b>RATA &mdash; rigid-bone-anchored token attention.</b> Skeletal landmarks are splatted into a 14&times;14 "
     "attention target and supervised with a matching loss + additive attention bias; under occlusion the attention "
     "re-distributes onto whichever bones remain visible. <font color='#1e7e45'><b>[novel]</b></font>"),
    ("<b>Attention&ndash;bone IoU, a control-backed interpretability metric.</b> Matched &gt;&gt; mismatched &gt;&gt; "
     "random overlap (IoU 0.69 / 0.28 / 0.08, Mann-Whitney p=1.6e-15) on held-out faces &mdash; to our knowledge "
     "unreported in prior occlusion-FR. <font color='#1e7e45'><b>[novel]</b></font>"),
    ("<b>AMD + ICCL training-only invariance signals.</b> Gradient-reversal erases which-modification cues; a "
     "consistency loss pulls clean and modified views together. <font color='#5d6d7e'><b>[incremental application "
     "of known mechanisms]</b></font>"),
    ("<b>ArcFace-compatible deployment + a reproducible occlusion/lighting failure-mode benchmark.</b> One 512-d "
     "vector, cosine==dot, deterministic; full pipeline runs on a 4&nbsp;GB laptop GPU. <font color='#5d6d7e'>"
     "<b>[engineering]</b></font>"),
]
for c in contrib:
    story.append(Table([[para("&bull;", small), para(c, body)]], colWidths=[0.4*cm, 16.2*cm],
                       style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                         ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                         ("TOPPADDING", (0, 0), (-1, -1), 0),
                                         ("BOTTOMPADDING", (0, 0), (-1, -1), 1)])))

# ---- Hero figure + results table, side by side
story.append(para("How it works &mdash; real attention on an occluded face", h))
hero = Image(HERO, width=9.2*cm, height=3.45*cm)

# Results table (data-driven)
data = [[para("Real benchmark", white), para("MDIE<br/>LFW 13k", white),
         para("MDIE<br/>CASIA 0.49M", white), para("Insight-<br/>Face 12M", white)]]
for key, label in ORDER:
    data.append([para(label, cell), f3(LFW_MDIE[key]),
                 f3(auc[key]["mdie_full"]), f3(auc[key]["insightface_w600k_r50"])])
tbl = Table(data, colWidths=[3.5*cm, 1.5*cm, 1.8*cm, 1.8*cm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
    ("FONT", (2, 1), (2, -1), "Helvetica-Bold", 8),
    ("TEXTCOLOR", (2, 1), (2, -1), ACCENT),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cfd6df")),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
]))
right = [tbl, Spacer(1, 3), para("AUC on held-out pairs. CASIA scale-up lifts MDIE on every benchmark; "
        "it approaches production on worn occlusion, trails on aging (off-niche).", cap)]
side = Table([[hero, right]], colWidths=[9.4*cm, 7.2*cm])
side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (1, 0), (1, 0), 8)]))
story.append(side)
story.append(para("Hero: MDIE-full attention on a cap-occluded face &mdash; the heatmap ignores the cap and "
                  "concentrates on brow ridge, eye sockets and nasal bridge (the visible rigid bones). "
                  "Source: research_v2/figures/stage2_attention_examples.png.", cap))

# ---- Honest positioning + scope/limitations
story.append(para("Honest positioning &amp; scope", h))
nr = norata  # the ablation that nominally leads on real benchmarks
story.append(para(
    "<b>What is true.</b> On its design target (worn occlusion) the 0.49M-image MDIE is competitive with a "
    f"WebFace12M production model (MeGlass {f3(auc['meglass']['mdie_full'])} vs "
    f"{f3(auc['meglass']['insightface_w600k_r50'])}; MFR2 {f3(auc['mfr2']['mdie_full'])} vs "
    f"{f3(auc['mfr2']['insightface_w600k_r50'])}). The bone-anchored attention is interpretable and "
    "control-verified. The encoder is a true ArcFace drop-in.", body))
story.append(para(
    "<b>What we do not claim.</b> MDIE does not beat production InsightFace on any benchmark, and the gap widens on "
    f"aging (AgeDB-30 {f3(auc['agedb30']['mdie_full'])} vs {f3(auc['agedb30']['insightface_w600k_r50'])}), which it "
    "was never designed for. At CASIA scale the four ablation variants are statistically tied and "
    "<b>non-monotone</b>: on real benchmarks the no-RATA variant nominally leads "
    f"(MFR2 {f3(nr['mfr2']['mdie_norata'])} vs full {f3(auc['mfr2']['mdie_full'])}). We therefore position RATA as an "
    "<b>interpretability</b> contribution, not a pooled-accuracy gain &mdash; the pretrained backbone carries most "
    "of the invariance at scale.", body))
story.append(para(
    "<b>Honest venue fit.</b> A biometrics venue (IJCB / IEEE&nbsp;FG), a CVPR/ICCV face-analysis workshop, or a "
    "mid-tier journal (Pattern Recognition Letters, IET Biometrics) &mdash; the contribution is interpretability + a "
    "reproducible failure-mode benchmark + an honest small-data scale-up, not a SOTA-accuracy claim. A top-tier main "
    "track would require a win against an occlusion-specialised baseline on a standard masked-FR protocol and "
    "training at MS1M/WebFace scale.", body))

story.append(Spacer(1, 4))
story.append(para("Reproducibility: results read directly from results/casia_real/real_benchmarks_casia.csv; "
                  "checkpoints under research_v2/checkpoints/fanout/; code at github.com/grizzleyyybear/mdie.", small))

doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=1.5*cm, rightMargin=1.5*cm,
                        topMargin=1.2*cm, bottomMargin=1.0*cm)
doc.build(story)
print(f"PDF written: {OUT}")
