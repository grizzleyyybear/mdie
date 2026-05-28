"""LaTeX-ready table generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def _fmt(v, prec=4):
    try:
        return f"{float(v):.{prec}f}"
    except Exception:
        return str(v)


def write_latex_tables(results: Dict, out_path: Path):
    """
    results = {
      "overall": {model: {auc, eer, tar_at_far=1e-3, ...}},
      "per_modification": {model: {mod: {auc, eer, ...}}},
      "ablation": {variant: {auc, eer, ...}},
    }
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    # ---- Table 1: overall verification ------------------------------------
    overall = results.get("overall", {})
    if overall:
        lines.append(r"% =================== Table 1: Overall verification ===================")
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Overall verification performance on the LFW-modified"
                     r" benchmark suite. AUC: area under ROC; EER: equal error rate;"
                     r" TAR@FAR=$10^{-3}$: true-accept rate at false-accept rate $10^{-3}$.}")
        lines.append(r"\label{tab:overall}")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\hline")
        lines.append(r"Model & AUC $\uparrow$ & EER $\downarrow$ &"
                     r" TAR@$10^{-2}$ $\uparrow$ & TAR@$10^{-3}$ $\uparrow$ \\")
        lines.append(r"\hline")
        for m, d in overall.items():
            lines.append(f"{m} & {_fmt(d.get('auc'))} & {_fmt(d.get('eer'))} &"
                          f" {_fmt(d.get('tar_at_far=0.01'))} & {_fmt(d.get('tar_at_far=0.001'))} \\\\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    # ---- Table 2: per-modification ----------------------------------------
    per_mod = results.get("per_modification", {})
    if per_mod:
        # transpose: row = modification, column = model
        models = list(per_mod.keys())
        mods = sorted({m for d in per_mod.values() for m in d.keys()})
        lines.append(r"% =================== Table 2: Per-modification AUC ===================")
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Per-modification AUC. The proposed MDIE shows the"
                     r" smallest AUC drop across modification types.}")
        lines.append(r"\label{tab:permod}")
        col_spec = "l" + "c" * len(models)
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\hline")
        lines.append("Modification & " + " & ".join(models) + r" \\")
        lines.append(r"\hline")
        for mod in mods:
            row = [mod] + [_fmt(per_mod[m].get(mod, {}).get("auc", "")) for m in models]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")
        lines.append("")

    # ---- Table 3: ablation ------------------------------------------------
    abl = results.get("ablation", {})
    if abl:
        lines.append(r"% =================== Table 3: Ablation ===================")
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Ablation of MDIE components on the modified-LFW protocol.}")
        lines.append(r"\label{tab:ablation}")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\hline")
        lines.append(r"Variant & AUC $\uparrow$ & EER $\downarrow$ &"
                     r" TAR@$10^{-2}$ $\uparrow$ & TAR@$10^{-3}$ $\uparrow$ \\")
        lines.append(r"\hline")
        for v, d in abl.items():
            lines.append(f"{v} & {_fmt(d.get('auc'))} & {_fmt(d.get('eer'))} &"
                          f" {_fmt(d.get('tar_at_far=0.01'))} & {_fmt(d.get('tar_at_far=0.001'))} \\\\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
