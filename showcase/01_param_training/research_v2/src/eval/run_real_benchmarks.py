"""
Real-benchmark evaluation harness.

Runs every model in ``--models`` against every available benchmark in ``--benchmarks``
and writes a CSV report + per-benchmark ROC figures.

Loaders for free benchmarks (mfr2, calfw, agedb30) auto-download on first use.
Gated benchmarks (iiitd_surgery, ijbc_occ) are silently skipped if their env
vars are unset, so this is safe to run anywhere.

Usage:
    python -m research_v2.src.eval.run_real_benchmarks --models arcface mdie
    python -m research_v2.src.eval.run_real_benchmarks --benchmarks mfr2 calfw agedb30
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch

from ..config import CKPT_DIR, FIGURES_DIR, RESULTS_DIR, get_device, seed_all
from ..data import PairSet
from ..data.benchmarks import Benchmark, list_benchmarks, load_benchmark
from .embeddings import extract_embeddings_for_pairs, score_pairs, summarize_run


# ----------------------------------------------------------------------------
# Model loaders. Each returns ``(encode_fn, friendly_name)`` or ``None`` if the
# model checkpoint is not on disk yet.
# ----------------------------------------------------------------------------

def _baseline_loader(name: str, device: torch.device):
    """Build a baseline model and load its checkpoint if available."""
    from ..baselines import build_baseline
    ck = CKPT_DIR / f"baseline_{name}.pt"
    if not ck.exists():
        legacy = CKPT_DIR / f"{name}.pt"
        if legacy.exists():
            ck = legacy
        else:
            return None
    try:
        sd = torch.load(ck, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        # Infer head class count from checkpoint to avoid size-mismatch errors.
        n_classes = 1
        if "head.W" in sd:
            n_classes = int(sd["head.W"].shape[0])
        elif "head.weight" in sd:
            n_classes = int(sd["head.weight"].shape[0])
        model = build_baseline(name, n_classes=n_classes).to(device).eval()
        model.load_state_dict(sd, strict=False)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] {name} checkpoint load failed: {e}")
        return None

    def enc(x):
        with torch.no_grad():
            return model.backbone(x.to(device))["embedding"]
    return enc, name


def _mdie_loader(device: torch.device):
    from ..novel import MDIE
    # Resolve checkpoint with backward-compat for legacy hyphen names.
    candidates = [CKPT_DIR / "mdie_full.pt", CKPT_DIR / "mdie-full.pt",
                  CKPT_DIR / "mdie.pt"]
    ck = next((c for c in candidates if c.exists()), None)
    if ck is None:
        return None
    state = torch.load(ck, map_location=device, weights_only=False)
    cfg = state.get("config", {}) if isinstance(state, dict) else {}
    sd = state.get("model", state) if isinstance(state, dict) and "model" in state else state
    has_rata = any(k.startswith(("attn.", "post_pool.")) for k in sd.keys())
    has_amd = any(k.startswith(("mod_head.", "grl.")) for k in sd.keys())
    # Infer identity-head size to avoid size mismatches for legacy checkpoints.
    n_id = cfg.get("n_identity_classes")
    if n_id is None:
        if "identity_head.W" in sd:
            n_id = int(sd["identity_head.W"].shape[0])
        else:
            n_id = 1000
    n_mod = cfg.get("n_modification_classes", 10)
    # The backbone family must match what the checkpoint was trained with, else
    # the backbone weights silently fail to load (key mismatch) and embeddings
    # are garbage. Prefer the saved config; fall back to sniffing the keys
    # (pretrained w600k backbone stores ``backbone.net.*``).
    pretrained_bb = cfg.get("pretrained_backbone")
    if pretrained_bb is None:
        pretrained_bb = any(k.startswith("backbone.net.") for k in sd.keys())
    model = MDIE(
        n_identity_classes=n_id,
        n_modification_classes=n_mod,
        embedding_dim=cfg.get("embedding_dim", 512),
        use_region_prior=cfg.get("use_region_prior", has_rata),
        use_amd=cfg.get("use_amd", has_amd),
        amd_lambda=cfg.get("amd_lambda", 0.10),
        pretrained_backbone=pretrained_bb,
    ).to(device).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Guard: if the backbone weights didn't actually load, the eval is invalid.
    bb_missing = [k for k in missing if k.startswith("backbone.")]
    if bb_missing:
        print(f"  [warn] mdie backbone weights not loaded ({len(bb_missing)} keys "
              f"missing) — check pretrained_backbone setting")

    def enc(x):
        # Verification-time inference: a single 512-d, single-forward, unit-norm
        # fused embedding (bone-anchored attention + native identity merged in a
        # learned fusion head) -- an ArcFace drop-in. See MDIE.encode_verify.
        return model.encode_verify(x.to(device))
    return enc, "mdie"


def _pretrained_ir50_loader(device: torch.device):
    """Loads the public InsightFace IR-50 weights (downloads on first call)."""
    from ..pretrained import load_pretrained_ir50
    model = load_pretrained_ir50()
    if model is None:
        return None
    model = model.to(device).eval()

    def enc(x):
        with torch.no_grad():
            return model(x.to(device))["embedding"]
    return enc, "ir50_pretrained"


def _insightface_w600k_loader(device: torch.device):
    """Strong external baseline: InsightFace IR-50 trained on WebFace12M.

    Downloads ``Icar/buffalo_l-torch/w600k_r50.pth`` (~174 MB) on first use,
    then runs as a normal torch encoder. Inputs are expected to be the
    standard MTCNN-aligned 112×112 crops, normalised to (img-127.5)/128.0.
    """
    from ..models.iresnet import load_iresnet50_w600k
    model = load_iresnet50_w600k(device)
    if model is None:
        return None

    def enc(x):
        with torch.no_grad():
            return model(x.to(device))
    return enc, "insightface_w600k_r50"


MODEL_LOADERS: Dict[str, Callable[[torch.device], object]] = {
    "facenet":       lambda d: _baseline_loader("facenet", d),
    "arcface":       lambda d: _baseline_loader("arcface", d),
    "cosface":       lambda d: _baseline_loader("cosface", d),
    "mobilefacenet": lambda d: _baseline_loader("mobilefacenet", d),
    "mdie":          _mdie_loader,
    "ir50_pretrained": _pretrained_ir50_loader,
    "insightface_w600k_r50": _insightface_w600k_loader,
}


# ----------------------------------------------------------------------------

def _evaluate_one(enc, bench: Benchmark, device: torch.device,
                   batch_size: int = 64) -> dict:
    if not bench.pairs:
        return {"auc": float("nan"), "eer": float("nan"), "n_pairs": 0}
    pair_set = PairSet(pairs=bench.pairs)
    left, right, labels = extract_embeddings_for_pairs(
        encode_fn=enc, pair_set=pair_set, device=device,
        batch_size=batch_size, modification=None, apply_to="none",
    )
    scores = score_pairs(left, right)
    summary = summarize_run(scores, labels)
    summary["scores"] = scores.tolist()
    summary["labels"] = labels.tolist()
    return summary


def _plot_roc(per_model: Dict[str, dict], bench_name: str, out_pdf: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=200)
    for model_name, summary in per_model.items():
        if not summary or not np.isfinite(summary.get("auc", float("nan"))):
            continue
        ax.plot(summary["fpr"], summary["tpr"],
                label=f"{model_name}  AUC={summary['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.0); ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate (log scale)")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {bench_name}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["arcface", "cosface", "facenet", "mobilefacenet",
                            "mdie", "insightface_w600k_r50"])
    p.add_argument("--benchmarks", nargs="+", default=list_benchmarks())
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "real_benchmarks.csv"))
    args = p.parse_args()

    seed_all()
    device = get_device()
    print(f"[device] {device}")

    # 1. Materialise benchmarks (CPU-only).
    benches: List[Benchmark] = []
    for name in args.benchmarks:
        try:
            b = load_benchmark(name)
        except KeyError as e:
            print(f"[skip] {e}")
            continue
        print(f"[bench] {b.name}: {len(b.pairs)} pairs  ({b.notes})")
        if b.pairs:
            benches.append(b)

    if not benches:
        print("No benchmarks available — set IIITD_ROOT / IJBC_ROOT or "
              "ensure network access for mfr2/calfw/agedb30.")
        return

    # 2. Load models (skips silently if checkpoint or pretrained weights missing).
    models = {}
    for name in args.models:
        loader = MODEL_LOADERS.get(name)
        if loader is None:
            print(f"[skip-model] unknown {name}")
            continue
        try:
            result = loader(device)
        except Exception as e:  # noqa: BLE001
            print(f"[skip-model] {name}: {e}")
            continue
        if result is None:
            print(f"[skip-model] {name}: no checkpoint available")
            continue
        enc, friendly = result
        models[friendly] = enc
        print(f"[model] {friendly} ready")

    if not models:
        print("No models available — train baselines (run_stage1) and MDIE "
              "(run_stage2), or download pretrained weights first.")
        return

    # 3. Run.
    rows = []
    per_bench: Dict[str, Dict[str, dict]] = {}
    for bench in benches:
        per_bench[bench.name] = {}
        for mname, enc in models.items():
            print(f"  [eval] {mname} on {bench.name} ...")
            summary = _evaluate_one(enc, bench, device, batch_size=args.batch)
            per_bench[bench.name][mname] = summary
            rows.append({
                "benchmark": bench.name,
                "model": mname,
                "n_pairs": summary.get("n_pairs", 0),
                "auc": summary.get("auc", float("nan")),
                "eer": summary.get("eer", float("nan")),
                **{k: v for k, v in summary.items()
                   if k.startswith("tar_at_far")},
            })

    # 4. Write outputs.
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"[done] CSV -> {out_csv}")

    for bench_name, per_model in per_bench.items():
        out_pdf = FIGURES_DIR / f"roc_{bench_name}.pdf"
        _plot_roc(per_model, bench_name, out_pdf)
        print(f"[done] ROC -> {out_pdf}")

    # 5. Human-readable JSON dump (minus the raw curves for size).
    light = {b: {m: {k: v for k, v in s.items()
                      if k not in ("fpr", "tpr", "scores", "labels")}
                 for m, s in per_model.items()}
             for b, per_model in per_bench.items()}
    with open(RESULTS_DIR / "real_benchmarks.json", "w", encoding="utf-8") as f:
        json.dump(light, f, indent=2)
    print(f"[done] summary -> {RESULTS_DIR / 'real_benchmarks.json'}")


if __name__ == "__main__":
    main()
