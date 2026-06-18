"""
Merge the per-variant outputs produced by the Stage-2 fan-out (hpc/
slurm_fanout_train.sh) into a single combined metrics file and ablation table.

Each fan-out task trains one MDIE variant with ``--only-variant`` and writes its
own ``research_v2/results/fanout/<slug>/stage2_metrics.json`` (containing the
shared baselines plus that single variant). This script gathers them, keeps the
baselines once, stitches in every variant, and emits:

  research_v2/results/fanout/stage2_metrics_merged.json   (full per-mod metrics)
  research_v2/results/fanout/ablation_merged.json         (pooled ablation table)

It is read-only with respect to the individual fan-out dirs and never touches
the default (non-fan-out) results, so nothing in the LFW pipeline changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import RESULTS_DIR

VARIANTS = ["MDIE-full", "MDIE-noRATA", "MDIE-noAMD", "MDIE-noICCL"]
_POOLED_KEYS = ("auc", "eer", "tar_at_far=0.01", "tar_at_far=0.001")


def _slug(variant: str) -> str:
    return variant.lower().replace("-", "_")


def merge(fanout_dir: Path | None = None) -> dict:
    fanout_dir = fanout_dir or (RESULTS_DIR / "fanout")
    merged: dict = {}
    ablation: dict = {}
    found = []

    for variant in VARIANTS:
        metrics_path = fanout_dir / _slug(variant) / "stage2_metrics.json"
        if not metrics_path.exists():
            print(f"  [merge] skip {variant}: {metrics_path} not found")
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        found.append(variant)

        # Keep the shared baselines once (from the first file that has them).
        for model_name, per_mod in data.items():
            if not model_name.startswith("MDIE-") and model_name not in merged:
                merged[model_name] = per_mod

        # Stitch in this run's own variant.
        if variant in data:
            merged[variant] = data[variant]
            pooled = data[variant].get("pooled", {})
            ablation[variant] = {k: pooled.get(k) for k in _POOLED_KEYS}
        else:
            print(f"  [merge] warning: {variant} absent from {metrics_path}")

    if not found:
        raise SystemExit(
            f"no fan-out metrics found under {fanout_dir}; run the fan-out "
            "array first (hpc/slurm_fanout_train.sh)")

    fanout_dir.mkdir(parents=True, exist_ok=True)
    (fanout_dir / "stage2_metrics_merged.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8")
    (fanout_dir / "ablation_merged.json").write_text(
        json.dumps(ablation, indent=2), encoding="utf-8")

    print(f"[merge] combined {len(found)} variants: {', '.join(found)}")
    print(f"[merge] wrote {fanout_dir / 'stage2_metrics_merged.json'}")
    print(f"[merge] wrote {fanout_dir / 'ablation_merged.json'}")
    for v, m in ablation.items():
        print(f"    {v:14s} pooled AUC={m.get('auc')}")
    return {"merged": merged, "ablation": ablation}


def main() -> None:
    merge()


if __name__ == "__main__":
    main()
