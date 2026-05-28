"""
Re-run evaluation from a saved checkpoint without retraining.

Useful on the A100 server when you want to:
  * sanity-check a long run that died after epoch 30 of 50
  * try a new metric / pair set without re-training
  * compare ``_best_auc.pt`` vs ``_last.pt`` of the same variant

Examples
--------
# Re-evaluate the best-by-AUC MDIE checkpoint on the LFW pair pool:
python -m research_v2.src.eval.eval_from_ckpt \
    --ckpt research_v2/checkpoints/mdie_full_best_auc.pt --model mdie

# Evaluate an ArcFace baseline checkpoint:
python -m research_v2.src.eval.eval_from_ckpt \
    --ckpt research_v2/checkpoints/baseline_arcface.pt --model arcface
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..baselines import build_baseline
from ..config import (
    CKPT_DIR, DATA_DIR, RESULTS_DIR, SETTINGS, get_device, seed_all,
)
from ..data import (
    build_face_dataset, build_verification_pairs, prepare_lfw,
)
from ..eval import extract_embeddings_for_pairs, score_pairs, summarize_run
from ..novel import MDIE


def _load_ckpt(path: Path, map_location):
    state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        return state["model"], state.get("config", {})
    return state, {}


def build_model(name: str, config: dict, n_classes: int, device):
    name = name.lower()
    if name == "mdie":
        cfg = dict(config)
        cfg.setdefault("n_identity_classes", n_classes)
        cfg.setdefault("n_modification_classes", 9)
        cfg.setdefault("embedding_dim", SETTINGS.train.embedding_dim)
        cfg.setdefault("amd_lambda", SETTINGS.novel.amd_lambda)
        cfg.pop("name", None)
        return MDIE(**cfg).to(device)
    return build_baseline(name, n_classes=n_classes,
                          embedding_dim=SETTINGS.train.embedding_dim).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--model", required=True,
                     choices=["facenet", "arcface", "cosface",
                              "mobilefacenet", "mdie"])
    ap.add_argument("--n-pos", type=int, default=3000)
    ap.add_argument("--n-neg", type=int, default=3000)
    ap.add_argument("--tag", default=None,
                     help="suffix added to result filenames")
    args = ap.parse_args()

    seed_all()
    device = get_device()
    print(f"[eval] {args.ckpt}  ({args.model})  on {device}")

    lfw_dir = prepare_lfw(DATA_DIR, min_faces_per_person=8)
    paths, labels, names = build_face_dataset(lfw_dir, min_imgs=4)
    n_classes = len(names)
    pairs = build_verification_pairs(paths, labels,
                                      n_pos=args.n_pos, n_neg=args.n_neg, seed=42)
    print(f"  {len(paths)} images / {n_classes} ids / {len(pairs)} pairs")

    state_sd, config = _load_ckpt(args.ckpt, device)
    model = build_model(args.model, config, n_classes, device)
    missing, unexpected = model.load_state_dict(state_sd, strict=False)
    print(f"  loaded ({len(missing)} missing, {len(unexpected)} unexpected keys)")
    model.eval()

    encode = (lambda x: model.encode(x)[0]) if args.model == "mdie" \
             else (lambda x: model.extract(x))
    embs = extract_embeddings_for_pairs(encode, pairs, device)
    metrics = summarize_run(*score_pairs(embs, pairs))
    print(f"  AUC={metrics['auc']:.4f}  EER={metrics['eer']:.4f}  "
          f"FAR@FRR=1e-3 ~ {metrics.get('far_at_frr_1e-3', float('nan')):.4f}")

    tag = args.tag or args.ckpt.stem
    out = RESULTS_DIR / f"eval_from_ckpt_{tag}.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
