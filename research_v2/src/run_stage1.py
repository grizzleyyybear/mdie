"""
Stage 1 entry-point: SOTA baseline failure-mode analysis.

Trains FaceNet, ArcFace, CosFace, MobileFaceNet on LFW and evaluates each on
clean + 8 modification protocols. Produces:

  - results/stage1_metrics.json        (all numerics)
  - figures/stage1_roc_*.png/.pdf      (ROC per modification)
  - figures/stage1_per_mod_auc.png     (bar chart)
  - figures/stage1_score_dist_*.png    (genuine vs impostor)
  - figures/stage1_occl_heatmap_*.png  (occlusion sensitivity)
  - results/stage1_tables.tex          (LaTeX tables 1, 2)

Usage:
    python -m src.run_stage1 --epochs 15 --batch 32
    python -m src.run_stage1 --quick                # 4 epochs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .baselines import BASELINE_REGISTRY, build_baseline, train_baseline
from .config import (
    CKPT_DIR, DATA_DIR, FIGURES_DIR, RESULTS_DIR, SETTINGS, get_device, seed_all,
)
from .data import (
    FaceClassificationDataset, MODIFICATION_TYPES,
    build_face_dataset, build_verification_pairs, make_loaders, prepare_lfw,
)
from .data.samplers import IdentityBalancedSampler
from .eval import (
    extract_embeddings_for_pairs, region_sensitivity_map, score_pairs, summarize_run,
)
from .eval.quick_val import quick_verification_auc
from .hw import describe_environment, recommend_preset, tune_for_device
from .paper import (
    plot_occlusion_heatmap, plot_per_modification_bars, plot_roc_curves,
    plot_score_distributions, plot_training_curves, write_latex_tables,
)


def _enc_fn(model, device):
    def fn(x):
        model.eval()
        return model.extract(x.to(device))
    return fn


def main():
    preset = recommend_preset()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=preset.batch_size,
                     help=f"default {preset.batch_size} (preset={preset.name})")
    ap.add_argument("--lr", type=float, default=preset.lr)
    ap.add_argument("--workers", type=int, default=preset.workers,
                     help="DataLoader workers (default: hardware preset)")
    ap.add_argument("--grad-clip", type=float, default=5.0,
                     help="L2 gradient-norm clip; 0 disables")
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--channels-last", action="store_true",
                     default=preset.channels_last,
                     help="use channels-last memory format (A100 win)")
    ap.add_argument("--no-channels-last", dest="channels_last",
                     action="store_false")
    ap.add_argument("--balanced-sampler", action="store_true",
                     help="identity-balanced sampler (classes x samples/class)")
    ap.add_argument("--classes-per-batch", type=int, default=16)
    ap.add_argument("--samples-per-class", type=int, default=4)
    ap.add_argument("--val-pairs", type=int, default=600,
                     help="verification pairs for per-epoch val-AUC (0 disables)")
    ap.add_argument("--compile", dest="compile_model", action="store_true",
                     help="wrap model with torch.compile (falls back on failure)")
    ap.add_argument("--resume", action="store_true",
                     help="resume each baseline from its _last.pt snapshot")
    ap.add_argument("--quick", action="store_true",
                     help="4 epochs, smaller pair set (sanity-check run)")
    ap.add_argument("--baselines", nargs="+", default=BASELINE_REGISTRY,
                     help=f"subset of {BASELINE_REGISTRY}")
    args = ap.parse_args()

    if args.quick:
        args.epochs = 4
        SETTINGS.eval.n_pos_pairs = 400
        SETTINGS.eval.n_neg_pairs = 400
        args.val_pairs = min(args.val_pairs, 200)

    seed_all()
    tune_for_device()
    device = get_device()
    from .trainer_utils import autodetect_workers, dump_run_manifest
    n_workers = autodetect_workers(args.workers)
    manifest = dump_run_manifest(RESULTS_DIR, run_name="stage1",
                                 settings=SETTINGS, extra=vars(args))
    print(f"\n========== Stage 1: SOTA baseline failure analysis ==========")
    print(f"  Env     : {describe_environment()}")
    print(f"  Preset  : {preset.name}  (batch={preset.batch_size}, lr={preset.lr})")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Workers : {n_workers}")
    print(f"  Models  : {args.baselines}")
    print(f"  Manifest: {manifest.name}\n")

    # ----- 1. Dataset --------------------------------------------------------
    print("[1/4] Preparing LFW ...")
    lfw_dir = prepare_lfw(DATA_DIR, min_faces_per_person=8)
    paths, labels, names = build_face_dataset(lfw_dir, min_imgs=4)
    n_classes = len(names)
    print(f"      {len(paths)} images, {n_classes} identities")

    # split: identity-disjoint train (80%) / test (20%) for verification pair pool
    rng = np.random.RandomState(0)
    perm_ids = rng.permutation(n_classes)
    train_ids = set(perm_ids[: int(0.8 * n_classes)].tolist())
    test_ids = set(perm_ids[int(0.8 * n_classes):].tolist())
    train_paths = [p for p, l in zip(paths, labels) if l in train_ids]
    train_lbls = [l for l in labels if l in train_ids]
    # remap train labels to contiguous range
    remap = {l: i for i, l in enumerate(sorted(train_ids))}
    train_lbls = [remap[l] for l in train_lbls]
    test_paths = [p for p, l in zip(paths, labels) if l in test_ids]
    test_lbls = [l for l in labels if l in test_ids]

    pair_set = build_verification_pairs(test_paths, test_lbls,
                                         n_pos=SETTINGS.eval.n_pos_pairs,
                                         n_neg=SETTINGS.eval.n_neg_pairs,
                                         seed=42)
    print(f"      verification pairs: {len(pair_set)}  (test ids: {len(test_ids)})")

    train_ds = FaceClassificationDataset(train_paths, train_lbls,
                                          image_size=SETTINGS.train.image_size,
                                          augment=True)
    if args.balanced_sampler:
        cpb = args.classes_per_batch or max(2, args.batch // max(args.samples_per_class, 1))
        # Need at least cpb usable classes.
        from torch.utils.data import DataLoader
        sampler = IdentityBalancedSampler(
            train_lbls, classes_per_batch=cpb,
            samples_per_class=args.samples_per_class,
            num_batches=max(1, len(train_paths) // (cpb * args.samples_per_class)),
            seed=42)
        pin = torch.cuda.is_available()
        train_loader = DataLoader(
            train_ds, batch_sampler=sampler, num_workers=n_workers,
            pin_memory=pin, persistent_workers=n_workers > 0)
        print(f"      balanced sampler: {cpb} classes x "
              f"{args.samples_per_class} samples = batch {cpb * args.samples_per_class}")
    else:
        train_loader, _ = make_loaders(train_ds, None, batch_size=args.batch,
                                        num_workers=n_workers)

    # Per-epoch verification AUC on a tiny disjoint pair pool (best-by-AUC checkpoint).
    val_auc_pairs = None
    if args.val_pairs > 0:
        val_n = max(args.val_pairs // 2, 50)
        val_pair_set = build_verification_pairs(
            test_paths, test_lbls, n_pos=val_n, n_neg=val_n, seed=13)
        val_auc_pairs = list(val_pair_set.pairs)
        print(f"      val-AUC pairs: {len(val_auc_pairs)}")

    # ----- 2. Train each baseline -------------------------------------------
    histories = {}
    for name in args.baselines:
        print(f"\n[2/4] Training baseline: {name}")
        n_train_classes = len(remap)
        model = build_baseline(name, n_classes=n_train_classes,
                                embedding_dim=SETTINGS.train.embedding_dim)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"      params: {n_params/1e6:.2f}M")

        val_fn = None
        if val_auc_pairs is not None:
            def _vf(m=model, n=name):
                enc = _enc_fn(m, device)
                return quick_verification_auc(enc, val_auc_pairs, device,
                                              batch_size=max(32, args.batch))
            val_fn = _vf

        h = train_baseline(name, model, train_loader, None, device,
                            epochs=args.epochs, lr=args.lr,
                            grad_accum_steps=SETTINGS.train.grad_accum_steps,
                            grad_clip=args.grad_clip,
                            warmup_epochs=args.warmup_epochs,
                            resume=args.resume,
                            channels_last=args.channels_last,
                            compile_model=args.compile_model,
                            val_auc_fn=val_fn,
                            model_config={"name": name,
                                          "n_classes": n_train_classes,
                                          "embedding_dim": SETTINGS.train.embedding_dim})
        histories[name] = h
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_training_curves(histories, FIGURES_DIR / "stage1_training_curves.png")

    # ----- 3. Evaluate each baseline on each modification --------------------
    print("\n[3/4] Evaluating ...")
    overall = {}
    per_mod = {name: {} for name in args.baselines}

    # ROC pooled across modifications for the headline figure
    headline_roc_data = {}

    for name in args.baselines:
        print(f"  evaluating {name}")
        model = build_baseline(name,
                                n_classes=len(remap),
                                embedding_dim=SETTINGS.train.embedding_dim)
        ckpt_path = CKPT_DIR / f"baseline_{name}.pt"
        if not ckpt_path.exists():  # backward-compat for old runs
            legacy = CKPT_DIR / f"{name}.pt"
            if legacy.exists():
                ckpt_path = legacy
        from .trainer_utils import extract_state_dict
        sd = extract_state_dict(ckpt_path, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        model.to(device).eval()
        encode = _enc_fn(model, device)

        all_scores, all_labels = [], []
        for mod in MODIFICATION_TYPES:
            le, re, lbl = extract_embeddings_for_pairs(
                encode, pair_set, device=device,
                batch_size=64, image_size=SETTINGS.train.image_size,
                modification=mod, apply_to="right",
            )
            sc = score_pairs(le, re)
            summary = summarize_run(sc, lbl)
            per_mod[name][mod] = summary
            all_scores.append(sc); all_labels.append(lbl)
            print(f"    [{name:<14}] mod={mod:<18} AUC={summary['auc']:.3f}  "
                  f"EER={summary['eer']:.3f}  TAR@1e-3={summary['tar_at_far=0.001']:.3f}")
        # overall pooled
        all_scores = np.concatenate(all_scores); all_labels = np.concatenate(all_labels)
        overall[name] = summarize_run(all_scores, all_labels)
        headline_roc_data[name] = {
            "fpr": overall[name]["fpr"], "tpr": overall[name]["tpr"],
            "auc": overall[name]["auc"],
        }

        # score distribution figure (clean only)
        clean = per_mod[name]["clean"]
        scores_clean = score_pairs(*extract_embeddings_for_pairs(
            encode, pair_set, device=device, modification="clean",
            image_size=SETTINGS.train.image_size)[:2])
        gen = scores_clean[np.array([l for _, _, l in pair_set.pairs]) == 1]
        imp = scores_clean[np.array([l for _, _, l in pair_set.pairs]) == 0]
        plot_score_distributions(gen, imp,
            FIGURES_DIR / f"stage1_score_dist_{name}.png", model_name=name)

        # occlusion sensitivity heatmap (small subset, fast)
        n_subset = min(60, len(pair_set))
        sub_pair_idx = []
        path_index = {str(p): i for i, p in enumerate(test_paths)}
        for p1, p2, _ in pair_set.pairs[:n_subset]:
            if str(p1) in path_index and str(p2) in path_index:
                sub_pair_idx.append((path_index[str(p1)], path_index[str(p2)]))
        if len(sub_pair_idx) > 5:
            drops = region_sensitivity_map(encode, [Path(p) for p in test_paths],
                                            sub_pair_idx, device=device, grid=7,
                                            image_size=SETTINGS.train.image_size)
            plot_occlusion_heatmap(drops,
                FIGURES_DIR / f"stage1_occl_heatmap_{name}.png", model_name=name)
            np.save(RESULTS_DIR / f"stage1_occl_{name}.npy", drops)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- 4. Figures + tables ----------------------------------------------
    print("\n[4/4] Producing publication artefacts ...")
    plot_roc_curves(headline_roc_data,
                     FIGURES_DIR / "stage1_roc_pooled.png",
                     title="Stage 1: SOTA baselines pooled across modifications")

    # Per-modification AUC bar chart
    auc_per_mod = {name: {mod: per_mod[name][mod]["auc"] for mod in MODIFICATION_TYPES}
                    for name in args.baselines}
    plot_per_modification_bars(auc_per_mod, "auc",
                                FIGURES_DIR / "stage1_per_mod_auc.png",
                                title="Stage 1: AUC per modification")

    # Per-modification EER bar chart
    eer_per_mod = {name: {mod: per_mod[name][mod]["eer"] for mod in MODIFICATION_TYPES}
                    for name in args.baselines}
    plot_per_modification_bars(eer_per_mod, "eer",
                                FIGURES_DIR / "stage1_per_mod_eer.png",
                                title="Stage 1: EER per modification (lower is better)")

    # Save all numeric results
    json_safe = {
        "overall": {k: {kk: vv for kk, vv in v.items() if kk not in ("fpr", "tpr")}
                     for k, v in overall.items()},
        "per_modification": {
            name: {mod: {kk: vv for kk, vv in d.items() if kk not in ("fpr", "tpr")}
                    for mod, d in mods.items()}
            for name, mods in per_mod.items()
        },
    }
    (RESULTS_DIR / "stage1_metrics.json").write_text(
        json.dumps(json_safe, indent=2), encoding="utf-8")

    write_latex_tables({
        "overall": json_safe["overall"],
        "per_modification": json_safe["per_modification"],
    }, RESULTS_DIR / "stage1_tables.tex")

    print(f"\n[done] Artefacts under {FIGURES_DIR} and {RESULTS_DIR}")


if __name__ == "__main__":
    main()
