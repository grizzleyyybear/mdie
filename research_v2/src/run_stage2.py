"""
Stage 2 entry-point: train the proposed MDIE model and compare to baselines.

Assumes Stage 1 was already run (baseline checkpoints + metrics on disk).
Adds:
  - MDIE training (paired-modification dataloader + ICCL + adversarial AMD)
  - Ablation study (no-RATA, no-AMD, no-ICCL, full)
  - Final publication figures + LaTeX tables (Tables 1, 2, 3)

Usage:
    python -m src.run_stage2 --epochs 20
    python -m src.run_stage2 --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .baselines import build_baseline
from .config import (
    CKPT_DIR, DATA_DIR, FIGURES_DIR, RESULTS_DIR, SETTINGS, get_device, seed_all,
)
from .data import (
    MODIFICATION_TYPES, PairedModificationDataset, build_face_dataset,
    build_verification_pairs, make_loaders, prepare_lfw,
)
from .data.samplers import IdentityBalancedSampler
from .eval import extract_embeddings_for_pairs, score_pairs, summarize_run
from .eval.quick_val import quick_verification_auc
from .hw import describe_environment, recommend_preset, tune_for_device
from .novel import MDIE, train_mdie
from .paper import (
    plot_attention_overlay, plot_per_modification_bars, plot_roc_curves,
    plot_training_curves, write_latex_tables,
)


def _enc(model, device, is_mdie=False):
    def fn(x):
        model.eval()
        if is_mdie:
            emb, _ = model.encode(x.to(device))
            return emb
        return model.extract(x.to(device))
    return fn


def main():
    preset = recommend_preset()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=preset.batch_size,
                     help=f"default {preset.batch_size} (preset={preset.name})")
    ap.add_argument("--lr", type=float, default=preset.lr)
    ap.add_argument("--workers", type=int, default=preset.workers,
                     help="DataLoader workers (default: hardware preset)")
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--channels-last", action="store_true",
                     default=preset.channels_last)
    ap.add_argument("--no-channels-last", dest="channels_last",
                     action="store_false")
    ap.add_argument("--val-pairs", type=int, default=600,
                     help="Verification pairs used for per-epoch val-AUC (0 disables)")
    ap.add_argument("--balanced-sampler", action="store_true",
                     help="identity-balanced sampler (classes x samples/class)")
    ap.add_argument("--classes-per-batch", type=int, default=16)
    ap.add_argument("--samples-per-class", type=int, default=4)
    ap.add_argument("--compile", dest="compile_model", action="store_true",
                     help="wrap model with torch.compile (falls back on failure)")
    ap.add_argument("--resume", action="store_true",
                     help="resume each variant from its _last.pt snapshot")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ablation", action="store_true",
                     help="also train no-RATA, no-AMD, no-ICCL variants")
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
    manifest = dump_run_manifest(RESULTS_DIR, run_name="stage2",
                                 settings=SETTINGS, extra=vars(args))
    print(f"\n========== Stage 2: Train + evaluate MDIE ==========")
    print(f"  Env    : {describe_environment()}")
    print(f"  Preset : {preset.name}  (batch={preset.batch_size}, lr={preset.lr})")
    print(f"  Epochs : {args.epochs}")
    print(f"  Batch  : {args.batch}")
    print(f"  Workers: {n_workers}")
    print(f"  Manifest: {manifest.name}\n")

    # --- dataset (same protocol as Stage 1) ---------------------------------
    lfw_dir = prepare_lfw(DATA_DIR, min_faces_per_person=8)
    paths, labels, names = build_face_dataset(lfw_dir, min_imgs=4)
    n_classes = len(names)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n_classes)
    train_ids = set(perm[: int(0.8 * n_classes)].tolist())
    test_ids = set(perm[int(0.8 * n_classes):].tolist())
    train_paths = [p for p, l in zip(paths, labels) if l in train_ids]
    train_lbls = [l for l in labels if l in train_ids]
    remap = {l: i for i, l in enumerate(sorted(train_ids))}
    train_lbls = [remap[l] for l in train_lbls]
    test_paths = [p for p, l in zip(paths, labels) if l in test_ids]
    test_lbls = [l for l in labels if l in test_ids]
    pair_set = build_verification_pairs(test_paths, test_lbls,
                                         n_pos=SETTINGS.eval.n_pos_pairs,
                                         n_neg=SETTINGS.eval.n_neg_pairs, seed=42)
    print(f"  pairs: {len(pair_set)}  classes: {len(remap)}")

    # Smaller disjoint pool for per-epoch val-AUC.
    val_auc_pairs = None
    if args.val_pairs > 0:
        val_n = max(args.val_pairs // 2, 50)
        val_pair_set = build_verification_pairs(
            test_paths, test_lbls, n_pos=val_n, n_neg=val_n, seed=13)
        val_auc_pairs = list(val_pair_set.pairs)
        print(f"  val-AUC pairs: {len(val_auc_pairs)}")

    paired_ds = PairedModificationDataset(train_paths, train_lbls,
                                           image_size=SETTINGS.train.image_size,
                                           modifications=MODIFICATION_TYPES)
    if args.balanced_sampler:
        from torch.utils.data import DataLoader
        cpb = min(args.classes_per_batch,
                  max(1, args.batch // max(args.samples_per_class, 1)))
        try:
            bsamp = IdentityBalancedSampler(
                train_lbls, classes_per_batch=cpb,
                samples_per_class=args.samples_per_class)
            paired_loader = DataLoader(
                paired_ds, batch_sampler=bsamp, num_workers=n_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(n_workers > 0),
                prefetch_factor=(4 if n_workers > 0 else None))
            print(f"      balanced sampler: {cpb} classes x "
                  f"{args.samples_per_class} samples ({len(bsamp)} batches/epoch)")
        except ValueError as e:
            print(f"      [warn] balanced sampler unavailable ({e}); "
                  "falling back to random shuffle")
            paired_loader, _ = make_loaders(paired_ds, None,
                                             batch_size=args.batch,
                                             num_workers=n_workers)
    else:
        paired_loader, _ = make_loaders(paired_ds, None,
                                         batch_size=args.batch,
                                         num_workers=n_workers)

    # --- variants to train --------------------------------------------------
    # Each variant: (display_name, model_kwargs, iccl_lambda)
    variants = [("MDIE-full", dict(use_region_prior=True, use_amd=True),
                 SETTINGS.novel.iccl_lambda)]
    if args.ablation:
        variants += [
            ("MDIE-noRATA", dict(use_region_prior=False, use_amd=True),
             SETTINGS.novel.iccl_lambda),
            ("MDIE-noAMD",  dict(use_region_prior=True,  use_amd=False),
             SETTINGS.novel.iccl_lambda),
            ("MDIE-noICCL", dict(use_region_prior=True,  use_amd=True), 0.0),
        ]

    def _cname(v: str) -> str:
        """File-system safe checkpoint stem: 'MDIE-full' -> 'mdie_full'."""
        return v.lower().replace("-", "_")

    histories = {}
    for vname, vkwargs, iccl_l in variants:
        print(f"\n[*] training {vname}  (iccl_lambda={iccl_l})")
        model = MDIE(n_identity_classes=len(remap),
                     n_modification_classes=len(MODIFICATION_TYPES),
                     embedding_dim=SETTINGS.train.embedding_dim,
                     amd_lambda=SETTINGS.novel.amd_lambda,
                     **vkwargs)
        n = sum(p.numel() for p in model.parameters())
        print(f"    params: {n/1e6:.2f}M")
        # Snapshot the architectural config so the eval loader can rebuild MDIE.
        model_config = dict(
            n_identity_classes=len(remap),
            n_modification_classes=len(MODIFICATION_TYPES),
            embedding_dim=SETTINGS.train.embedding_dim,
            amd_lambda=SETTINGS.novel.amd_lambda,
            **vkwargs,
        )
        h = train_mdie(model, paired_loader, device, epochs=args.epochs,
                        lr=args.lr,
                        iccl_lambda=iccl_l,
                        grad_accum_steps=SETTINGS.train.grad_accum_steps,
                        name=_cname(vname),
                        model_config=model_config,
                        grad_clip=args.grad_clip,
                        warmup_epochs=args.warmup_epochs,
                        resume=args.resume,
                        channels_last=args.channels_last,
                        compile_model=args.compile_model,
                        val_auc_fn=(
                            (lambda m=model: quick_verification_auc(
                                lambda x: m.encode(x)[0], val_auc_pairs, device,
                                batch_size=max(32, args.batch)))
                            if val_auc_pairs is not None else None
                        ))
        histories[vname] = h
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_training_curves(histories, FIGURES_DIR / "stage2_training_curves.png")

    # --- evaluate every variant + the four baselines on every modification --
    print("\n[*] evaluating ...")
    all_models_results = {}    # for per-mod table + headline ROC
    pooled_roc = {}

    # baselines (load from Stage 1 checkpoints if present)
    for bname in ("facenet", "arcface", "cosface", "mobilefacenet"):
        ckpt = CKPT_DIR / f"baseline_{bname}.pt"
        if not ckpt.exists():
            legacy = CKPT_DIR / f"{bname}.pt"
            if legacy.exists():
                ckpt = legacy
            else:
                print(f"    [skip] no checkpoint for baseline {bname}; run Stage 1 first")
                continue
        sd_b = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd_b, dict) and "model" in sd_b:
            sd_b = sd_b["model"]
        # Infer head class count from checkpoint so legacy/mismatched n_classes
        # still loads cleanly (head weights are irrelevant for inference).
        n_cls = len(remap)
        if "head.W" in sd_b:
            n_cls = int(sd_b["head.W"].shape[0])
        elif "head.weight" in sd_b:
            n_cls = int(sd_b["head.weight"].shape[0])
        model = build_baseline(bname, n_classes=n_cls,
                                embedding_dim=SETTINGS.train.embedding_dim)
        model.load_state_dict(sd_b, strict=False)
        model.to(device).eval()
        encode = _enc(model, device, is_mdie=False)
        all_models_results[bname] = _eval_all_mods(encode, pair_set, device)
        pooled_roc[bname] = _pooled_roc(encode, pair_set, device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # novel variants
    for vname, vkwargs, _iccl in variants:
        model = MDIE(n_identity_classes=len(remap),
                     n_modification_classes=len(MODIFICATION_TYPES),
                     embedding_dim=SETTINGS.train.embedding_dim,
                     amd_lambda=SETTINGS.novel.amd_lambda,
                     **vkwargs)
        ckpt = CKPT_DIR / f"{_cname(vname)}.pt"
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd)
        model.to(device).eval()
        encode = _enc(model, device, is_mdie=True)
        all_models_results[vname] = _eval_all_mods(encode, pair_set, device)
        pooled_roc[vname] = _pooled_roc(encode, pair_set, device)

        # save an attention overlay figure for the paper (full model only)
        if vname == "MDIE-full":
            _save_attention_overlay(model, test_paths[:1], device,
                                     FIGURES_DIR / "stage2_attention_examples.png")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- figures ------------------------------------------------------------
    plot_roc_curves(pooled_roc, FIGURES_DIR / "stage2_roc_pooled.png",
                     title="Stage 2: pooled ROC across modifications")

    auc_per_mod = {m: {mod: r[mod]["auc"] for mod in MODIFICATION_TYPES}
                    for m, r in all_models_results.items()}
    plot_per_modification_bars(auc_per_mod, "auc",
                                FIGURES_DIR / "stage2_per_mod_auc.png",
                                title="AUC per modification — baselines vs MDIE")
    eer_per_mod = {m: {mod: r[mod]["eer"] for mod in MODIFICATION_TYPES}
                    for m, r in all_models_results.items()}
    plot_per_modification_bars(eer_per_mod, "eer",
                                FIGURES_DIR / "stage2_per_mod_eer.png",
                                title="EER per modification (lower is better)")

    # --- tables -------------------------------------------------------------
    overall = {m: {kk: r["pooled"].get(kk) for kk in
                    ("auc", "eer", "tar_at_far=0.01", "tar_at_far=0.001")}
                for m, r in all_models_results.items()}
    per_mod_table = {m: {mod: {"auc": r[mod]["auc"]} for mod in MODIFICATION_TYPES}
                      for m, r in all_models_results.items()}
    ablation = {v: overall[v] for v, _, _ in variants if v in overall}

    write_latex_tables({
        "overall": overall,
        "per_modification": per_mod_table,
        "ablation": ablation,
    }, RESULTS_DIR / "stage2_tables.tex")

    json_safe = {m: {mod: {k: v for k, v in d.items() if k not in ("fpr", "tpr")}
                      for mod, d in r.items()}
                  for m, r in all_models_results.items()}
    (RESULTS_DIR / "stage2_metrics.json").write_text(
        json.dumps(json_safe, indent=2), encoding="utf-8")

    print(f"\n[done] Stage 2 artefacts under {FIGURES_DIR} and {RESULTS_DIR}")


def _eval_all_mods(encode, pair_set, device):
    out = {}
    pooled_s, pooled_l = [], []
    for mod in MODIFICATION_TYPES:
        le, re, lbl = extract_embeddings_for_pairs(
            encode, pair_set, device=device, batch_size=64,
            image_size=SETTINGS.train.image_size,
            modification=mod, apply_to="right",
        )
        sc = score_pairs(le, re)
        out[mod] = summarize_run(sc, lbl)
        pooled_s.append(sc); pooled_l.append(lbl)
    out["pooled"] = summarize_run(np.concatenate(pooled_s), np.concatenate(pooled_l))
    return out


def _pooled_roc(encode, pair_set, device):
    pooled_s, pooled_l = [], []
    for mod in MODIFICATION_TYPES:
        le, re, lbl = extract_embeddings_for_pairs(
            encode, pair_set, device=device, batch_size=64,
            image_size=SETTINGS.train.image_size,
            modification=mod, apply_to="right",
        )
        pooled_s.append(score_pairs(le, re)); pooled_l.append(lbl)
    summ = summarize_run(np.concatenate(pooled_s), np.concatenate(pooled_l))
    return {"fpr": summ["fpr"], "tpr": summ["tpr"], "auc": summ["auc"]}


def _save_attention_overlay(model, sample_paths, device, out_path):
    from .data.modifications import apply_modification
    import torch
    sample = []
    for p in sample_paths[:1]:
        img = cv2.imread(str(p)); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SETTINGS.train.image_size, SETTINGS.train.image_size))
        sample.append(img)
    if not sample:
        return
    x = torch.from_numpy(np.stack(sample).astype(np.float32)).permute(0, 3, 1, 2)
    x = (x - 127.5) / 128.0
    with torch.no_grad():
        emb, attn = model.encode(x.to(device))
    if attn is None:
        return
    plot_attention_overlay(sample[0], attn[0].cpu().numpy(), out_path,
                            title="MDIE-full Region-Aware Token Attention")


if __name__ == "__main__":
    main()
