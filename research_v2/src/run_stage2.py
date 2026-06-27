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
    IdentityBalancedSampler, MODIFICATION_TYPES, PairedModificationDataset, build_face_dataset,
    build_train_dataset, build_verification_pairs, make_loaders, prepare_lfw,
)
from .eval import (
    extract_embeddings_for_pairs, quick_verification_auc, score_pairs, summarize_run,
)
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
    ap.add_argument("--epochs", type=int, default=None)
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
    ap.add_argument("--dataset", choices=["lfw", "casia"], default="lfw",
                     help="training corpus (default: lfw — the safe fallback)")
    ap.add_argument("--backbone", choices=["ir50", "ir100"], default="ir50",
                     help="from-scratch backbone when --no-pretrained-backbone "
                          "is set (ignored when the pretrained w600k backbone is used)")
    ap.add_argument("--ddp", action="store_true",
                     help="enable DistributedDataParallel (launch with torchrun); "
                          "single-GPU/CPU stays the default")
    ap.add_argument("--ablation", action="store_true",
                     help="also train no-RATA, no-AMD, no-ICCL variants")
    ap.add_argument("--only-variant", dest="only_variant", default=None,
                     choices=["MDIE-full", "MDIE-noRATA", "MDIE-noAMD",
                              "MDIE-noICCL"],
                     help="train/evaluate only this single variant (for per-model "
                          "fan-out across GPUs); default: all selected variants")
    ap.add_argument("--pretrained-backbone", dest="pretrained_backbone",
                     action="store_true", default=True,
                     help="build MDIE on the production w600k IResNet50 backbone "
                          "(default: on — gives face-centric attention)")
    ap.add_argument("--no-pretrained-backbone", dest="pretrained_backbone",
                     action="store_false",
                     help="train the MDIE backbone from scratch instead")
    ap.add_argument("--freeze-backbone", dest="freeze_backbone",
                     action="store_true", default=False,
                     help="freeze the pretrained backbone (default: light fine-tune)")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1,
                     help="LR multiplier for the pretrained backbone (light fine-tune)")
    ap.add_argument("--attn-lambda", type=float, default=0.5,
                     help="weight for the RATA face-attention (anti-background) loss")
    args = ap.parse_args()

    if args.quick:
        args.epochs = args.epochs or 4
        SETTINGS.eval.n_pos_pairs = 400
        SETTINGS.eval.n_neg_pairs = 400
        args.val_pairs = min(args.val_pairs, 200)
    if args.epochs is None:
        args.epochs = 20

    # Wall-clock fail-safe for time-capped schedulers (e.g. SLURM 8 h walltime).
    # MDIE_TRAIN_MAX_SECONDS (seconds, measured from now) sets an absolute
    # deadline after which training stops cleanly at the next epoch boundary so a
    # resubmitted `--resume` job continues from the last checkpoint.
    import os as _os
    import time as _time
    _budget = _os.environ.get("MDIE_TRAIN_MAX_SECONDS")
    deadline_ts = (_time.time() + float(_budget)) if _budget else None
    if deadline_ts is not None:
        print(f"  [budget] training deadline in {float(_budget)/3600:.2f} h "
              f"(MDIE_TRAIN_MAX_SECONDS={_budget}); will stop at an epoch "
              f"boundary and exit 64 if not finished.")

    seed_all()
    tune_for_device()
    if args.ddp:
        from .train.ddp import setup_ddp
        rank, world_size, local_rank = setup_ddp()
        device = (torch.device(f"cuda:{local_rank}")
                  if torch.cuda.is_available() else get_device())
        print(f"  [ddp] rank {rank}/{world_size} local_rank {local_rank} device {device}")
    else:
        rank, world_size, local_rank = 0, 1, 0
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
    print(f"  Dataset: {args.dataset}")
    paths, labels, names = build_train_dataset(args.dataset, DATA_DIR, min_imgs=4)
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

    # --- per-image rigid bone-landmark attention targets --------------------
    # Detect each training face's actual bone landmarks (InsightFace 68-pt) and
    # cache a 7x7 attention target. This supervises RATA onto each individual's
    # real bone geometry (data-driven, varies per face) instead of a fixed
    # template. The detector is a *training-time teacher* only — inference needs
    # no landmarks. Cache is built once and reused.
    from .data.landmarks import (build_cache, load_target_cache,
                                 load_cache_version, GRID as BONE_GRID,
                                 TARGET_VERSION as BONE_VERSION)
    bone_cache_path = Path(DATA_DIR) / "bone_targets.npz"
    bone_targets = load_target_cache(bone_cache_path)
    cache_version = load_cache_version(bone_cache_path)
    # Rebuild if any path is missing, the cached grid resolution is stale
    # (e.g. an old 7x7 cache when the model now expects 14x14), or the target
    # definition changed (point set / placement / weights bumped the version).
    def _stale_grid(bt):
        for v in bt.values():
            if getattr(v, "shape", (0,))[-1] != BONE_GRID:
                return True
            break
        return False
    missing = [p for p in train_paths if str(p) not in bone_targets]
    stale_version = bool(bone_targets) and cache_version != BONE_VERSION
    need_build = bool(missing) or (bone_targets and _stale_grid(bone_targets)) or stale_version
    # Under DDP only rank 0 builds the shared cache; the others wait on a
    # barrier and then load the finished .npz from disk (avoids a write race).
    if need_build and (not args.ddp or rank == 0):
        if missing:
            why = "missing entries"
        elif stale_version:
            why = f"stale target version (need v{BONE_VERSION}, got v{cache_version})"
        else:
            why = f"stale grid (need {BONE_GRID})"
        print(f"  [landmarks] building bone-target cache ({why}) for "
              f"{len(train_paths)} images (one-off) ...")
        try:
            build_cache(train_paths, bone_cache_path, grid=BONE_GRID)
        except Exception as e:  # noqa: BLE001
            print(f"  [landmarks] cache build failed ({e}); "
                  "training without bone supervision")
    if args.ddp:
        from .train.ddp import barrier
        barrier()
    if need_build:
        bone_targets = load_target_cache(bone_cache_path) or None
    else:
        n_face = sum(1 for p in train_paths if bone_targets[str(p)].sum() > 0)
        print(f"  [landmarks] loaded bone targets for {len(train_paths)} images "
              f"({n_face} with a detected face)")

    paired_ds = PairedModificationDataset(train_paths, train_lbls,
                                           image_size=SETTINGS.train.image_size,
                                           modifications=MODIFICATION_TYPES,
                                           bone_targets=bone_targets,
                                           grid=BONE_GRID)
    if args.balanced_sampler:
        from torch.utils.data import DataLoader
        cpb = min(args.classes_per_batch,
                  max(1, args.batch // max(args.samples_per_class, 1)))
        try:
            bsamp = IdentityBalancedSampler(
                train_lbls, classes_per_batch=cpb,
                samples_per_class=args.samples_per_class,
                rank=local_rank, num_replicas=world_size)
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

    if args.only_variant is not None:
        variants = [v for v in variants if v[0] == args.only_variant]
        if not variants:
            raise SystemExit(
                f"--only-variant {args.only_variant} not in the selected set "
                f"(did you forget --ablation?)")
        print(f"  [fan-out] training only variant: {args.only_variant}")

    def _cname(v: str) -> str:
        """File-system safe checkpoint stem: 'MDIE-full' -> 'mdie_full'."""
        return v.lower().replace("-", "_")

    # Backbone configuration shared by every variant (train + eval rebuild).
    backbone_kwargs = dict(backbone=args.backbone,
                           pretrained_backbone=args.pretrained_backbone,
                           freeze_backbone=args.freeze_backbone)
    print(f"  Backbone: {'pretrained w600k IResNet50' if args.pretrained_backbone else 'from-scratch ' + args.backbone.upper()}"
          f"{'' if not args.pretrained_backbone else (' (frozen)' if args.freeze_backbone else f' (fine-tune x{args.backbone_lr_mult})')}")

    histories = {}
    for vname, vkwargs, iccl_l in variants:
        print(f"\n[*] training {vname}  (iccl_lambda={iccl_l})")
        model = MDIE(n_identity_classes=len(remap),
                     n_modification_classes=len(MODIFICATION_TYPES),
                     embedding_dim=SETTINGS.train.embedding_dim,
                     amd_lambda=SETTINGS.novel.amd_lambda,
                     **backbone_kwargs,
                     **vkwargs)
        n = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    params: {n/1e6:.2f}M  (trainable {n_trainable/1e6:.2f}M)")
        # Snapshot the architectural config so the eval loader can rebuild MDIE.
        model_config = dict(
            n_identity_classes=len(remap),
            n_modification_classes=len(MODIFICATION_TYPES),
            embedding_dim=SETTINGS.train.embedding_dim,
            amd_lambda=SETTINGS.novel.amd_lambda,
            **backbone_kwargs,
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
                        deadline_ts=deadline_ts,
                        channels_last=args.channels_last,
                        compile_model=args.compile_model,
                        backbone_lr_mult=args.backbone_lr_mult,
                        attn_lambda=args.attn_lambda,
                        ddp=args.ddp,
                        local_rank=local_rank,
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

    # Wall-clock fail-safe: if any variant stopped before its last epoch, skip
    # the (premature) eval/figures and exit 64 so the scheduler resubmits with
    # --resume instead of treating a partial model as the final result. Every
    # epoch was already checkpointed to <variant>_last.pt, so no work is lost.
    incomplete = [v for v, hh in histories.items() if not hh.get("completed", True)]
    if incomplete:
        print(f"\n[budget] stopped early before completion: {incomplete}")
        print("[budget] skipping eval/figures; resubmit with --resume to continue.")
        if args.ddp:
            from .train.ddp import cleanup_ddp
            cleanup_ddp()
        raise SystemExit(64)

    plot_training_curves(histories, FIGURES_DIR / "stage2_training_curves.png")

    # Under DDP, only rank 0 runs the (single-GPU) evaluation + figure/table
    # generation; the other ranks synchronise and exit cleanly here.
    if args.ddp:
        from .train.ddp import barrier, cleanup_ddp
        barrier()
        if rank != 0:
            cleanup_ddp()
            return

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

    # Strong production reference: the raw w600k IResNet50 (no MDIE components).
    # This is the honest, hard-to-dismiss baseline — MDIE is the same backbone
    # plus modification-invariance training, so beating this under degradation
    # is the actual contribution.
    try:
        from .models.iresnet import load_iresnet50_w600k
        ref_net = load_iresnet50_w600k(device)
    except Exception as e:  # noqa: BLE001
        print(f"    [skip] insightface_w600k_r50 reference unavailable: {e}")
        ref_net = None
    if ref_net is not None:
        ref_net.eval()

        def _ref_encode(x):
            with torch.no_grad():
                return ref_net(x.to(device))
        all_models_results["insightface_w600k_r50"] = _eval_all_mods(
            _ref_encode, pair_set, device)
        pooled_roc["insightface_w600k_r50"] = _pooled_roc(
            _ref_encode, pair_set, device)
        del ref_net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # novel variants
    for vname, vkwargs, _iccl in variants:
        model = MDIE(n_identity_classes=len(remap),
                     n_modification_classes=len(MODIFICATION_TYPES),
                     embedding_dim=SETTINGS.train.embedding_dim,
                     amd_lambda=SETTINGS.novel.amd_lambda,
                     **backbone_kwargs,
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
    if args.ddp:
        from .train.ddp import cleanup_ddp
        cleanup_ddp()


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
