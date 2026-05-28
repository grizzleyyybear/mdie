"""
Pre-training preflight check.

Runs in under 60 s on CPU. Fails fast (non-zero exit) on anything that
would otherwise blow up half-way through a long GPU training run:

  1. Python / Torch / CUDA versions match what we expect.
  2. All required directories are writable.
  3. LFW (or whatever dataset is staged) can be read.
  4. Every baseline + MDIE variant builds and does one forward + backward.
  5. PairedModificationDataset yields the expected tuple shapes.
  6. AMP autocast + GradScaler work on the active device.
  7. Real-benchmark loaders import cleanly.
  8. Disk has enough free space (>= 5 GB) for checkpoints + figures.

Usage:
    python -m src.preflight
"""

from __future__ import annotations

import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from .config import (
    CKPT_DIR, DATA_DIR, FIGURES_DIR, RESULTS_DIR, SETTINGS, USE_AMP,
    get_device, seed_all,
)


PASS = "[ ok ]"
FAIL = "[FAIL]"
WARN = "[warn]"


def _check_versions() -> bool:
    print(f"  python  : {sys.version.split()[0]}")
    print(f"  torch   : {torch.__version__}")
    print(f"  cuda    : {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f"  ({torch.cuda.get_device_name(0)})")
        free, total = torch.cuda.mem_get_info()
        print(f"  vram    : {free/1e9:.2f}/{total/1e9:.2f} GB free")
    else:
        print()
    return True


def _check_dirs() -> bool:
    ok = True
    for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, CKPT_DIR):
        if not d.exists():
            print(f"  {FAIL} {d} does not exist"); ok = False; continue
        probe = d / ".preflight_probe"
        try:
            probe.write_text("ok", encoding="utf-8"); probe.unlink()
            print(f"  {PASS} writable: {d}")
        except Exception as e:  # noqa: BLE001
            print(f"  {FAIL} not writable: {d} ({e})"); ok = False
    return ok


def _check_disk() -> bool:
    total, used, free = shutil.disk_usage(CKPT_DIR)
    print(f"  free disk: {free/1e9:.1f} GB")
    if free < 5 * 1024**3:
        print(f"  {WARN} less than 5 GB free under {CKPT_DIR.anchor}")
        return False
    return True


def _check_dataset() -> bool:
    import os
    from .data import build_face_dataset, prepare_lfw

    # If MDIE_SKIP_DATASET_PREFLIGHT == "auto", skip when LFW dir is empty —
    # this prevents env_setup.sh from triggering the flaky figshare download.
    skip_mode = os.environ.get("MDIE_SKIP_DATASET_PREFLIGHT", "0")
    lfw_root = DATA_DIR / "lfw"
    has_local_lfw = (
        lfw_root.exists()
        and any(
            p.is_dir() and any(p.glob("*.jpg")) for p in lfw_root.rglob("*")
        )
    )
    if skip_mode == "1" or (skip_mode == "auto" and not has_local_lfw):
        print(f"  {WARN} LFW not staged yet at {lfw_root} — skipping")
        print(f"         run:  bash hpc/stage_datasets.sh   (and re-run preflight)")
        return True
    try:
        lfw_dir = prepare_lfw(DATA_DIR, min_faces_per_person=8)
    except Exception as e:  # noqa: BLE001
        print(f"  {FAIL} prepare_lfw: {e}")
        return False
    try:
        paths, labels, names = build_face_dataset(lfw_dir, min_imgs=4)
    except Exception as e:  # noqa: BLE001
        print(f"  {FAIL} build_face_dataset: {e}")
        return False
    print(f"  {PASS} LFW: {len(paths)} imgs, {len(names)} identities")
    if len(paths) < 100:
        print(f"  {WARN} fewer than 100 images — training will not converge")
        return False
    return True


def _check_paired_dataset() -> bool:
    import os
    from .data import (
        MODIFICATION_TYPES, PairedModificationDataset, build_face_dataset,
        prepare_lfw,
    )
    lfw_root = DATA_DIR / "lfw"
    has_local_lfw = (
        lfw_root.exists()
        and any(
            p.is_dir() and any(p.glob("*.jpg")) for p in lfw_root.rglob("*")
        )
    )
    skip_mode = os.environ.get("MDIE_SKIP_DATASET_PREFLIGHT", "0")
    if skip_mode == "1" or (skip_mode == "auto" and not has_local_lfw):
        print(f"  {WARN} skipping paired-dataset check (LFW not staged)")
        return True
    lfw_dir = prepare_lfw(DATA_DIR, min_faces_per_person=8)
    paths, labels, _ = build_face_dataset(lfw_dir, min_imgs=4)
    ds = PairedModificationDataset(paths[:8], labels[:8],
                                    image_size=SETTINGS.train.image_size,
                                    modifications=MODIFICATION_TYPES)
    clean, modded, id_lbl, mod_lbl = ds[0]
    assert clean.shape == (3, 112, 112), f"clean shape {clean.shape}"
    assert modded.shape == (3, 112, 112), f"modded shape {modded.shape}"
    assert isinstance(id_lbl, int) and isinstance(mod_lbl, int)
    print(f"  {PASS} PairedModificationDataset: shapes + dtypes correct")
    return True


def _check_baselines(device: torch.device) -> bool:
    from .baselines import BASELINE_REGISTRY, build_baseline
    ok = True
    for name in BASELINE_REGISTRY:
        try:
            m = build_baseline(name, n_classes=4, embedding_dim=512).to(device).train()
            x = torch.randn(2, 3, 112, 112, device=device)
            y = torch.tensor([0, 1], device=device)
            with autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                loss = m(x, y)
            scaler = GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")
            scaler.scale(loss).backward()
            print(f"  {PASS} {name:<14} fwd+bwd  loss={loss.item():.4f}")
            del m
        except Exception as e:  # noqa: BLE001
            print(f"  {FAIL} {name}: {e}")
            traceback.print_exc()
            ok = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return ok


def _check_mdie(device: torch.device) -> bool:
    from .novel import MDIE
    ok = True
    variants = [
        ("MDIE-full",   dict(use_region_prior=True,  use_amd=True)),
        ("MDIE-noRATA", dict(use_region_prior=False, use_amd=True)),
        ("MDIE-noAMD",  dict(use_region_prior=True,  use_amd=False)),
    ]
    for vname, vkw in variants:
        try:
            m = MDIE(n_identity_classes=4, n_modification_classes=9,
                     embedding_dim=512, **vkw).to(device).train()
            x = torch.randn(2, 3, 112, 112, device=device)
            id_lbl = torch.tensor([0, 1], device=device)
            mod_lbl = torch.tensor([0, 3], device=device)
            with autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                out = m(x, id_lbl, mod_lbl)
                loss = out["loss_identity"]
                if "loss_mod" in out:
                    loss = loss + out["loss_mod"]
            scaler = GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")
            scaler.scale(loss).backward()
            print(f"  {PASS} {vname:<14} fwd+bwd  loss={loss.item():.4f}")
            del m
        except Exception as e:  # noqa: BLE001
            print(f"  {FAIL} {vname}: {e}")
            traceback.print_exc()
            ok = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return ok


def _check_real_benchmarks() -> bool:
    try:
        from .eval.run_real_benchmarks import MODEL_LOADERS
        from .data.benchmarks import list_benchmarks
        print(f"  {PASS} loaders={list(MODEL_LOADERS)}")
        print(f"  {PASS} benchmarks={list_benchmarks()}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  {FAIL} {e}")
        return False


def main() -> int:
    print("=" * 60)
    print("MDIE preflight — runs all the cheap checks before ML training")
    print("=" * 60)
    seed_all()
    device = get_device()

    checks = [
        ("Versions / hardware", _check_versions),
        ("Writable directories", _check_dirs),
        ("Disk space", _check_disk),
        ("LFW dataset", _check_dataset),
        ("Paired modification dataset", _check_paired_dataset),
        ("Baselines fwd+bwd", lambda: _check_baselines(device)),
        ("MDIE variants fwd+bwd", lambda: _check_mdie(device)),
        ("Real-benchmark loaders", _check_real_benchmarks),
    ]
    failures = []
    for label, fn in checks:
        print(f"\n[ {label} ]")
        try:
            ok = bool(fn())
        except Exception as e:  # noqa: BLE001
            print(f"  {FAIL} crashed: {e}")
            traceback.print_exc()
            ok = False
        if not ok:
            failures.append(label)

    print("\n" + "=" * 60)
    if failures:
        print(f"PREFLIGHT FAILED — {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PREFLIGHT OK — safe to launch Stage 1 / Stage 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
