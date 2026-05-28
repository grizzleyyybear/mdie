"""
Training loop for the four SOTA baselines (Stage 1).

Mixed-precision (bf16 on A100, fp16 elsewhere), gradient accumulation,
warmup→cosine LR, gradient clipping, NaN guard, channels-last,
resume-from-checkpoint. Tracks both best-by-train-loss and (optionally)
best-by-validation-AUC checkpoints.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..config import CKPT_DIR, RESULTS_DIR, SETTINGS, USE_AMP
from ..hw import autocast_dtype
from ..trainer_utils import (
    Throughput, is_finite_loss, load_resumable, save_resumable, set_lr,
    try_torch_compile, warmup_cosine_lr,
)
from .baseline_models import BaselineModel


def train_baseline(
    name: str,
    model: BaselineModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    grad_accum_steps: int = 1,
    grad_clip: float = 5.0,
    warmup_epochs: int = 1,
    resume: bool = False,
    model_config: dict | None = None,
    channels_last: bool = False,
    val_auc_fn: Callable[[], float] | None = None,
    compile_model: bool = False,
) -> dict:
    """
    Train one baseline. Writes:
      - ``checkpoints/baseline_<name>.pt``           best by train loss
      - ``checkpoints/baseline_<name>_best_auc.pt``  best by val-AUC (if provided)
      - ``checkpoints/baseline_<name>_last.pt``      last-epoch resumable snapshot
      - ``results/history_<name>.csv``               per-epoch metrics
    """
    model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = try_torch_compile(model, enable=compile_model)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # bf16 doesn't need a GradScaler; fp16 does.
    ac_dtype = autocast_dtype()
    use_scaler = USE_AMP and ac_dtype is torch.float16
    scaler = GradScaler("cuda", enabled=use_scaler)

    steps_per_epoch = max(len(train_loader) // max(grad_accum_steps, 1), 1)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * max(warmup_epochs, 0)

    best_ckpt = CKPT_DIR / f"baseline_{name}.pt"
    best_auc_ckpt = CKPT_DIR / f"baseline_{name}_best_auc.pt"
    last_ckpt = CKPT_DIR / f"baseline_{name}_last.pt"
    start_epoch = 1
    best_loss = float("inf")
    best_auc = -float("inf")

    if resume and last_ckpt.exists():
        meta = load_resumable(last_ckpt, model=model, optim=optim,
                              scaler=scaler if use_scaler else None,
                              map_location=device)
        start_epoch = meta["epoch"] + 1
        best_loss = meta["best_metric"]
        print(f"  [resume] {name}: starting from epoch {start_epoch}, "
              f"best_loss={best_loss:.4f}")

    history = {"train_loss": [], "val_loss": [], "val_auc": [],
               "epoch_time": [], "imgs_per_sec": []}
    global_step = (start_epoch - 1) * steps_per_epoch
    skipped_steps = 0

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tput = Throughput()
        model.train()
        running, n = 0.0, 0
        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            imgs = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            if channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            with autocast("cuda", dtype=ac_dtype, enabled=USE_AMP):
                loss = model(imgs, labels) / grad_accum_steps

            if not is_finite_loss(loss):
                skipped_steps += 1
                optim.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_accum_steps == 0:
                if grad_clip and grad_clip > 0:
                    if use_scaler:
                        scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if use_scaler:
                    scaler.step(optim); scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                set_lr(optim, warmup_cosine_lr(global_step, total_steps,
                                               warmup_steps, lr))
                global_step += 1
            running += loss.item() * grad_accum_steps * imgs.size(0)
            n += imgs.size(0)
            tput.add(imgs.size(0))
        train_loss = running / max(n, 1)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                v_run, v_n = 0.0, 0
                for imgs, labels in val_loader:
                    imgs = imgs.to(device); labels = labels.to(device)
                    if channels_last:
                        imgs = imgs.to(memory_format=torch.channels_last)
                    with autocast("cuda", dtype=ac_dtype, enabled=USE_AMP):
                        v = model(imgs, labels)
                    v_run += v.item() * imgs.size(0); v_n += imgs.size(0)
                val_loss = v_run / max(v_n, 1)

        val_auc = float("nan")
        if val_auc_fn is not None:
            model.eval()
            try:
                val_auc = float(val_auc_fn())
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] val_auc_fn failed: {e}")

        dt = time.time() - t0
        ips = tput.imgs_per_sec()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["epoch_time"].append(dt)
        history["imgs_per_sec"].append(ips)
        print(f"  [{name}] epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  "
              f"val={val_loss:.4f}  auc={val_auc:.4f}  "
              f"lr={optim.param_groups[0]['lr']:.2e}  "
              f"skip={skipped_steps}  {dt:.1f}s  {ips:.0f} img/s")

        if train_loss < best_loss:
            best_loss = train_loss
            save_resumable(best_ckpt, model=model, optim=optim,
                           scaler=scaler if use_scaler else None,
                           epoch=epoch, best_metric=best_loss,
                           model_config=model_config)
        if val_auc == val_auc and val_auc > best_auc:  # NaN-safe
            best_auc = val_auc
            save_resumable(best_auc_ckpt, model=model, optim=optim,
                           scaler=scaler if use_scaler else None,
                           epoch=epoch, best_metric=best_auc,
                           model_config=model_config)
        save_resumable(last_ckpt, model=model, optim=optim,
                       scaler=scaler if use_scaler else None,
                       epoch=epoch, best_metric=best_loss,
                       model_config=model_config)

    with open(RESULTS_DIR / f"history_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_auc",
                    "epoch_time_s", "imgs_per_sec"])
        for i in range(len(history["train_loss"])):
            w.writerow([start_epoch + i, history["train_loss"][i],
                        history["val_loss"][i], history["val_auc"][i],
                        history["epoch_time"][i], history["imgs_per_sec"][i]])

    history["skipped_steps"] = skipped_steps
    history["best_auc"] = best_auc
    return history
