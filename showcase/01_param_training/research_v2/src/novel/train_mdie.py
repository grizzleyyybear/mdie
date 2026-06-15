"""
Trainer for the MDIE model.

Loss = ArcFace(identity)
       + λ_iccl  · ICCL(clean_emb, modified_emb, identity_labels)
       - λ_amd   · CE(modification_classifier — via gradient reversal)

The negative sign on the AMD term is implemented inside the GradientReversal layer.
ICCL = Identity-Consistency Contrastive Loss with modification-aware mining.
"""

from __future__ import annotations

import csv
import time

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..config import CKPT_DIR, RESULTS_DIR, USE_AMP
from .mdie import MDIE


def identity_consistency_contrastive_loss(
    clean_emb: torch.Tensor,
    modified_emb: torch.Tensor,
    identity_labels: torch.Tensor,
    modification_labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Pulls (clean, modified) of the same identity together; pushes apart distinct
    identities. The hardest negative term emphasizes the *same modification* on
    a different identity (modification-aware mining).
    """
    # InfoNCE between clean and modified
    z = F.normalize(clean_emb, dim=1)
    zp = F.normalize(modified_emb, dim=1)
    logits = (z @ zp.t()) / temperature                              # (B, B)
    targets = torch.arange(z.size(0), device=z.device)
    base = F.cross_entropy(logits, targets)

    # Modification-aware penalty: penalize high similarity between (clean_i, modified_j)
    # when identity differs but modification matches (the genuinely confusing case).
    same_mod = modification_labels.unsqueeze(0) == modification_labels.unsqueeze(1)
    diff_id = identity_labels.unsqueeze(0) != identity_labels.unsqueeze(1)
    hard_mask = (same_mod & diff_id).float()
    if hard_mask.sum() > 0:
        sim = z @ zp.t()
        hard = (F.relu(sim - 0.30) * hard_mask).sum() / hard_mask.sum()
    else:
        hard = torch.zeros((), device=z.device)

    return base + 0.5 * hard


def train_mdie(
    model: MDIE,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    iccl_lambda: float = 0.5,
    grad_accum_steps: int = 1,
    name: str = "mdie",
    model_config: dict | None = None,
    grad_clip: float = 5.0,
    warmup_epochs: int = 1,
    resume: bool = False,
    channels_last: bool = False,
    val_auc_fn=None,
    compile_model: bool = False,
    backbone_lr_mult: float = 1.0,
    attn_lambda: float = 0.5,
) -> dict:
    """The PairedModificationDataset must yield (clean, modified, id_label, mod_label, bone_target)."""
    from ..hw import autocast_dtype
    from ..trainer_utils import (
        Throughput, is_finite_loss, load_resumable, save_resumable, set_lr,
        try_torch_compile, warmup_cosine_lr,
    )

    model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = try_torch_compile(model, enable=compile_model)
    # Discriminative LR: a pretrained backbone is lightly fine-tuned with a
    # smaller LR (backbone_lr_mult < 1), while the new RATA/heads train at full
    # LR. set_lr() honours the per-group lr_scale through the schedule.
    backbone_params, other_params = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone." in pname:
            backbone_params.append(p)
        else:
            other_params.append(p)
    param_groups = [{"params": other_params, "lr": lr, "lr_scale": 1.0}]
    if backbone_params and backbone_lr_mult > 0:
        param_groups.append({"params": backbone_params,
                             "lr": lr * backbone_lr_mult,
                             "lr_scale": backbone_lr_mult})
    elif backbone_params:
        for p in backbone_params:
            p.requires_grad_(False)
    optim = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    ac_dtype = autocast_dtype()
    use_scaler = USE_AMP and ac_dtype is torch.float16
    scaler = GradScaler("cuda", enabled=use_scaler)

    steps_per_epoch = max(len(train_loader) // max(grad_accum_steps, 1), 1)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * max(warmup_epochs, 0)

    best_ckpt = CKPT_DIR / f"{name}.pt"
    best_auc_ckpt = CKPT_DIR / f"{name}_best_auc.pt"
    last_ckpt = CKPT_DIR / f"{name}_last.pt"
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

    history = {"train_loss": [], "loss_id": [], "loss_iccl": [], "loss_mod": [],
               "val_auc": [], "epoch_time": [], "imgs_per_sec": []}
    global_step = (start_epoch - 1) * steps_per_epoch
    skipped_steps = 0

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tput = Throughput()
        model.train()
        agg = {"train_loss": 0.0, "loss_id": 0.0, "loss_iccl": 0.0, "loss_mod": 0.0, "n": 0}
        optim.zero_grad(set_to_none=True)
        for step, (clean, modded, id_lbl, mod_lbl, bone_t) in enumerate(train_loader, start=1):
            clean = clean.to(device, non_blocking=True)
            modded = modded.to(device, non_blocking=True)
            id_lbl = id_lbl.to(device, non_blocking=True)
            mod_lbl = mod_lbl.to(device, non_blocking=True)
            bone_t = bone_t.to(device, non_blocking=True)
            if channels_last:
                clean = clean.to(memory_format=torch.channels_last)
                modded = modded.to(memory_format=torch.channels_last)

            with autocast("cuda", dtype=ac_dtype, enabled=USE_AMP):
                out_clean = model(clean, id_lbl, mod_lbl, bone_target=bone_t)
                out_mod   = model(modded, id_lbl, mod_lbl, bone_target=bone_t)
                loss_id = 0.5 * (out_clean["loss_identity"] + out_mod["loss_identity"])
                loss_iccl = identity_consistency_contrastive_loss(
                    out_clean["embedding"], out_mod["embedding"],
                    id_lbl, mod_lbl)
                if "loss_mod" in out_clean:
                    loss_mod = 0.5 * (out_clean["loss_mod"] + out_mod["loss_mod"])
                else:
                    loss_mod = torch.zeros((), device=device)
                if "loss_attn" in out_clean:
                    loss_attn = 0.5 * (out_clean["loss_attn"] + out_mod["loss_attn"])
                else:
                    loss_attn = torch.zeros((), device=device)
                loss = loss_id + iccl_lambda * loss_iccl + loss_mod \
                    + attn_lambda * loss_attn
                loss = loss / grad_accum_steps

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

            bs = clean.size(0)
            agg["train_loss"] += loss.item() * grad_accum_steps * bs
            agg["loss_id"] += loss_id.item() * bs
            agg["loss_iccl"] += loss_iccl.item() * bs
            agg["loss_mod"] += loss_mod.item() * bs
            agg["n"] += bs
            tput.add(bs * 2)  # clean + modified forward passes

        n = max(agg["n"], 1)
        for k in ("train_loss", "loss_id", "loss_iccl", "loss_mod"):
            history[k].append(agg[k] / n)
        val_auc = float("nan")
        if val_auc_fn is not None:
            model.eval()
            try:
                val_auc = float(val_auc_fn())
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] val_auc_fn failed: {e}")
        history["val_auc"].append(val_auc)
        dt = time.time() - t0
        ips = tput.imgs_per_sec()
        history["epoch_time"].append(dt)
        history["imgs_per_sec"].append(ips)
        print(f"  [{name}] epoch {epoch:3d}/{epochs}  total={agg['train_loss']/n:.4f}  "
              f"id={agg['loss_id']/n:.4f}  iccl={agg['loss_iccl']/n:.4f}  "
              f"mod={agg['loss_mod']/n:.4f}  auc={val_auc:.4f}  "
              f"lr={optim.param_groups[0]['lr']:.2e}  "
              f"skip={skipped_steps}  {dt:.1f}s  {ips:.0f} img/s")

        if agg["train_loss"] / n < best_loss:
            best_loss = agg["train_loss"] / n
            save_resumable(best_ckpt, model=model, optim=optim,
                           scaler=scaler if use_scaler else None,
                           epoch=epoch, best_metric=best_loss,
                           model_config=model_config)
        if val_auc == val_auc and val_auc > best_auc:
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
        w.writerow(["epoch", "total", "loss_id", "loss_iccl", "loss_mod",
                    "val_auc", "epoch_time_s", "imgs_per_sec"])
        for i in range(len(history["train_loss"])):
            w.writerow([start_epoch + i, history["train_loss"][i], history["loss_id"][i],
                        history["loss_iccl"][i], history["loss_mod"][i],
                        history["val_auc"][i], history["epoch_time"][i],
                        history["imgs_per_sec"][i]])

    history["skipped_steps"] = skipped_steps
    history["best_auc"] = best_auc
    return history
