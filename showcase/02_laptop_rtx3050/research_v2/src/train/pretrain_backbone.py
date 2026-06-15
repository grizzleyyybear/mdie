"""
Pretrain (or seed) the IR-50 backbone on a large-scale public face corpus
before MDIE fine-tuning.

On an RTX 3050 4 GB, training MS1MV3 from scratch is impractical (it wants
weeks of compute). Two operating modes:

  1. ``--use-pretrained``  (default; CPU-friendly)
        Downloads the public InsightFace IR-50 weights via
        ``src/pretrained.py`` and saves them to
        ``checkpoints/ir50_ms1m_seed.pt``. This is the path the rest of
        the pipeline expects.

  2. ``--from-scratch DATA_ROOT``
        Trains IR-50 + partial-FC ArcFace from scratch on the ImageFolder
        at DATA_ROOT (one folder per identity). Use this on an external
        GPU server. Mixed-precision, gradient checkpointing, AdamW.

The default mode is intentionally CPU-only so the rest of the pipeline can
proceed without a GPU.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from ..config import CKPT_DIR, get_device, seed_all
from ..models.backbones import IR50
from ..models.heads import ArcFaceHead
from ..pretrained import load_pretrained_ir50


SEED_CKPT = CKPT_DIR / "ir50_ms1m_seed.pt"


def _save_seed(model: IR50, source: str) -> None:
    SEED_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "source": source,
                "embedding_dim": 512}, SEED_CKPT)
    print(f"[save] {SEED_CKPT}  ({source})")


# ----------------------------------------------------------------------------
# Mode 1 — download public weights (CPU)
# ----------------------------------------------------------------------------

def use_public_pretrained() -> bool:
    model = load_pretrained_ir50(embedding_dim=512, return_maps=False)
    if model is None:
        print("[error] no mirror succeeded; cannot seed without a GPU + dataset")
        return False
    _save_seed(model, source="insightface_public_ir50")
    return True


# ----------------------------------------------------------------------------
# Mode 2 — train from scratch on a large corpus (needs GPU)
# ----------------------------------------------------------------------------

def _train_from_scratch(data_root: Path, epochs: int, batch: int,
                         lr: float, sample_id_frac: float):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    device = get_device()
    if device.type != "cuda":
        print("[warn] running from-scratch on CPU is impractical; use --use-pretrained")

    tfm = transforms.Compose([
        transforms.Resize(112), transforms.CenterCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = datasets.ImageFolder(str(data_root), transform=tfm)
    n_classes = len(ds.classes)
    print(f"[data] {len(ds)} images, {n_classes} identities")

    loader = DataLoader(ds, batch_size=batch, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    backbone = IR50(embedding_dim=512).to(device)
    head = ArcFaceHead(512, n_classes).to(device)
    opt = torch.optim.AdamW(list(backbone.parameters()) + list(head.parameters()),
                             lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    backbone.train(); head.train()
    for ep in range(epochs):
        t0 = time.time()
        for i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                emb = backbone(x)["embedding"]
                loss = head(emb, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            if i % 50 == 0:
                print(f"  ep{ep} it{i}  loss={loss.item():.3f}")
        print(f"[epoch {ep}] {time.time()-t0:.1f}s")

    _save_seed(backbone, source=f"from_scratch:{data_root.name}:ep{epochs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-pretrained", action="store_true", default=True,
                    help="Download InsightFace IR-50 weights (CPU). Default.")
    ap.add_argument("--from-scratch", type=str, default=None,
                    help="Path to ImageFolder root (e.g. MS1MV3); needs GPU.")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--id-sample", type=float, default=0.10,
                    help="Partial-FC identity sampling fraction (reserved).")
    args = ap.parse_args()
    seed_all()

    if args.from_scratch:
        root = Path(args.from_scratch)
        assert root.exists(), f"data root not found: {root}"
        _train_from_scratch(root, args.epochs, args.batch, args.lr, args.id_sample)
    else:
        ok = use_public_pretrained()
        if not ok:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
