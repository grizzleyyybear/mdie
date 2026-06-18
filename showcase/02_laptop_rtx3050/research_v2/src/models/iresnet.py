"""
InsightFace-style IResNet50 backbone — *bit-compatible* with the public
``w600k_r50.pth`` checkpoint distributed via ``Icar/buffalo_l-torch``
(originally InsightFace's ``buffalo_l`` recognition model, trained on
WebFace12M).

This is used in the paper as a *strong external baseline*: rather than
comparing MDIE only against an ArcFace we trained ourselves on a small
dataset, we also report against the production-grade ArcFace from
InsightFace. If MDIE wins under degradation against *that*, the result
is much harder to dismiss.

The architecture is the standard ``iresnet50`` from
https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/iresnet.py
(MIT license). We re-implement only what is needed to load the state
dict and produce a 512-D embedding — no training-time bells and whistles.
"""
from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["IResNet50", "IResNet100", "load_iresnet50_w600k"]


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    """Block matching ``Icar/buffalo_l-torch/w600k_r50.pth`` layout:
    bn1 → conv1(bias) → prelu → conv2(bias) → + identity (or downsample conv with bias).
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, eps=1e-05)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=1, padding=1, bias=True)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    """Matches ``w600k_r50.pth`` layout exactly (263 keys)."""

    fc_scale = 7 * 7

    def __init__(self, block, layers, num_features: int = 512):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, 3, stride=1, padding=1, bias=True)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn1_after = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
        # Always 1×1 downsample with bias (no BN) — matches the file.
        downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                               1, stride=stride, bias=True)
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_with_maps(self, x: torch.Tensor, return_hi: bool = False):
        """Return ``(embedding, feature_map[, feature_map_hi])``.

        ``feature_map`` is the 512x7x7 tensor after ``bn1_after`` (identity-rich,
        used for the embedding). ``feature_map_hi`` is the layer3 256x14x14 map
        (spatially finer, used by MDIE's region attention for crisp per-bone
        localisation). Only returned when ``return_hi`` is True.
        """
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        hi = self.layer3(x)                      # 256 x 14 x 14
        x = self.layer4(hi)
        fmap = self.bn1_after(x)
        e = torch.flatten(fmap, 1)
        e = self.fc(e)
        e = self.features(e)
        if return_hi:
            return e, fmap, hi
        return e, fmap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb, _ = self.forward_with_maps(x)
        return emb


def _remap_w600k_keys(sd: dict) -> dict:
    """Renames the file's top-level ``bn1.*`` → ``bn1_after.*`` to avoid
    collision with the per-block ``bn1`` modules. Block-level ``bn1`` keys
    (``layer*.bn1.*``) are untouched."""
    out = {}
    for k, v in sd.items():
        if k.startswith("bn1.") or k == "bn1":
            out["bn1_after." + k[len("bn1."):]] = v
        else:
            out[k] = v
    return out


def IResNet50(num_features: int = 512) -> IResNet:
    return IResNet(IBasicBlock, [3, 4, 14, 3], num_features=num_features)


def IResNet100(num_features: int = 512) -> IResNet:
    return IResNet(IBasicBlock, [3, 13, 30, 3], num_features=num_features)


def load_iresnet50_w600k(device: torch.device | str = "cpu") -> IResNet | None:
    """
    Downloads ``Icar/buffalo_l-torch/w600k_r50.pth`` via HuggingFace Hub and
    returns an evaluation-mode IResNet50 instance. Returns ``None`` on any
    failure so callers can degrade gracefully.
    """
    from pathlib import Path
    from ..config import CKPT_DIR

    local = Path(CKPT_DIR) / "w600k_r50.pth"
    src = None
    if local.exists():
        src = local
    else:
        try:
            from huggingface_hub import hf_hub_download
            src = Path(hf_hub_download("Icar/buffalo_l-torch", "w600k_r50.pth"))
        except Exception as e:  # noqa: BLE001
            print(f"  [iresnet50] HF download failed: {e}")
            return None

    try:
        sd = torch.load(src, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = _remap_w600k_keys(sd)
        model = IResNet50()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  [iresnet50] missing={len(missing)} unexpected={len(unexpected)}")
            if missing:  print(f"    e.g. missing: {missing[:3]}")
            if unexpected: print(f"    e.g. unexpected: {unexpected[:3]}")
        model.eval().to(device)
        return model
    except Exception as e:  # noqa: BLE001
        print(f"  [iresnet50] load failed: {e}")
        return None
