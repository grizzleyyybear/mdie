"""
Backbone networks used by the four SOTA baselines.

All backbones return a dict with at least ``embedding`` (B, D) and (when
relevant) ``feature_maps`` (B, C, H, W) for downstream attention.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Inception-ResNet-V1 (FaceNet original)
# =============================================================================

class _BasicConv2d(nn.Module):
    def __init__(self, c_in, c_out, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Block35(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.b0 = _BasicConv2d(256, 32, 1)
        self.b1 = nn.Sequential(_BasicConv2d(256, 32, 1), _BasicConv2d(32, 32, 3, p=1))
        self.b2 = nn.Sequential(_BasicConv2d(256, 32, 1), _BasicConv2d(32, 32, 3, p=1),
                                 _BasicConv2d(32, 32, 3, p=1))
        self.up = nn.Conv2d(96, 256, 1)

    def forward(self, x):
        out = torch.cat([self.b0(x), self.b1(x), self.b2(x)], 1)
        return F.relu(x + self.scale * self.up(out))


class _Block17(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.b0 = _BasicConv2d(896, 128, 1)
        self.b1 = nn.Sequential(_BasicConv2d(896, 128, 1),
                                 _BasicConv2d(128, 128, (1, 7), p=(0, 3)),
                                 _BasicConv2d(128, 128, (7, 1), p=(3, 0)))
        self.up = nn.Conv2d(256, 896, 1)

    def forward(self, x):
        out = torch.cat([self.b0(x), self.b1(x)], 1)
        return F.relu(x + self.scale * self.up(out))


class _Block8(nn.Module):
    def __init__(self, scale: float = 1.0, no_relu: bool = False):
        super().__init__()
        self.scale = scale
        self.no_relu = no_relu
        self.b0 = _BasicConv2d(1792, 192, 1)
        self.b1 = nn.Sequential(_BasicConv2d(1792, 192, 1),
                                 _BasicConv2d(192, 192, (1, 3), p=(0, 1)),
                                 _BasicConv2d(192, 192, (3, 1), p=(1, 0)))
        self.up = nn.Conv2d(384, 1792, 1)

    def forward(self, x):
        out = torch.cat([self.b0(x), self.b1(x)], 1)
        out = x + self.scale * self.up(out)
        return out if self.no_relu else F.relu(out)


class _ReductionA(nn.Module):
    def __init__(self):
        super().__init__()
        self.b0 = _BasicConv2d(256, 384, 3, s=2)
        self.b1 = nn.Sequential(_BasicConv2d(256, 192, 1), _BasicConv2d(192, 192, 3, p=1),
                                 _BasicConv2d(192, 256, 3, s=2))
        self.b2 = nn.MaxPool2d(3, 2)

    def forward(self, x):
        return torch.cat([self.b0(x), self.b1(x), self.b2(x)], 1)


class _ReductionB(nn.Module):
    def __init__(self):
        super().__init__()
        self.b0 = nn.Sequential(_BasicConv2d(896, 256, 1), _BasicConv2d(256, 384, 3, s=2))
        self.b1 = nn.Sequential(_BasicConv2d(896, 256, 1), _BasicConv2d(256, 256, 3, s=2))
        self.b2 = nn.Sequential(_BasicConv2d(896, 256, 1), _BasicConv2d(256, 256, 3, p=1),
                                 _BasicConv2d(256, 256, 3, s=2))
        self.b3 = nn.MaxPool2d(3, 2)

    def forward(self, x):
        return torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], 1)


class InceptionResnetV1(nn.Module):
    """Compact Inception-ResNet-V1 (FaceNet) with 512-d output."""

    def __init__(self, embedding_dim: int = 512, dropout: float = 0.4,
                 return_maps: bool = False):
        super().__init__()
        self.return_maps = return_maps
        self.stem = nn.Sequential(
            _BasicConv2d(3, 32, 3, s=2),
            _BasicConv2d(32, 32, 3),
            _BasicConv2d(32, 64, 3, p=1),
            nn.MaxPool2d(3, 2),
            _BasicConv2d(64, 80, 1),
            _BasicConv2d(80, 192, 3),
            _BasicConv2d(192, 256, 3, s=2),
        )
        self.block35 = nn.Sequential(*[_Block35(0.17) for _ in range(5)])
        self.reductionA = _ReductionA()
        self.block17 = nn.Sequential(*[_Block17(0.10) for _ in range(10)])
        self.reductionB = _ReductionB()
        self.block8 = nn.Sequential(*[_Block8(0.20) for _ in range(5)])
        self.block8_last = _Block8(no_relu=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1792, embedding_dim, bias=False)
        self.bn = nn.BatchNorm1d(embedding_dim, eps=1e-3, momentum=0.1)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.block35(x)
        x = self.reductionA(x)
        x = self.block17(x)
        x = self.reductionB(x)
        x = self.block8(x)
        feat = self.block8_last(x)
        emb = self.bn(self.linear(self.dropout(self.avgpool(feat).flatten(1))))
        emb = F.normalize(emb, dim=1)
        out = {"embedding": emb}
        if self.return_maps:
            out["feature_maps"] = feat
        return out


# =============================================================================
# 2. IR-50 (ArcFace / CosFace standard backbone)
# =============================================================================

class _SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class _IRBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        if c_in == c_out and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out))
        self.body = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.PReLU(c_out),
            nn.Conv2d(c_out, c_out, 3, stride, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.se = _SEBlock(c_out)

    def forward(self, x):
        return self.se(self.body(x)) + self.shortcut(x)


class IR50(nn.Module):
    """ArcFace IR-50 backbone (~25M params)."""

    BLOCK_PLAN = [(64, 3), (128, 4), (256, 14), (512, 3)]

    def __init__(self, embedding_dim: int = 512, dropout: float = 0.4,
                 return_maps: bool = False):
        super().__init__()
        self.return_maps = return_maps
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.PReLU(64))
        layers = []
        c_in = 64
        for c_out, n in self.BLOCK_PLAN:
            layers.append(_IRBlock(c_in, c_out, stride=2))
            for _ in range(n - 1):
                layers.append(_IRBlock(c_out, c_out, stride=1))
            c_in = c_out
        self.body = nn.Sequential(*layers)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = self.input_layer(x)
        hi = None
        for blk in self.body:
            x = blk(x)
            # capture the 256-channel 14x14 stage (group3) for finer attention
            if x.shape[1] == 256 and x.shape[-1] == 14:
                hi = x
        feat = x
        emb = F.normalize(self.output_layer(feat), dim=1)
        out = {"embedding": emb}
        if self.return_maps:
            out["feature_maps"] = feat
            if hi is not None:
                out["feature_maps_hi"] = hi
        return out


# =============================================================================
# 3. MobileFaceNet
# =============================================================================

class _ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.PReLU(c_out)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _DepthWise(nn.Module):
    def __init__(self, c_in, c_out, residual=False, k=3, s=2, p=1, groups=1):
        super().__init__()
        self.residual = residual
        self.conv1 = _ConvBlock(c_in, groups, 1, 1, 0)
        self.dw = _ConvBlock(groups, groups, k, s, p, groups=groups)
        self.proj = nn.Sequential(
            nn.Conv2d(groups, c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x):
        y = self.proj(self.dw(self.conv1(x)))
        return x + y if self.residual else y


class _Residuals(nn.Module):
    def __init__(self, c, n, groups, k=3, s=1, p=1):
        super().__init__()
        self.layers = nn.Sequential(*[
            _DepthWise(c, c, residual=True, k=k, s=s, p=p, groups=groups)
            for _ in range(n)
        ])

    def forward(self, x):
        return self.layers(x)


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_dim: int = 512, return_maps: bool = False):
        super().__init__()
        self.return_maps = return_maps
        self.conv1 = _ConvBlock(3, 64, 3, 2, 1)
        self.conv2_dw = _ConvBlock(64, 64, 3, 1, 1, groups=64)
        self.conv_23 = _DepthWise(64, 64, k=3, s=2, p=1, groups=128)
        self.conv_3 = _Residuals(64, 4, groups=128)
        self.conv_34 = _DepthWise(64, 128, k=3, s=2, p=1, groups=256)
        self.conv_4 = _Residuals(128, 6, groups=256)
        self.conv_45 = _DepthWise(128, 128, k=3, s=2, p=1, groups=512)
        self.conv_5 = _Residuals(128, 2, groups=256)
        self.conv_6_sep = _ConvBlock(128, 512, 1, 1, 0)
        self.conv_6_dw = nn.Sequential(
            nn.Conv2d(512, 512, 7, 1, 0, groups=512, bias=False),
            nn.BatchNorm2d(512),
        )
        self.linear = nn.Linear(512, embedding_dim, bias=False)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        feat = self.conv_6_sep(x)
        x = self.conv_6_dw(feat).flatten(1)
        emb = F.normalize(self.bn(self.linear(x)), dim=1)
        out = {"embedding": emb}
        if self.return_maps:
            out["feature_maps"] = feat
        return out


# =============================================================================
# Pretrained production backbone (InsightFace w600k IResNet50)
# =============================================================================

class PretrainedIResNet50(nn.Module):
    """Adapter around the production ``w600k_r50`` IResNet50 so it matches the
    backbone interface MDIE expects: ``{"embedding": (B, 512),
    "feature_maps": (B, 512, 7, 7)}``.

    The weights are trained on WebFace12M; using them as the MDIE backbone is
    what makes the region attention focus on the face instead of conv border
    artefacts. ``freeze=True`` keeps it fixed; otherwise it is lightly
    fine-tuned with a small learning rate (see ``train_mdie``).
    """

    def __init__(self, embedding_dim: int = 512, return_maps: bool = False,
                 freeze: bool = False):
        super().__init__()
        from .iresnet import load_iresnet50_w600k

        net = load_iresnet50_w600k("cpu")
        if net is None:
            raise RuntimeError(
                "could not load the pretrained w600k IResNet50 backbone; "
                "drop checkpoints/w600k_r50.pth or enable HuggingFace download "
                "(pip install huggingface_hub).")
        if embedding_dim != 512:
            raise ValueError(
                f"pretrained IResNet50 emits 512-d embeddings, got embedding_dim={embedding_dim}")
        self.net = net
        self.return_maps = return_maps
        if freeze:
            for p in self.net.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        emb, fmap, fmap_hi = self.net.forward_with_maps(x, return_hi=True)
        emb = F.normalize(emb, dim=1)
        out: Dict[str, "torch.Tensor"] = {"embedding": emb}
        if self.return_maps:
            out["feature_maps"] = fmap
            out["feature_maps_hi"] = fmap_hi
        return out


# =============================================================================
# Factory
# =============================================================================

_BACKBONES = {
    "facenet": InceptionResnetV1,
    "ir50": IR50,
    "mobilefacenet": MobileFaceNet,
    "iresnet50_w600k": PretrainedIResNet50,
}


def build_backbone(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in _BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; valid: {list(_BACKBONES)}")
    return _BACKBONES[name](**kwargs)
