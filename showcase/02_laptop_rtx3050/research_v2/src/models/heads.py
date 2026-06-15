"""Classification heads and compact training-only loss utilities."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambda_ * grad_out, None


class GradientReversal(nn.Module):
    """Identity in the forward pass, negative scaled gradient in backward."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return _GradReverseFn.apply(x, self.lambda_)


def focal_arcface(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal weighting over angular-margin logits for hard-example emphasis."""
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    ce = F.nll_loss(log_p, labels, reduction="none")
    pt = p.gather(1, labels.unsqueeze(1)).squeeze(1)
    return ((1 - pt) ** gamma * ce).mean()


class SoftmaxHead(nn.Module):
    def __init__(self, embedding_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, n_classes)

    def forward(self, emb, labels):
        return F.cross_entropy(self.fc(emb), labels)


class ArcFaceHead(nn.Module):
    """Original ArcFace (additive angular margin) head."""

    def __init__(self, embedding_dim: int, n_classes: int,
                 margin: float = 0.50, scale: float = 64.0):
        super().__init__()
        self.s, self.m = scale, margin
        self.W = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.W)
        self.cos_m = math.cos(margin); self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, emb, labels):
        W = F.normalize(self.W, dim=1)
        cos = F.linear(F.normalize(emb, dim=1), W).clamp(-1 + 1e-7, 1 - 1e-7)
        sin = (1 - cos ** 2).clamp_min(0).sqrt()
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        oh = F.one_hot(labels, self.W.shape[0]).float()
        logits = (oh * phi + (1 - oh) * cos) * self.s
        return F.cross_entropy(logits, labels)


class CosFaceHead(nn.Module):
    """CosFace (large-margin cosine) head."""

    def __init__(self, embedding_dim: int, n_classes: int,
                 margin: float = 0.35, scale: float = 64.0):
        super().__init__()
        self.s, self.m = scale, margin
        self.W = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, emb, labels):
        W = F.normalize(self.W, dim=1)
        cos = F.linear(F.normalize(emb, dim=1), W).clamp(-1 + 1e-7, 1 - 1e-7)
        oh = F.one_hot(labels, self.W.shape[0]).float()
        logits = (cos - oh * self.m) * self.s
        return F.cross_entropy(logits, labels)


class TripletHead(nn.Module):
    """
    Online semi-hard triplet loss (FaceNet-style).
    Operates directly on the L2-normalized embedding batch.
    """

    def __init__(self, margin: float = 0.30):
        super().__init__()
        self.margin = margin

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # cosine distance on normalized embeddings = 1 - cos_sim
        sim = emb @ emb.t()
        dist = 1.0 - sim                                # (B, B)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
        pos_mask = same & ~eye
        neg_mask = ~same

        # hardest positive per anchor
        pos_dist = (dist * pos_mask.float()).max(dim=1).values
        # semi-hard negative: smallest dist that is still > pos_dist (else hardest)
        big = dist.clone()
        big[~neg_mask] = float("inf")
        # add anchors with no positives mask out later
        loss_per = F.relu(pos_dist + self.margin - big.min(dim=1).values)
        valid = pos_mask.any(dim=1).float()
        return (loss_per * valid).sum() / valid.sum().clamp_min(1.0)
