"""Auxiliary losses + utility layers used by the novel method."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient reversal layer for adversarial modification disentanglement
# ---------------------------------------------------------------------------

class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambda_ * grad_out, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return _GradReverseFn.apply(x, self.lambda_)


# ---------------------------------------------------------------------------
# Optional focal-flavored ArcFace (for hard-example emphasis in long tails)
# ---------------------------------------------------------------------------

def focal_arcface(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    ce = F.nll_loss(log_p, labels, reduction="none")
    pt = p.gather(1, labels.unsqueeze(1)).squeeze(1)
    return ((1 - pt) ** gamma * ce).mean()
