"""Prediction losses: L1 + Sobel-gradient (Liu et al. 2018, eq. 5–7).

The gradient term encourages sharp edges in the predicted frame; without it,
L1/MSE-trained predictors smooth motion boundaries and lose discriminative
power on the small, fast anomalies that dominate UCSD.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0],
                         [-2.0, 0.0, 2.0],
                         [-1.0, 0.0, 1.0]])
_SOBEL_Y = _SOBEL_X.T


def _grouped_sobel(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Sobel-x and Sobel-y to every channel independently."""
    C = x.shape[1]
    kx = _SOBEL_X.to(x.device, x.dtype).view(1, 1, 3, 3).expand(C, 1, 3, 3)
    ky = _SOBEL_Y.to(x.device, x.dtype).view(1, 1, 3, 3).expand(C, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return gx, gy


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px, py = _grouped_sobel(pred)
    tx, ty = _grouped_sobel(target)
    return (px - tx).abs().mean() + (py - ty).abs().mean()


class PredictionLoss(nn.Module):
    """L1 intensity loss + lambda_grad * Sobel-gradient L1."""

    def __init__(self, lambda_grad: float = 1.0):
        super().__init__()
        self.lambda_grad = lambda_grad

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target) + self.lambda_grad * gradient_loss(pred, target)
