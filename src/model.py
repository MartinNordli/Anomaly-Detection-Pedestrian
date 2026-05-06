"""UNet predictor for video anomaly detection (Liu et al. 2018-style).

A standard 4-down/4-up UNet with skip connections, GroupNorm and SiLU.
Bilinear upsampling followed by 1x1 conv is used in the decoder to avoid
checkerboard artifacts that ConvTranspose2d sometimes produces; this matters
because the per-pixel anomaly score IS the prediction error, and structured
artifacts create false positives.

Designed for ~1.5M parameters at base=64 (small enough to train on M4 MPS in
reasonable time, large enough to hit AUC > 0.85 on raw UCSD).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    g = min(max_groups, num_channels)
    while num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _gn(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _gn(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, in_ch // 2, 1)
        self.conv = _DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetPredictor(nn.Module):
    """4-level UNet, ~1.5M params at base=64.

    Args:
        in_channels: total channels in the stacked input
            (window * channels_per_frame).
        out_channels: number of channels in target frame (1 for raw/S/etc.).
        base: base channel width; doubled at each downsample.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 1, base: int = 64):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 8

        self.inc = _DoubleConv(in_channels, c1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(c1, c2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(c2, c3))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(c3, c4))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(c4, c5))

        self.up1 = _Up(c5, c4, c4)
        self.up2 = _Up(c4, c3, c3)
        self.up3 = _Up(c3, c2, c2)
        self.up4 = _Up(c2, c1, c1)

        self.out = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        y = self.out(u4)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
