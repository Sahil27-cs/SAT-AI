"""Multi-spectral crop damage segmentation model for SAT-AI.

Accepts dual-temporal pre- and post-flood optical/spectral features:
1. Pre-flood Red (B4)
2. Pre-flood NIR (B8)
3. Post-flood Red (B4)
4. Post-flood NIR (B8)
5. NDVI_pre  = (NIR_pre - Red_pre) / (NIR_pre + Red_pre + 1e-6)
6. NDVI_post = (NIR_post - Red_post) / (NIR_post + Red_post + 1e-6)
7. Delta_NDVI = NDVI_post - NDVI_pre

Output:
3 classes:
- 0: No damage (healthy / unaffected cropland)
- 1: Partial damage (submerged foliage, silt deposition, reduced vigour)
- 2: Full damage (complete crop washout, permanent waterlogging)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CropDamageUNet(nn.Module):
    """Dual-temporal U-Net for 3-class agricultural flood damage segmentation."""

    def __init__(self, in_channels: int = 7, num_classes: int = 3, base_features: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features, base_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 2, base_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 4, base_features * 8))

        self.up1 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_features * 4, base_features * 2)

        self.up3 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_features * 2, base_features)

        self.outc = nn.Conv2d(base_features, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        return self.outc(x)
