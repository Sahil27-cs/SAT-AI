"""U-Net for SAR flood segmentation, self-contained.

**No `segmentation-models-pytorch`, and no ImageNet encoder.** That is a
deliberate reversal of the earlier design, for a reason that only became clear
once the data was on disk: the input here is two bands of Sentinel-1
backscatter in decibels, spanning roughly -50 to +1 dB. ImageNet weights encode
filters for 8-bit RGB photographs. Adapting them means replacing the first
convolution anyway, and what survives is a feature hierarchy tuned to natural
images and edges that SAR speckle does not produce.

Claiming "pretrained encoder" for a 2-channel dB stack would be the kind of
detail that sounds impressive in a report and does not survive a question about
it. A small U-Net trained from scratch on 446 chips is the honest architecture
for this dataset size, and it removes a dependency that would otherwise have to
be installed on every training host.

Architecture: standard encoder-decoder with skip connections. Depth and base
width are configurable because 6 GB of VRAM is a real constraint -- the
defaults fit a 512x512 chip at batch 8 on an RTX 4050.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class UNetSpec:
    """Shape of the network. Serialised into the checkpoint.

    Stored rather than inferred because a checkpoint loaded into a differently
    shaped network fails with a shape error if you are lucky and silently
    mismatches if you are not.
    """

    in_channels: int = 2
    out_channels: int = 1
    base_width: int = 32
    depth: int = 4
    dropout: float = 0.1


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU.

    BatchNorm rather than GroupNorm: batches here are 8-16 chips, large enough
    for batch statistics to be stable, and BatchNorm converges faster on a
    dataset this small.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Annotated local rather than a cast: nn.Module.__call__ is typed as
        # returning Any in some torch versions and a Tensor in others, so a
        # cast is required on one and flagged as redundant on the next. The
        # annotation keeps the declared return type meaningful under --strict
        # either way.
        out: torch.Tensor = self.block(x)
        return out


class UNet(nn.Module):
    """Encoder-decoder with skip connections.

    Returns raw logits, not probabilities. The loss applies the sigmoid
    internally for numerical stability (``BCEWithLogitsLoss``), and returning
    logits keeps a sigmoid from being applied twice -- a mistake that produces
    a model which trains fine and predicts everything at 0.5 at inference.
    """

    def __init__(self, spec: UNetSpec | None = None) -> None:
        super().__init__()
        self.spec = spec or UNetSpec()
        widths = [self.spec.base_width * (2**i) for i in range(self.spec.depth + 1)]

        self.encoders = nn.ModuleList()
        channels = self.spec.in_channels
        for width in widths[:-1]:
            self.encoders.append(DoubleConv(channels, width))
            channels = width

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(widths[-2], widths[-1], dropout=self.spec.dropout)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(self.spec.depth - 1, -1, -1):
            self.upsamples.append(nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2))
            # in_ch is doubled: the upsampled tensor is concatenated with the
            # matching encoder output, which is what makes this a U-Net rather
            # than a plain autoencoder.
            self.decoders.append(DoubleConv(widths[i] * 2, widths[i]))

        self.head = nn.Conv2d(widths[0], self.spec.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips), strict=True
        ):
            x = upsample(x)
            x = decoder(torch.cat([skip, x], dim=1))

        logits: torch.Tensor = self.head(x)
        return logits

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_unet(in_channels: int, **kwargs: object) -> UNet:
    """Construct a U-Net for a given band count."""
    return UNet(UNetSpec(in_channels=in_channels, **kwargs))  # type: ignore[arg-type]
