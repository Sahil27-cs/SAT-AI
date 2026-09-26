"""Losses for flood segmentation, all of them masked.

The masking is the whole point. Sen1Floods11 marks unannotated pixels -1, and
they are roughly a third of every chip -- 89,173 of 262,144 in the first India
chip. Training on them means training against a label nobody assigned, and
because the sentinel is -1 rather than NaN, a naive loss consumes it silently
and converges to something confidently wrong.

So every loss here takes a validity mask and excludes those pixels from both
the numerator and the denominator, rather than zeroing them -- zeroing still
lets them pull the mean down.

Dice is combined with BCE because the classes are imbalanced and imbalanced in
a direction that changes per chip. BCE alone lets a model that predicts "dry"
everywhere reach a low loss on a mostly-dry chip; Dice responds to overlap and
punishes that directly. The sum of the two is standard for segmentation and is
more stable than either alone: Dice has poor gradients when the prediction is
near-empty, which is exactly where training starts.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy over valid pixels only.

    Computed from logits rather than probabilities: ``BCEWithLogitsLoss`` fuses
    the sigmoid and the log, which avoids the overflow that a separate
    ``sigmoid`` then ``log`` hits once logits pass about +/-80.
    """
    per_pixel = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denominator = valid.sum()
    if denominator == 0:
        # Every pixel in the batch is unlabelled. Returning 0 keeps the step
        # well-defined and contributes no gradient, which is correct: there is
        # nothing to learn from this batch.
        return logits.sum() * 0.0
    return (per_pixel * valid).sum() / denominator


def masked_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft Dice loss over valid pixels only.

    ``smooth`` is added to both numerator and denominator. Without it, a chip
    with no water at all gives 0/0; with it, a correct empty prediction scores
    a perfect 1.0 rather than NaN. That case is common here -- many chips are
    entirely dry.
    """
    probabilities = torch.sigmoid(logits) * valid
    truth = targets * valid

    intersection = (probabilities * truth).sum(dim=(1, 2, 3))
    cardinality = probabilities.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return cast(torch.Tensor, 1.0 - dice.mean())


class FloodLoss(nn.Module):
    """``bce_weight * BCE + dice_weight * Dice``, both masked.

    Defaults weight them equally. The ratio is a hyperparameter and is recorded
    in the checkpoint, because a metric produced under one weighting is not
    comparable with one produced under another.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if bce_weight + dice_weight == 0:
            raise ValueError("at least one loss term must carry weight")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns ``(total, components)``; components are logged per epoch."""
        bce = masked_bce(logits, targets, valid)
        dice = masked_dice(logits, targets, valid)
        total = self.bce_weight * bce + self.dice_weight * dice
        return total, {
            "bce": float(bce.detach()),
            "dice": float(dice.detach()),
            "total": float(total.detach()),
        }
