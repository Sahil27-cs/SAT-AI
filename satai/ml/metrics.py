"""Segmentation metrics for flood mapping.

**Accuracy is deliberately not the headline metric here.** Water typically
occupies 2–15 % of pixels in a Sen1Floods11 chip, so a model that predicts
"dry" everywhere scores 85–98 % accuracy while being completely useless. Two of
the papers in this project's Phase 0 corpus report exactly that number for
flood segmentation. IoU, Dice and per-class recall are what actually measure
the task.

**Nodata handling is the other thing that goes quietly wrong.** Sen1Floods11
labels mark unannotated pixels with a sentinel (conventionally ``-1``) rather
than a class. Counting those as negatives silently inflates every metric,
because most of them genuinely are not water. Every function here takes an
explicit validity mask and refuses to guess.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

from satai.errors import ValidationError

__all__ = [
    "SegmentationMetrics",
    "confusion_counts",
    "evaluate",
    "sweep_threshold",
    "valid_mask_from_labels",
]

IntArray = npt.NDArray[np.integer]
FloatArray = npt.NDArray[np.floating]

#: Sen1Floods11's "not annotated" sentinel. Confirm against the data before
#: relying on it -- `scripts/inspect_sen1floods11_chips.py` reports the actual
#: nodata tag, and this project does not assume dataset conventions.
IGNORE_LABEL = -1


@dataclass(frozen=True)
class SegmentationMetrics:
    """Binary segmentation scores over a defined valid region."""

    iou: float
    dice: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    specificity: float

    tp: int
    fp: int
    fn: int
    tn: int

    n_valid: int
    n_ignored: int
    positive_rate: float
    threshold: float = 0.5

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"IoU={self.iou:.4f} Dice={self.dice:.4f} P={self.precision:.4f} "
            f"R={self.recall:.4f} F1={self.f1:.4f} "
            f"(pos_rate={self.positive_rate:.3f}, n={self.n_valid:,})"
        )

    @property
    def accuracy_is_misleading(self) -> bool:
        """True when the positive class is rare enough that accuracy means little.

        At a 5 % positive rate, predicting all-negative scores 95 % accuracy.
        Surfaced so reports can carry the caveat automatically rather than
        relying on whoever writes them to remember.
        """
        return self.positive_rate < 0.20


def valid_mask_from_labels(labels: npt.ArrayLike, ignore_value: int = IGNORE_LABEL) -> IntArray:
    """Pixels that carry an actual annotation.

    Unannotated pixels must be excluded from both the loss and the metrics.
    Treating them as negatives is the single most common way a flood model's
    reported IoU ends up higher than its real performance.
    """
    array = np.asarray(labels)
    mask: IntArray = (array != ignore_value) & np.isfinite(array.astype(np.float64))
    return mask


def confusion_counts(
    pred: npt.ArrayLike,
    truth: npt.ArrayLike,
    valid: npt.ArrayLike | None = None,
) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, fn, tn)`` over the valid region only."""
    p = np.asarray(pred)
    t = np.asarray(truth)
    if p.shape != t.shape:
        raise ValidationError(
            f"prediction and truth must have the same shape; got {p.shape} and {t.shape}"
        )

    if valid is None:
        valid_arr = valid_mask_from_labels(t)
    else:
        valid_arr = np.asarray(valid).astype(bool)
        if valid_arr.shape != t.shape:
            raise ValidationError(
                f"valid mask shape {valid_arr.shape} does not match labels {t.shape}"
            )

    p_bool = p.astype(bool)[valid_arr]
    t_bool = t.astype(bool)[valid_arr]

    tp = int(np.sum(p_bool & t_bool))
    fp = int(np.sum(p_bool & ~t_bool))
    fn = int(np.sum(~p_bool & t_bool))
    tn = int(np.sum(~p_bool & ~t_bool))
    return tp, fp, fn, tn


def evaluate(
    probability: npt.ArrayLike,
    truth: npt.ArrayLike,
    *,
    threshold: float = 0.5,
    valid: npt.ArrayLike | None = None,
) -> SegmentationMetrics:
    """Score a probability map against ground truth.

    Division-by-zero cases return 0.0 rather than NaN, with one exception worth
    knowing: a chip containing no water at all yields recall 0.0 and IoU 0.0
    even for a perfect all-dry prediction. That is why aggregate metrics must
    be computed by pooling confusion counts across chips
    (:func:`aggregate`), not by averaging per-chip IoU.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValidationError(f"threshold must be in [0, 1]; got {threshold}")

    prob = np.asarray(probability, dtype=np.float64)
    pred = prob >= threshold
    tp, fp, fn, tn = confusion_counts(pred, truth, valid)

    n_valid = tp + fp + fn + tn
    if n_valid == 0:
        raise ValidationError("no valid pixels to evaluate: the validity mask is empty")

    t = np.asarray(truth)
    n_ignored = int(t.size - n_valid)

    def safe(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0 else 0.0

    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)

    return SegmentationMetrics(
        iou=safe(tp, tp + fp + fn),
        dice=safe(2 * tp, 2 * tp + fp + fn),
        precision=precision,
        recall=recall,
        f1=safe(2 * precision * recall, precision + recall),
        accuracy=safe(tp + tn, n_valid),
        specificity=safe(tn, tn + fp),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n_valid=n_valid,
        n_ignored=n_ignored,
        positive_rate=safe(tp + fn, n_valid),
        threshold=threshold,
    )


def aggregate(per_chip: list[SegmentationMetrics]) -> SegmentationMetrics:
    """Pool confusion counts across chips, rather than averaging their scores.

    Averaging per-chip IoU weights a 10-pixel chip the same as a 10,000-pixel
    one and is dominated by chips with almost no water, where IoU is unstable
    by construction. Pooling the counts gives the dataset-level quantity that
    should be reported.
    """
    if not per_chip:
        raise ValidationError("cannot aggregate an empty list of metrics")

    tp = sum(m.tp for m in per_chip)
    fp = sum(m.fp for m in per_chip)
    fn = sum(m.fn for m in per_chip)
    tn = sum(m.tn for m in per_chip)
    n_valid = tp + fp + fn + tn

    def safe(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0 else 0.0

    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)

    return SegmentationMetrics(
        iou=safe(tp, tp + fp + fn),
        dice=safe(2 * tp, 2 * tp + fp + fn),
        precision=precision,
        recall=recall,
        f1=safe(2 * precision * recall, precision + recall),
        accuracy=safe(tp + tn, n_valid),
        specificity=safe(tn, tn + fp),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n_valid=n_valid,
        n_ignored=sum(m.n_ignored for m in per_chip),
        positive_rate=safe(tp + fn, n_valid),
        threshold=per_chip[0].threshold,
    )


def sweep_threshold(
    probability: npt.ArrayLike,
    truth: npt.ArrayLike,
    *,
    valid: npt.ArrayLike | None = None,
    thresholds: npt.ArrayLike | None = None,
) -> list[SegmentationMetrics]:
    """Score across thresholds, for selecting one on the **validation** fold.

    The chosen threshold is a hyperparameter. Selecting it on the test fold
    would leak the test distribution into the reported number, so the LORO
    runner fixes it from validation and applies it unchanged at test time.
    """
    grid = (
        np.asarray(thresholds, dtype=np.float64)
        if thresholds is not None
        else np.linspace(0.05, 0.95, 19)
    )
    return [evaluate(probability, truth, threshold=float(t), valid=valid) for t in grid]


def best_threshold(sweep: list[SegmentationMetrics], metric: str = "iou") -> float:
    """Threshold maximising ``metric`` over a validation sweep."""
    if not sweep:
        raise ValidationError("cannot pick a threshold from an empty sweep")
    return max(sweep, key=lambda m: float(getattr(m, metric))).threshold
