"""Segmentation metrics.

The tests that matter here are not the arithmetic ones. They are the two
statements the module exists to enforce: that ignored pixels never enter a
score, and that dataset-level IoU is obtained by pooling confusion counts
rather than by averaging per-chip IoU. Both are easy to get wrong silently and
both would inflate a reported number.
"""

from __future__ import annotations

import numpy as np
import pytest

from satai.errors import ValidationError
from satai.ml.metrics import (
    IGNORE_LABEL,
    SegmentationMetrics,
    aggregate,
    best_threshold,
    confusion_counts,
    evaluate,
    sweep_threshold,
    valid_mask_from_labels,
)


def test_perfect_prediction_scores_one() -> None:
    truth = np.array([[1, 1], [0, 0]])
    metrics = evaluate(truth.astype(float), truth)
    assert metrics.iou == pytest.approx(1.0)
    assert metrics.dice == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert (metrics.fp, metrics.fn) == (0, 0)


def test_confusion_counts_match_definition() -> None:
    pred = np.array([1, 1, 0, 0])
    truth = np.array([1, 0, 1, 0])
    assert confusion_counts(pred, truth) == (1, 1, 1, 1)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValidationError, match="same shape"):
        confusion_counts(np.zeros((2, 2)), np.zeros((3, 3)))


def test_threshold_outside_unit_interval_is_rejected() -> None:
    with pytest.raises(ValidationError, match="threshold"):
        evaluate(np.array([0.5]), np.array([1]), threshold=1.5)


# --- the ignore label --------------------------------------------------------


def test_ignored_pixels_are_excluded_from_every_count() -> None:
    """A wrong prediction over an ignored pixel must not cost anything.

    Sen1Floods11 marks no-data as -1. If those pixels were scored as negatives
    the reported IoU would depend on how much of the chip was masked, which is
    a property of the scene rather than of the model.
    """
    truth = np.array([1, 0, IGNORE_LABEL, IGNORE_LABEL])
    pred = np.array([1, 0, 1, 1])  # both errors fall on ignored pixels

    metrics = evaluate(pred.astype(float), truth)
    assert metrics.n_valid == 2
    assert metrics.n_ignored == 2
    assert metrics.fp == 0
    assert metrics.iou == pytest.approx(1.0)


def test_valid_mask_from_labels_marks_only_the_ignore_value() -> None:
    labels = np.array([0, 1, IGNORE_LABEL])
    assert valid_mask_from_labels(labels).tolist() == [1, 1, 0]


def test_explicit_valid_mask_overrides_the_label_derived_one() -> None:
    truth = np.array([1, 1, 1, 1])
    pred = np.array([1, 1, 0, 0])
    metrics = evaluate(pred.astype(float), truth, valid=np.array([1, 1, 0, 0]))
    assert metrics.n_valid == 2
    assert metrics.recall == pytest.approx(1.0)


def test_empty_valid_region_raises_rather_than_returning_zero() -> None:
    """Zero would read as 'the model scored 0'. It did not; it was not scored."""
    truth = np.array([IGNORE_LABEL, IGNORE_LABEL])
    with pytest.raises(ValidationError, match="valid"):
        evaluate(np.array([1.0, 1.0]), truth)


# --- pooling, not averaging --------------------------------------------------


def _chip(tp: int, fp: int, fn: int, tn: int) -> SegmentationMetrics:
    pred = np.concatenate([np.ones(tp + fp), np.zeros(fn + tn)])
    truth = np.concatenate([np.ones(tp), np.zeros(fp), np.ones(fn), np.zeros(tn)])
    return evaluate(pred, truth)


def test_aggregate_pools_counts_instead_of_averaging_scores() -> None:
    """The failure this guards against, with numbers.

    A large chip scores IoU 0.5; a tiny chip with two pixels scores 1.0. Their
    mean is 0.75, which says the model is good. Pooled over 4,002 pixels the
    answer is ~0.50, which is the truth. Reporting the mean would overstate
    the model by half its error.
    """
    big = _chip(tp=1000, fp=1000, fn=0, tn=2000)
    tiny = _chip(tp=2, fp=0, fn=0, tn=0)

    assert big.iou == pytest.approx(0.5)
    assert tiny.iou == pytest.approx(1.0)

    pooled = aggregate([big, tiny])
    naive_mean = (big.iou + tiny.iou) / 2

    assert pooled.iou == pytest.approx(1002 / (1002 + 1000))
    assert pooled.iou < naive_mean - 0.2


def test_aggregate_preserves_total_counts() -> None:
    chips = [_chip(3, 1, 2, 4), _chip(5, 0, 1, 10)]
    pooled = aggregate(chips)
    assert pooled.tp == 8
    assert pooled.n_valid == sum(c.n_valid for c in chips)


def test_aggregate_of_empty_list_raises() -> None:
    with pytest.raises(ValidationError, match="empty"):
        aggregate([])


# --- accuracy caveat ---------------------------------------------------------


def test_accuracy_is_flagged_misleading_when_water_is_rare() -> None:
    truth = np.zeros(100, dtype=int)
    truth[:5] = 1  # 5 % positive
    metrics = evaluate(np.zeros(100), truth)

    assert metrics.accuracy == pytest.approx(0.95)
    assert metrics.iou == pytest.approx(0.0)
    assert metrics.accuracy_is_misleading


def test_accuracy_is_not_flagged_on_a_balanced_chip() -> None:
    truth = np.array([1] * 50 + [0] * 50)
    metrics = evaluate(truth.astype(float), truth)
    assert not metrics.accuracy_is_misleading


# --- threshold selection -----------------------------------------------------


def test_sweep_returns_one_score_per_threshold_and_records_it() -> None:
    prob = np.array([0.1, 0.4, 0.6, 0.9])
    truth = np.array([0, 0, 1, 1])
    sweep = sweep_threshold(prob, truth, thresholds=[0.2, 0.5, 0.8])

    assert [m.threshold for m in sweep] == [0.2, 0.5, 0.8]


def test_best_threshold_picks_the_maximising_entry() -> None:
    prob = np.array([0.1, 0.4, 0.6, 0.9])
    truth = np.array([0, 0, 1, 1])
    sweep = sweep_threshold(prob, truth, thresholds=[0.2, 0.5, 0.95])

    assert best_threshold(sweep) == 0.5


def test_best_threshold_of_empty_sweep_raises() -> None:
    with pytest.raises(ValidationError, match="empty"):
        best_threshold([])
