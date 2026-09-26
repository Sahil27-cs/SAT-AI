"""Flood-detection baselines.

The baselines are not here to win. They are here to make the deep model's
margin mean something: Otsu+HAND fixes what an unsupervised operational method
achieves on the same chips, and the per-pixel ensemble sees the same bands with
no spatial context, so the U-Net's margin over it measures what spatial context
is worth rather than what a bigger model is worth.

The tests below therefore concentrate on the two ways a baseline can quietly
lie: Otsu splitting a unimodal (dry) histogram down the middle and calling half
of it water, and a supervised model being fitted on pixels it will later be
scored on.
"""

from __future__ import annotations

import numpy as np
import pytest

from satai.errors import ValidationError
from satai.ml.baseline import (
    OtsuHandBaseline,
    PixelEnsembleBaseline,
    otsu_threshold,
    otsu_threshold_and_separability,
)


def _bimodal_vv(shape: tuple[int, int] = (64, 64), *, water_fraction: float = 0.3) -> np.ndarray:
    """VV dB with a genuine water mode near -20 and a land mode near -8."""
    rng = np.random.default_rng(0)
    land = rng.normal(-8.0, 1.5, shape)
    water = rng.normal(-20.0, 1.5, shape)
    mask = np.zeros(shape, dtype=bool)
    n_water = int(shape[0] * water_fraction)
    mask[:n_water, :] = True
    return np.where(mask, water, land), mask


# --- Otsu --------------------------------------------------------------------


def test_otsu_lands_between_the_two_modes() -> None:
    values, _ = _bimodal_vv()
    threshold = otsu_threshold(values)
    assert -20.0 < threshold < -8.0


def test_otsu_needs_at_least_two_finite_values() -> None:
    with pytest.raises(ValidationError, match="two finite"):
        otsu_threshold(np.array([np.nan, 1.0]))


def test_separability_separates_one_mode_from_two() -> None:
    """The two reference points the 0.75 default sits between.

    Splitting a single Gaussian at its mean explains exactly 2/pi of the
    variance -- about 0.64 -- because the two truncated halves sit
    sqrt(2/pi)*sigma either side of it. Land and water modes ~12 dB apart
    explain over 0.9. Without that gap there is no principled place to put
    the cut-off, so both numbers are asserted rather than left to a comment.
    """
    rng = np.random.default_rng(1)
    _, eta_unimodal = otsu_threshold_and_separability(rng.normal(-8.0, 1.5, 4096))

    values, _ = _bimodal_vv()
    _, eta_bimodal = otsu_threshold_and_separability(values)

    assert eta_unimodal == pytest.approx(2 / np.pi, abs=0.02)
    assert eta_bimodal > 0.9
    assert eta_unimodal < OtsuHandBaseline().min_separability < eta_bimodal


def test_otsu_ignores_non_finite_values() -> None:
    values, _ = _bimodal_vv()
    with_holes = values.copy()
    with_holes[0, :] = np.nan
    assert otsu_threshold(with_holes) == pytest.approx(otsu_threshold(values), abs=1.0)


# --- the unimodal guard ------------------------------------------------------


def test_a_dry_chip_is_reported_as_dry_rather_than_half_flooded() -> None:
    """The failure this guard exists for.

    Otsu always returns a threshold. On a chip with no water it returns one
    that bisects the land distribution, and a naive implementation then reports
    roughly half the chip as inundated. Over a dry season that is a flood map
    made entirely of false positives.
    """
    rng = np.random.default_rng(1)
    dry = rng.normal(-8.0, 1.5, (1, 64, 64))  # one mode only

    baseline = OtsuHandBaseline()
    prediction = baseline.predict(dry)

    assert prediction.sum() == 0
    assert baseline.separability_ is not None
    assert baseline.separability_ < baseline.min_separability


def test_a_genuinely_flooded_chip_is_detected() -> None:
    values, truth = _bimodal_vv()
    baseline = OtsuHandBaseline()
    prediction = baseline.predict(values[np.newaxis, ...])

    overlap = float((prediction.astype(bool) & truth).sum())
    assert overlap / truth.sum() > 0.9
    assert baseline.fitted_threshold_ is not None


def test_hand_mask_suppresses_water_detected_on_high_terrain() -> None:
    """Smooth dry surfaces mimic water. Height above drainage rules them out."""
    values, _ = _bimodal_vv()
    hand = np.full(values.shape, 50.0)  # every pixel far above drainage

    baseline = OtsuHandBaseline(hand_threshold_m=15.0)
    prediction = baseline.predict(values[np.newaxis, ...], hand=hand)

    assert prediction.sum() == 0


def test_predict_rejects_a_stack_that_is_not_band_first() -> None:
    with pytest.raises(ValidationError, match="band"):
        OtsuHandBaseline().predict(np.zeros((64, 64)))


def test_describe_states_the_hand_cutoff_is_an_assumption() -> None:
    described = OtsuHandBaseline().describe()
    assert described["supervised"] is False
    assert any("assumption" in c for c in described["caveats"])
    assert any("double-bounce" in c for c in described["caveats"])


# --- supervised pixel ensemble ----------------------------------------------


def _pixels(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.3).astype(int)
    vv = np.where(labels == 1, rng.normal(-20, 1.5, n), rng.normal(-8, 1.5, n))
    vh = vv - rng.normal(6.0, 1.0, n)
    elevation = rng.normal(50.0, 10.0, n)  # uninformative by construction
    return np.column_stack([vv, vh, elevation]), labels


def test_ensemble_learns_water_from_backscatter() -> None:
    features, labels = _pixels()
    valid = np.ones(labels.shape, dtype=bool)

    model = PixelEnsembleBaseline(n_estimators=40).fit(
        features, labels, valid, feature_names=["VV", "VH", "elevation"]
    )
    proba = model.predict_proba(features)

    assert ((proba >= 0.5).astype(int) == labels).mean() > 0.9


def test_top_feature_is_a_sar_band_not_terrain() -> None:
    """A model whose strongest feature is elevation is not detecting water."""
    features, labels = _pixels()
    model = PixelEnsembleBaseline(n_estimators=40).fit(
        features, labels, np.ones(labels.shape, dtype=bool), feature_names=["VV", "VH", "elevation"]
    )

    importance = model.feature_importance()
    assert next(iter(importance)) in {"VV", "VH"}
    assert importance["elevation"] < max(importance["VV"], importance["VH"])


def test_only_valid_pixels_are_used_for_fitting() -> None:
    features, labels = _pixels(2000)
    valid = np.zeros(labels.shape, dtype=bool)
    valid[:600] = True

    model = PixelEnsembleBaseline(n_estimators=20).fit(features, labels, valid)
    assert model.model_ is not None


def test_single_class_training_data_raises_instead_of_fitting() -> None:
    features, labels = _pixels(500)
    with pytest.raises(ValidationError, match="one class"):
        PixelEnsembleBaseline().fit(features, np.zeros_like(labels), np.ones(labels.shape, bool))


def test_no_valid_pixels_raises() -> None:
    features, labels = _pixels(500)
    with pytest.raises(ValidationError, match="no valid"):
        PixelEnsembleBaseline().fit(features, labels, np.zeros(labels.shape, dtype=bool))


def test_label_count_must_match_feature_rows() -> None:
    features, labels = _pixels(500)
    with pytest.raises(ValidationError, match="feature rows"):
        PixelEnsembleBaseline().fit(features, labels[:100], np.ones(100, dtype=bool))


def test_unfitted_model_refuses_to_predict() -> None:
    with pytest.raises(ValidationError, match="not fitted"):
        PixelEnsembleBaseline().predict_proba(np.zeros((10, 3)))


def test_unfitted_model_refuses_to_report_importance() -> None:
    with pytest.raises(ValidationError, match="not fitted"):
        PixelEnsembleBaseline().feature_importance()


def test_fitting_is_reproducible_under_a_fixed_seed() -> None:
    features, labels = _pixels()
    valid = np.ones(labels.shape, dtype=bool)

    a = PixelEnsembleBaseline(n_estimators=20, seed=7).fit(features, labels, valid)
    b = PixelEnsembleBaseline(n_estimators=20, seed=7).fit(features, labels, valid)

    np.testing.assert_allclose(a.predict_proba(features), b.predict_proba(features))
