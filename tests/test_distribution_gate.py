"""Track A / Track B distribution gate (ADR-010).

The gate exists to answer one question before Phase 5: does a model trained on
Sen1Floods11's pre-processed chips face the same backscatter distribution when
it is shown SAT-AI's own GRD pipeline output? If it does not, a strong test IoU
on Track A says nothing about Track B, and the honest move is to say so rather
than to deploy the model over India and report the Track A number.

So the properties under test are: identical distributions pass, a calibration
offset of the size that would move a decision boundary fails, and the verdict
of the whole report is the worst of its bands rather than their average.
"""

from __future__ import annotations

import numpy as np
import pytest

from satai.errors import DataQualityError, ValidationError
from satai.ml.distribution_gate import (
    FAIL_WASSERSTEIN_DB,
    WARN_WASSERSTEIN_DB,
    compare_band,
    run_gate,
)


def _vv(n: int = 5000, *, offset: float = 0.0, scale: float = 1.0, seed: int = 1) -> np.ndarray:
    """Plausible VV backscatter in dB: land around -8, water far below."""
    rng = np.random.default_rng(seed)
    return rng.normal(-8.0 + offset, 2.5 * scale, n)


def test_identical_distributions_pass() -> None:
    values = _vv()
    result = compare_band(values, values.copy(), band="VV")

    assert result.verdict == "pass"
    assert result.passed
    assert result.wasserstein == pytest.approx(0.0, abs=1e-9)
    assert result.ks_statistic == pytest.approx(0.0, abs=1e-9)


def test_same_distribution_different_draws_passes() -> None:
    """Sampling noise alone must not trip the gate."""
    result = compare_band(_vv(seed=1), _vv(seed=2), band="VV")
    assert result.verdict == "pass"


def test_large_calibration_offset_fails() -> None:
    """A 5 dB shift is half a typical land/water separation. It must fail."""
    result = compare_band(_vv(), _vv(offset=5.0, seed=2), band="VV")

    assert result.verdict == "fail"
    assert not result.passed
    assert result.wasserstein >= FAIL_WASSERSTEIN_DB
    assert result.mean_shift == pytest.approx(5.0, abs=0.3)


def test_moderate_offset_warns_rather_than_failing() -> None:
    result = compare_band(_vv(), _vv(offset=1.2, seed=2), band="VV")

    assert result.verdict == "warn"
    assert not result.passed  # `passed` is strict; a warning is not a clean pass
    assert WARN_WASSERSTEIN_DB <= result.wasserstein < FAIL_WASSERSTEIN_DB


def test_a_warning_annotates_but_does_not_block_phase_5() -> None:
    """Band-level `passed` is strict; the gate's decision is `may_proceed`."""
    report = run_gate(
        {"VV": _vv()},
        {"VV": _vv(offset=1.2, seed=2)},
        track_a_source="sen1floods11-chips",
        track_b_source="satai-grd-pipeline",
    )

    assert report.verdict == "warn"
    assert report.may_proceed


def test_variance_mismatch_is_reported_as_a_speckle_filter_difference() -> None:
    """Same mean, very different spread -- the classic speckle-filter mismatch."""
    result = compare_band(_vv(), _vv(scale=3.0, seed=2), band="VV")

    assert result.verdict in {"warn", "fail"}
    assert result.std_ratio > 2.0
    assert any("speckle" in note for note in result.notes)


def test_ks_pvalue_is_reported_but_explicitly_not_the_gate() -> None:
    """With 10^5 pixels every difference is 'significant'; that is not evidence."""
    result = compare_band(_vv(), _vv(seed=2), band="VV")

    assert 0.0 <= result.ks_pvalue <= 1.0
    assert any("p-value" in note for note in result.notes)


def test_empty_track_raises_rather_than_passing_vacuously() -> None:
    with pytest.raises(DataQualityError):
        compare_band(np.array([]), _vv(), band="VV")


def test_non_finite_values_are_dropped_before_comparison() -> None:
    a = _vv(1000)
    b = np.concatenate([a.copy(), [np.nan, np.inf, -np.inf]])
    result = compare_band(a, b, band="VV")

    assert result.n_b == 1000
    assert result.verdict == "pass"


# --- whole-report behaviour --------------------------------------------------


def test_report_verdict_is_the_worst_band_not_the_average() -> None:
    """One broken band is enough. Averaging verdicts would hide it."""
    report = run_gate(
        {"VV": _vv(), "VH": _vv(seed=3)},
        {"VV": _vv(seed=2), "VH": _vv(offset=6.0, seed=4)},
        track_a_source="sen1floods11-chips",
        track_b_source="satai-grd-pipeline",
    )

    assert report.verdict == "fail"
    assert not report.may_proceed
    assert {c.band for c in report.bands} == {"VV", "VH"}


def test_matching_tracks_may_proceed() -> None:
    report = run_gate(
        {"VV": _vv(), "VH": _vv(seed=3)},
        {"VV": _vv(seed=2), "VH": _vv(seed=4)},
        track_a_source="sen1floods11-chips",
        track_b_source="satai-grd-pipeline",
    )

    assert report.verdict == "pass"
    assert report.may_proceed
    assert "satai-grd-pipeline" in report.report()


def test_disjoint_band_sets_raise() -> None:
    with pytest.raises(ValidationError, match="no bands in common"):
        run_gate(
            {"VV": _vv()},
            {"HH": _vv()},
            track_a_source="a",
            track_b_source="b",
        )


def test_report_serialises_to_a_dict_that_records_both_sources(tmp_path) -> None:
    report = run_gate(
        {"VV": _vv()},
        {"VV": _vv(seed=2)},
        track_a_source="sen1floods11-chips",
        track_b_source="satai-grd-pipeline",
    )
    payload = report.to_dict()

    assert payload["track_a_source"] == "sen1floods11-chips"
    assert payload["track_b_source"] == "satai-grd-pipeline"

    written = report.save(tmp_path / "gate.json")
    assert written.exists()
