"""One-at-a-time sensitivity analysis of the risk exponents.

The exponents alpha, beta and gamma default to 1.0 for want of any evidence
that says otherwise. That is an arbitrary modelling choice, and a risk map
published on top of an arbitrary choice is only defensible if the choice's
influence has been measured. This module is what turns the choice into a
reported quantity, so the tests check that it measures the right things and
reports honestly when it cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from satai.errors import ValidationError
from satai.risk.engine import RiskConfig, RiskEngine
from satai.risk.sensitivity import DEFAULT_GRID, run_sensitivity


@pytest.fixture
def engine() -> RiskEngine:
    return RiskEngine(RiskConfig())


@pytest.fixture
def layers() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    shape = (24, 24)
    return (
        {"flood": rng.random(shape)},
        rng.random(shape),
        rng.random(shape),
    )


def test_sweep_covers_every_exponent_on_every_grid_point(engine, layers) -> None:
    hazards, exposure, vulnerability = layers
    report = run_sensitivity(engine, hazards, exposure, vulnerability, hazard="flood")

    assert len(report.points) == 3 * len(DEFAULT_GRID)
    for parameter in ("alpha", "beta", "gamma"):
        values = [p.value for p in report.for_parameter(parameter)]
        assert values == list(DEFAULT_GRID)


def test_the_default_point_is_identical_to_the_baseline(engine, layers) -> None:
    """rho = 1 and no band moves, because it is the same computation."""
    hazards, exposure, vulnerability = layers
    report = run_sensitivity(engine, hazards, exposure, vulnerability, hazard="flood")

    for point in (p for p in report.points if p.is_default):
        assert point.spearman_rho == pytest.approx(1.0)
        assert point.band_reassignment_fraction == pytest.approx(0.0)
        assert point.top_decile_jaccard == pytest.approx(1.0)
        assert point.mean_risk_delta == pytest.approx(0.0)


def test_summary_statistics_exclude_the_default_point(engine, layers) -> None:
    """Otherwise the default's perfect scores would flatter the worst case."""
    hazards, exposure, vulnerability = layers
    report = run_sensitivity(engine, hazards, exposure, vulnerability, hazard="flood")

    assert report.min_rank_correlation < 1.0
    assert report.max_band_reassignment > 0.0
    assert report.min_top_decile_jaccard < 1.0


def test_gamma_is_skipped_and_not_faked_when_vulnerability_is_absent(engine, layers) -> None:
    """A gamma row with no vulnerability layer would be a fabricated result."""
    hazards, exposure, _ = layers
    report = run_sensitivity(engine, hazards, exposure, None, hazard="flood")

    assert report.for_parameter("gamma") == []
    assert len(report.points) == 2 * len(DEFAULT_GRID)


def test_a_grid_without_the_default_is_rejected(engine, layers) -> None:
    """Every point is measured against 1.0; without it there is no reference."""
    hazards, exposure, vulnerability = layers
    with pytest.raises(ValidationError, match=r"1\.0"):
        run_sensitivity(engine, hazards, exposure, vulnerability, hazard="flood", grid=(0.5, 2.0))


def test_unknown_hazard_is_rejected(engine, layers) -> None:
    hazards, exposure, vulnerability = layers
    with pytest.raises(ValidationError, match="not in"):
        run_sensitivity(engine, hazards, exposure, vulnerability, hazard="wildfire")


# --- what the numbers distinguish --------------------------------------------


def test_rank_order_is_far_more_stable_than_the_map(engine, layers) -> None:
    """The finding the analysis exists to surface.

    Exponents barely reorder which places are riskiest, but they do move a
    substantial share of cells between colour bands. So the ranking may be
    reported with confidence and the *map* may not -- a distinction that
    disappears if only one of the two numbers is computed.
    """
    hazards, exposure, vulnerability = layers
    report = run_sensitivity(engine, hazards, exposure, vulnerability, hazard="flood")

    assert report.min_rank_correlation > 0.9
    assert report.max_band_reassignment > report.min_rank_correlation - 0.9


def test_a_degenerate_field_shows_no_sensitivity(engine) -> None:
    """A constant hazard field cannot be reordered, whatever the exponents."""
    constant = {"flood": np.full((16, 16), 0.5)}
    exposure = np.full((16, 16), 0.5)

    report = run_sensitivity(engine, constant, exposure, None, hazard="flood")

    assert report.min_rank_correlation == pytest.approx(1.0)
    assert report.max_band_reassignment == pytest.approx(0.0)


# --- reporting ---------------------------------------------------------------


def test_interpretation_names_the_limits_of_a_one_at_a_time_design(engine, layers) -> None:
    hazards, exposure, vulnerability = layers
    report = run_sensitivity(engine, hazards, exposure, vulnerability, hazard="flood")

    assert any("interaction" in note for note in report.notes)
    assert report.interpretation()


def test_report_serialises_with_its_baseline_config_hash(engine, layers, tmp_path) -> None:
    hazards, exposure, vulnerability = layers
    report = run_sensitivity(engine, hazards, exposure, vulnerability, hazard="flood")

    payload = report.to_dict()
    assert payload["default_config_hash"] == engine.config.config_hash()
    assert payload["hazard"] == "flood"
    assert len(payload["points"]) == len(report.points)

    assert report.save(tmp_path / "sensitivity.json").exists()


def test_report_records_the_number_of_cells_it_measured(engine, layers) -> None:
    hazards, exposure, vulnerability = layers
    report = run_sensitivity(engine, hazards, exposure, vulnerability, hazard="flood")
    assert report.n_cells == 24 * 24
