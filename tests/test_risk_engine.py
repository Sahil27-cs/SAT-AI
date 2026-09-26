"""Multi-hazard risk engine (ADR-008).

Three claims are worth testing here, and they are the three that separate this
engine from the dashboards it is a reaction to:

1. Zero exposure implies zero risk. An uninhabited floodplain has high hazard
   and no risk. An additive form gets this wrong, which is why the formulation
   is multiplicative.
2. Per-hazard scores are never averaged into one number. They come from
   different models with different base rates and are not commensurable.
3. A missing vulnerability layer is excluded and declared, never imputed.
   Imputing 0 zeroes every cell; imputing 1 silently asserts the worst case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from satai.errors import ValidationError
from satai.provenance import SourceKind
from satai.risk.engine import RiskConfig, RiskEngine, normalise_percentile

CONFIG = Path("configs/risk.yaml")


@pytest.fixture
def engine() -> RiskEngine:
    return RiskEngine(RiskConfig())


@pytest.fixture
def field_() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((32, 32))


# --- normalisation -----------------------------------------------------------


def test_normalisation_maps_to_the_unit_interval() -> None:
    values = np.linspace(-50.0, 120.0, 500)
    scaled = normalise_percentile(values)
    assert scaled.min() == pytest.approx(0.0)
    assert scaled.max() == pytest.approx(1.0)


def test_one_extreme_cell_cannot_compress_the_whole_scale() -> None:
    """Why the bounds are percentiles rather than min and max.

    Under a multiplicative formulation a compressed scale does not merely look
    wrong, it drags every other cell's risk toward the floor.
    """
    ordinary = np.linspace(0.0, 1.0, 999)
    with_outlier = np.append(ordinary, 1e6)

    scaled = normalise_percentile(with_outlier)
    assert scaled[:999].mean() == pytest.approx(0.5, abs=0.05)


def test_a_constant_field_normalises_to_mid_scale() -> None:
    scaled = normalise_percentile(np.full(100, 7.0))
    assert np.allclose(scaled, 0.5)


def test_normalising_an_all_nan_field_raises() -> None:
    with pytest.raises(ValidationError, match="finite"):
        normalise_percentile(np.full(10, np.nan))


# --- the boundary condition that justifies the formulation -------------------


def test_zero_exposure_drives_risk_to_the_floor(engine: RiskEngine) -> None:
    """The defining property. High hazard over empty land is not high risk."""
    hazard = np.ones((8, 8))
    empty = np.zeros((8, 8))

    risk = engine.compute_hazard_risk(hazard, empty, None)

    assert risk.max() <= engine.config.floor + 1e-9


def test_an_additive_form_would_get_that_wrong(engine: RiskEngine) -> None:
    """Stated as a test so the design choice is checkable, not just asserted."""
    hazard, empty = np.ones((4, 4)), np.zeros((4, 4))

    multiplicative = engine.compute_hazard_risk(hazard, empty, None).max()
    naive_additive = float((hazard + empty).max() / 2)

    assert multiplicative < 0.02
    assert naive_additive == pytest.approx(0.5)


def test_risk_rises_monotonically_with_exposure(engine: RiskEngine) -> None:
    hazard = np.full((4, 4), 0.8)
    low = engine.compute_hazard_risk(hazard, np.full((4, 4), 0.2), None)
    high = engine.compute_hazard_risk(hazard, np.full((4, 4), 0.9), None)
    assert high.mean() > low.mean()


def test_component_shapes_must_agree(engine: RiskEngine) -> None:
    with pytest.raises(ValidationError, match="does not match"):
        engine.compute_hazard_risk(np.ones((4, 4)), np.ones((5, 5)), None)


def test_exponents_change_the_weighting_not_the_direction() -> None:
    hazard, exposure = np.full((4, 4), 0.5), np.full((4, 4), 0.9)

    flat = RiskEngine(RiskConfig()).compute_hazard_risk(hazard, exposure, None)
    hazard_led = RiskEngine(RiskConfig(alpha=2.0)).compute_hazard_risk(hazard, exposure, None)

    assert hazard_led.mean() < flat.mean()  # H < 1, so a larger alpha shrinks it


# --- missing vulnerability ---------------------------------------------------


def test_missing_vulnerability_is_excluded_and_declared(
    engine: RiskEngine, field_: np.ndarray
) -> None:
    result = engine.compute({"flood": field_}, field_, vulnerability=None)

    assert np.isfinite(result.risk_vector["flood"]).all()
    assert any("Vulnerability layer unavailable" in c for c in result.caveats)
    assert any("upper bound" in c for c in result.caveats)


def test_present_vulnerability_carries_its_proxy_caveat(
    engine: RiskEngine, field_: np.ndarray
) -> None:
    result = engine.compute({"flood": field_}, field_, vulnerability=field_)

    assert any("PROXY" in c for c in result.caveats)
    assert any("marginalised" in c for c in result.caveats)


def test_vulnerability_shape_must_match(engine: RiskEngine) -> None:
    with pytest.raises(ValidationError, match="vulnerability"):
        engine.compute_hazard_risk(np.ones((4, 4)), np.ones((4, 4)), np.ones((6, 6)))


# --- multi-hazard ------------------------------------------------------------


def test_hazards_are_kept_as_a_vector_never_averaged(engine: RiskEngine) -> None:
    rng = np.random.default_rng(1)
    hazards = {"flood": rng.random((16, 16)), "wildfire": rng.random((16, 16))}

    result = engine.compute(hazards, rng.random((16, 16)))

    assert set(result.risk_vector) == {"flood", "wildfire"}
    assert not hasattr(result, "overall_risk")
    assert not hasattr(result, "combined_risk")


def test_dominant_hazard_is_the_argmax_cell_by_cell(engine: RiskEngine) -> None:
    flood = np.full((4, 4), 0.9)
    fire = np.full((4, 4), 0.1)
    flood[0, 0], fire[0, 0] = 0.1, 0.9

    result = engine.compute(
        {"flood": flood, "wildfire": fire},
        np.ones((4, 4)),
        normalise=False,
    )

    assert result.dominant_hazard[1, 1] == "flood"
    assert result.dominant_hazard[0, 0] == "wildfire"
    assert (result.dominant_margin >= 0).all()


def test_no_hazards_raises(engine: RiskEngine) -> None:
    with pytest.raises(ValidationError, match="no hazards"):
        engine.compute({}, np.ones((4, 4)))


# --- bands -------------------------------------------------------------------


def test_bands_are_relative_percentiles_of_this_area(engine: RiskEngine) -> None:
    risk = np.linspace(0.0, 1.0, 1000)
    bands = engine.assign_bands(risk)

    assert bands[0] == "GREEN"
    assert bands[-1] == "RED"
    assert set(np.unique(bands)) == {"GREEN", "YELLOW", "ORANGE", "RED"}


def test_band_shares_follow_the_configured_percentiles(engine: RiskEngine) -> None:
    bands = engine.assign_bands(np.linspace(0.0, 1.0, 10_000))
    share = {name: float((bands == name).mean()) for name in np.unique(bands)}

    assert share["GREEN"] == pytest.approx(0.50, abs=0.01)
    assert share["YELLOW"] == pytest.approx(0.25, abs=0.01)
    assert share["ORANGE"] == pytest.approx(0.15, abs=0.01)
    assert share["RED"] == pytest.approx(0.10, abs=0.01)


def test_a_uniformly_low_field_still_produces_red_cells(engine: RiskEngine) -> None:
    """Relative banding, stated as a test because it is a real limitation.

    Every AOI gets a top decile, even a safe one. That is why the caveat about
    percentiles is attached to every result rather than left to the reader.
    """
    bands = engine.assign_bands(np.linspace(0.001, 0.003, 1000))
    assert "RED" in set(np.unique(bands))


def test_banding_an_all_nan_field_raises(engine: RiskEngine) -> None:
    with pytest.raises(ValidationError, match="finite"):
        engine.assign_bands(np.full(10, np.nan))


# --- configuration and provenance --------------------------------------------


@pytest.mark.skipif(not CONFIG.is_file(), reason="configs/risk.yaml not present")
def test_shipped_config_loads_and_defaults_to_unit_exponents() -> None:
    config = RiskConfig.from_yaml(CONFIG)

    assert config.method == "multiplicative"
    assert (config.alpha, config.beta, config.gamma) == (1.0, 1.0, 1.0)
    assert config.forbid_naive_average
    assert "NOT an official warning" in config.disclaimer


def test_missing_config_file_raises() -> None:
    with pytest.raises(ValidationError, match="not found"):
        RiskConfig.from_yaml(Path("configs/does-not-exist.yaml"))


def test_config_hash_tracks_the_parameters_that_change_the_output() -> None:
    base = RiskConfig()
    assert base.config_hash() == RiskConfig().config_hash()
    assert base.config_hash() != base.with_exponents(1.5, 1.0, 1.0).config_hash()


def test_config_hash_ignores_the_disclaimer_text() -> None:
    """Editing prose must not invalidate a cached numerical result."""
    base = RiskConfig()
    reworded = RiskConfig(disclaimer="Different words, same maths.")
    assert base.config_hash() == reworded.config_hash()


def test_envelope_is_an_index_not_a_model_output(engine: RiskEngine, field_: np.ndarray) -> None:
    """The distinction the whole provenance layer exists to preserve.

    A risk score is a documented composite of configurable exponents. Labelling
    it MODEL would let the interface describe it as a prediction.
    """
    result = engine.compute({"flood": field_}, field_, field_)
    env = engine.to_envelope(result, "flood")

    assert env.source.kind is SourceKind.INDEX
    assert env.source.id == "risk_engine"
    assert env.source.version == result.config_hash


def test_envelope_exposes_summary_values_for_grounding(
    engine: RiskEngine, field_: np.ndarray
) -> None:
    result = engine.compute({"flood": field_}, field_, field_)
    env = engine.to_envelope(result, "flood")

    assert "mean_risk" in env.value
    assert 0.0 <= env.value["mean_risk"] <= 1.0
    assert env.groundable_values()
