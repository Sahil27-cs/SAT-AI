"""The join between the flood model and the risk engine.

Two halves of this project used to be unconnected: a U-Net that emits a
calibrated per-pixel water probability, and a risk engine that computes
``R = H^a * E^b * V^g``. `ml/flood/to_risk.py` connects them, and the
connection has exactly two ways to go quietly wrong. Both are pinned here.

1. **Rescaling the probability.** The engine can percentile-normalise a hazard
   field, which is right for an index and wrong for a probability: it would
   turn a chip the model thinks is entirely dry into a chip with maximal
   hazard, because the top percentile of *any* field becomes 1.0.
2. **Treating unobserved ground as safe ground.** About a third of every
   Sen1Floods11 chip is unlabelled. Zero there would read as "dry"; and NaN
   there used to fall through the banding logic into the top band, painting the
   unobserved third RED.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.flood.to_risk import (  # noqa: E402
    BLOCKED_NO_EXPOSURE,
    couple,
    flood_hazard_field,
    main,
)

from satai.risk.engine import UNOBSERVED_BAND, RiskConfig, RiskEngine  # noqa: E402


@pytest.fixture
def engine() -> RiskEngine:
    return RiskEngine(RiskConfig())


# --- the hazard field --------------------------------------------------------


def test_the_probability_is_carried_through_unchanged() -> None:
    """A model probability means something on its own scale.

    If this ever starts rescaling, a 0.2 somewhere becomes a 1.0 and the map
    says the opposite of what the model said.
    """
    probability = np.array([[0.0, 0.2], [0.5, 0.9]])
    hazard = flood_hazard_field(probability)
    assert np.allclose(hazard, probability)


def test_unobserved_pixels_become_nan_not_zero() -> None:
    probability = np.array([[0.9, 0.9], [0.9, 0.9]])
    valid = np.array([[True, True], [False, False]])

    hazard = flood_hazard_field(probability, valid)

    assert np.allclose(hazard[0], 0.9)
    assert np.isnan(hazard[1]).all()


def test_a_mismatched_valid_mask_is_refused() -> None:
    with pytest.raises(ValueError, match="does not match"):
        flood_hazard_field(np.zeros((4, 4)), np.ones((2, 2), dtype=bool))


# --- the coupling ------------------------------------------------------------


def test_a_dry_scene_stays_low_risk_everywhere(engine: RiskEngine) -> None:
    """The test that percentile-normalisation would fail.

    Every pixel is at probability 0.02. Normalised, the highest of them becomes
    1.0 and the scene acquires maximal hazard out of nothing; unnormalised, the
    risk stays near the floor where it belongs.
    """
    probability = np.full((32, 32), 0.02)
    exposure = np.full((32, 32), 0.8)

    result = couple(engine, probability, exposure)

    assert float(np.nanmax(result.risk_vector["flood"])) < 0.05


def test_zero_exposure_gives_zero_risk_even_at_certain_flooding(engine: RiskEngine) -> None:
    """The boundary condition the multiplicative form exists to enforce.

    An uninhabited floodplain is a hazard, not a risk. Up to the configured
    floor, which is what keeps a missing component from annihilating the
    product.
    """
    result = couple(engine, np.full((16, 16), 1.0), np.zeros((16, 16)))
    assert float(np.nanmax(result.risk_vector["flood"])) <= engine.config.floor


def test_risk_follows_exposure_when_hazard_is_flat(engine: RiskEngine) -> None:
    probability = np.full((8, 8), 0.7)
    exposure = np.tile(np.linspace(0.0, 1.0, 8), (8, 1))

    risk = couple(engine, probability, exposure).risk_vector["flood"]

    # Monotone across the exposure gradient, in every row.
    assert np.all(np.diff(risk, axis=1) >= -1e-12)


def test_unobserved_ground_is_not_banded_as_the_highest_risk(engine: RiskEngine) -> None:
    """The defect this module would otherwise have introduced.

    Comparisons against NaN are all False, so a masked pixel passed no band cut
    and fell through to the last band. On a SAR chip that is a third of the
    scene reported as top-decile risk.
    """
    probability = np.random.default_rng(0).uniform(0.0, 1.0, (24, 24))
    valid = np.ones((24, 24), dtype=bool)
    valid[16:] = False

    result = couple(engine, probability, np.full((24, 24), 0.5), valid=valid)
    bands = result.bands["flood"]

    assert set(np.unique(bands[16:])) == {UNOBSERVED_BAND}
    assert UNOBSERVED_BAND not in set(np.unique(bands[:16]))
    assert "RED" in set(np.unique(bands[:16]))


def test_the_unobserved_share_is_reported_rather_than_hidden(engine: RiskEngine) -> None:
    valid = np.ones((20, 20), dtype=bool)
    valid[:5] = False

    result = couple(engine, np.full((20, 20), 0.6), np.full((20, 20), 0.6), valid=valid)
    fractions = result.band_fractions("flood")

    assert fractions[UNOBSERVED_BAND] == pytest.approx(0.25)
    assert sum(fractions.values()) == pytest.approx(1.0)


def test_a_missing_vulnerability_layer_is_declared_in_the_caveats(engine: RiskEngine) -> None:
    result = couple(engine, np.full((8, 8), 0.5), np.full((8, 8), 0.5))
    assert any("ulnerability" in caveat for caveat in result.caveats)


def test_the_result_never_drops_the_official_warning_disclaimer(engine: RiskEngine) -> None:
    result = couple(engine, np.full((8, 8), 0.5), np.full((8, 8), 0.5))
    assert any("IMD" in caveat or "NDMA" in caveat for caveat in result.caveats)


# --- the CLI's refusal -------------------------------------------------------


def test_the_command_reports_blocked_without_an_exposure_raster(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No exposure layer has been ingested, so there is nothing to substitute.

    A constant exposure field would make the risk map a rescaling of the hazard
    map while presenting itself as a multi-factor result.
    """
    code = main(["--probability", str(tmp_path / "does_not_matter.tif")])

    assert code == 2
    out = capsys.readouterr().out
    assert out.startswith("BLOCKED")
    assert "exposure" in out
    assert BLOCKED_NO_EXPOSURE in out
