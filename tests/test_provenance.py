"""Tests for the provenance contract.

These are the most important tests in Phase 1. The provenance envelope is what
structurally prevents the agent plane from stating a number no model produced,
so its behaviour needs to be pinned down before anything is built on top of it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError as PydanticValidationError

from satai.provenance import (
    DataSourceRef,
    HazardType,
    ProvenanceEnvelope,
    SourceKind,
    SpatialRef,
    TemporalValidity,
    envelope,
)


def test_envelope_carries_value_and_source() -> None:
    env = envelope(
        0.87,
        quantity="flood_extent_confidence",
        unit="probability",
        kind=SourceKind.MODEL,
        source_id="flood_unet",
        version="0.3.1",
    )
    assert env.value == 0.87
    assert env.source.id == "flood_unet"
    assert env.source.version == "0.3.1"
    assert env.is_model_output
    assert not env.is_observation


def test_observation_and_model_are_distinguishable() -> None:
    """A FIRMS detection is an observation; a danger score is a model output.

    Collapsing the distinction would let the interface describe a measurement
    as a prediction, which is exactly the kind of overclaim this project is
    built to avoid.
    """
    fire = envelope(
        12,
        quantity="active_fire_count",
        unit="count",
        kind=SourceKind.OBSERVATION,
        source_id="firms_viirs_snpp",
        hazard=HazardType.WILDFIRE,
    )
    danger = envelope(
        0.64,
        quantity="fire_danger",
        unit="probability",
        kind=SourceKind.MODEL,
        source_id="fire_danger_xgb",
        hazard=HazardType.WILDFIRE,
    )
    assert fire.is_observation and not fire.is_model_output
    assert danger.is_model_output and not danger.is_observation


def test_envelope_is_immutable() -> None:
    env = envelope(1.0, quantity="x", kind=SourceKind.DERIVED, source_id="d")
    with pytest.raises(PydanticValidationError):
        env.value = 2.0  # type: ignore[misc]


def test_with_caveat_returns_a_copy() -> None:
    env = envelope(0.5, quantity="x", kind=SourceKind.MODEL, source_id="m")
    updated = env.with_caveat("optical modality unavailable (cloud cover 78%)")
    assert env.caveats == []
    assert len(updated.caveats) == 1
    assert updated.value == env.value


def test_confidence_bounds_are_enforced() -> None:
    with pytest.raises(PydanticValidationError):
        envelope(0.5, quantity="x", kind=SourceKind.MODEL, source_id="m", confidence=1.4)


def test_confidence_defaults_to_none_not_one() -> None:
    """An unknown confidence must stay unknown.

    Defaulting to 1.0 would silently assert certainty the model never expressed.
    """
    env = envelope(0.5, quantity="x", kind=SourceKind.MODEL, source_id="m")
    assert env.confidence is None


class TestGroundableValues:
    """The allow-list the Phase 11 grounding validator checks generated text against."""

    def test_scalar(self) -> None:
        env = envelope(0.87, quantity="risk", kind=SourceKind.MODEL, source_id="m")
        assert env.groundable_values() == [0.87]

    def test_includes_confidence_and_uncertainty(self) -> None:
        env = ProvenanceEnvelope[float](
            value=0.5,
            quantity="risk",
            source={"kind": SourceKind.MODEL, "id": "m"},  # type: ignore[arg-type]
            confidence=0.9,
            uncertainty={"ci_low": 0.3, "ci_high": 0.7},
        )
        got = set(env.groundable_values())
        assert got == {0.5, 0.9, 0.3, 0.7}

    def test_dict_value(self) -> None:
        env = envelope(
            {"flooded_km2": 143.2, "population_exposed": 48000.0},
            quantity="impact",
            kind=SourceKind.DERIVED,
            source_id="zonal_stats",
        )
        assert set(env.groundable_values()) == {143.2, 48000.0}

    def test_booleans_are_not_numbers(self) -> None:
        """bool is a subclass of int in Python; True must not ground the number 1."""
        env = envelope(True, quantity="is_flooded", kind=SourceKind.MODEL, source_id="m")
        assert env.groundable_values() == []

    def test_string_value_grounds_nothing(self) -> None:
        env = envelope("RED", quantity="risk_level", kind=SourceKind.INDEX, source_id="risk")
        assert env.groundable_values() == []


class TestTemporalValidity:
    def test_observation_age_reflects_acquisition_not_computation(self) -> None:
        """A flood extent computed now from a 9-day-old scene is 9 days old.

        Requirement 32: the system must not present processing time as data
        freshness.
        """
        acquired = datetime.now(UTC) - timedelta(days=9)
        tv = TemporalValidity(observed_at=acquired)
        age = tv.observation_age_hours
        assert age is not None
        assert 215.0 < age < 217.0

    def test_age_is_none_without_an_observation(self) -> None:
        assert TemporalValidity().observation_age_hours is None

    def test_validity_window_ordering_is_checked(self) -> None:
        now = datetime.now(UTC)
        with pytest.raises(PydanticValidationError):
            TemporalValidity(valid_from=now, valid_to=now - timedelta(hours=1))


def test_spatial_resolution_is_recorded() -> None:
    """Resolution provenance keeps coarse inputs from masquerading as fine ones."""
    spatial = SpatialRef(
        bbox=(72.75, 18.85, 73.20, 19.35), region_name="Mumbai MMR", resolution_m=11132.0
    )
    assert spatial.resolution_m == 11132.0


def test_inputs_record_specific_scenes() -> None:
    """'We used Sentinel-1' is not a method; a scene identifier is."""
    env = envelope(
        0.91,
        quantity="flood_probability",
        kind=SourceKind.MODEL,
        source_id="flood_unet",
        inputs=[
            DataSourceRef(
                dataset="COPERNICUS/S1_GRD",
                provider="gee",
                scene_id="S1A_IW_GRDH_1SDV_20240715T003112_20240715T003137_054821_06AC2F",
                acquired_at=datetime(2024, 7, 15, 0, 31, 12, tzinfo=UTC),
            )
        ],
    )
    assert env.inputs[0].scene_id is not None
    assert "S1A_IW_GRDH" in env.inputs[0].scene_id


def test_summary_line_is_readable() -> None:
    env = envelope(
        0.87,
        quantity="flood_risk",
        unit="probability",
        kind=SourceKind.MODEL,
        source_id="flood_unet",
        version="0.3.1",
    )
    line = env.summary_line()
    assert "flood_risk=0.87" in line
    assert "flood_unet@0.3.1" in line
