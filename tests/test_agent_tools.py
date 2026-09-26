"""Agent tools -- the only channel through which a number may reach a reply.

The design rule these tests enforce: a tool either returns a provenance
envelope or raises. There is no third path in which it returns a plausible
default, because a default would arrive in the language layer indistinguishable
from a measurement, and the validator downstream would wave it through -- it
checks that numbers come from envelopes, not that the envelopes are true.

So "no data" has to be an exception, loudly, at the boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from satai.agents.tools import (
    TOOL_REGISTRY,
    ToolResult,
    get_flood_prediction,
    get_location_statistics,
    get_satellite_data,
    get_tool,
)
from satai.errors import DataUnavailableError, ValidationError
from satai.provenance import SourceKind

NOW = datetime(2026, 9, 1, 6, 30, tzinfo=UTC)

REGION = {
    "id": "bihar_ganga",
    "name": "Middle Ganga plain, Bihar",
    "bbox": [84.0, 25.0, 86.5, 26.5],
    "utm_epsg": "EPSG:32645",
    "area_km2": 24_500.0,
}

FLOOD_ROW: dict[str, Any] = {
    "risk_index": 0.8734,
    "risk_band": "ORANGE",
    "flooded_area_km2": 431.5,
    "population_exposed": 1_240_000.0,
    "confidence": 0.71,
    "model_id": "flood_unet",
    "model_version": "1.2.0",
    "observed_at": NOW,
    "computed_at": NOW,
    "sources": [
        {
            "dataset": "COPERNICUS/S1_GRD",
            "provider": "cdse",
            "scene_id": "S1A_IW_GRDH_1SDV_20260830T003512_20260830T003537_055012_06B3A1",
            "acquired_at": NOW,
        }
    ],
    "caveats": ["SAR cannot see flooding beneath dense canopy."],
    "scene_ids": ["S1A_IW_GRDH_20260830T003512"],
}


class FakeStore:
    """In-memory store. The tools take the protocol, so no database is needed."""

    def __init__(self, *, region: dict | None = REGION, flood: dict | None = None) -> None:
        self._region = region
        self._flood = flood

    def region_info(self, region: str) -> dict | None:
        return self._region

    def latest_hazard(self, region: str, hazard: str) -> dict | None:
        return self._flood if hazard == "flood" else None

    def explanation(self, region: str, hazard: str) -> dict | None:
        return None

    def observations(self, region: str, kind: str, days: int) -> list[dict]:
        return []

    def history(self, region: str, hazard: str, months: int) -> list[dict]:
        return []


# --- the registry ------------------------------------------------------------


def test_every_tool_is_registered_with_a_description_and_parameters() -> None:
    assert TOOL_REGISTRY
    for name, spec in TOOL_REGISTRY.items():
        assert spec.name == name
        assert len(spec.description) > 40, f"{name} needs a usable description"
        assert spec.parameters


def test_registry_covers_the_documented_tool_surface() -> None:
    expected = {
        "get_flood_prediction",
        "get_satellite_data",
        "get_model_explanation",
        "get_active_fires",
        "get_risk_map",
        "get_historical_risk",
        "get_damage_assessment",
        "get_weather_data",
        "get_location_statistics",
    }
    assert expected <= set(TOOL_REGISTRY)


def test_tool_schemas_are_valid_anthropic_tool_definitions() -> None:
    for spec in TOOL_REGISTRY.values():
        schema = spec.to_anthropic_schema()
        assert schema["name"] == spec.name
        assert schema["input_schema"]["type"] == "object"
        assert "store" not in schema["input_schema"]["properties"]
        assert "store" not in schema["input_schema"]["required"]


def test_tool_descriptions_do_not_promise_forecasting() -> None:
    """The description is what the model reads when deciding to call a tool.

    A description saying a tool 'predicts flooding' would invite exactly the
    claim the rest of the system is built to prevent.
    """
    forbidden = ("will flood", "forecast of", "predicts when", "official warning is")
    for spec in TOOL_REGISTRY.values():
        lowered = spec.description.lower()
        assert not any(phrase in lowered for phrase in forbidden), spec.name


def test_unknown_tool_lookup_raises_and_lists_what_exists() -> None:
    with pytest.raises(ValidationError, match="unknown tool"):
        get_tool("get_earthquake_prediction")


# --- results carry provenance ------------------------------------------------


def test_a_flood_result_is_a_model_output_with_its_scenes_and_caveats() -> None:
    result = get_flood_prediction("bihar_ganga", FakeStore(flood=FLOOD_ROW))

    assert isinstance(result, ToolResult)
    assert not result.is_empty
    env = result.envelopes[0]

    assert env.source.kind is SourceKind.MODEL
    assert env.source.id == "flood_unet"
    assert env.value["risk_index"] == pytest.approx(0.8734)
    assert any("not an official warning" in c.lower() for c in env.caveats)


def test_every_number_in_the_summary_is_also_in_the_envelope() -> None:
    """Otherwise the summary becomes an ungrounded channel of its own."""
    result = get_flood_prediction("bihar_ganga", FakeStore(flood=FLOOD_ROW))
    groundable = result.envelopes[0].groundable_values()

    assert any(abs(v - 0.8734) < 0.01 for v in groundable)
    assert any(abs(v - 431.5) < 0.1 for v in groundable)


def test_the_regions_own_caveats_survive_into_the_envelope() -> None:
    result = get_flood_prediction("bihar_ganga", FakeStore(flood=FLOOD_ROW))
    assert any("canopy" in c for c in result.envelopes[0].caveats)


def test_location_statistics_are_catalogue_not_model_output() -> None:
    """An area in km2 is a record, not a prediction, and is labelled as one."""
    result = get_location_statistics("bihar_ganga", FakeStore())
    assert result.envelopes[0].source.kind is SourceKind.CATALOGUE


# --- absence is an exception, never a default --------------------------------


def test_an_unconfigured_region_raises_rather_than_estimating() -> None:
    """'What is the flood risk in Chennai?' must not get an answer."""
    with pytest.raises(DataUnavailableError) as excinfo:
        get_flood_prediction("chennai", FakeStore(region=None))

    assert "chennai" in str(excinfo.value).lower()


def test_a_configured_region_with_no_run_raises_and_explains_why() -> None:
    with pytest.raises(DataUnavailableError) as excinfo:
        get_flood_prediction("bihar_ganga", FakeStore(flood=None))

    message = str(excinfo.value)
    assert "batch" in message or "revisit" in message
    assert excinfo.value.detail["region"] == "bihar_ganga"
    assert excinfo.value.detail["hazard"] == "flood"
    assert excinfo.value.to_problem()["status"] == 404


def test_no_tool_returns_a_zero_valued_result_when_data_is_missing() -> None:
    """The failure mode this design rules out.

    A tool returning risk 0.0 for a region with no analysis would be read by
    the language layer as 'no risk here', which is a different and false claim
    from 'this has not been computed'.
    """
    store = FakeStore(flood=None)
    for tool_name in ("get_flood_prediction", "get_satellite_data"):
        spec = get_tool(tool_name)
        kwargs = {"region": "bihar_ganga", "store": store}
        if "hazard" in spec.parameters:
            kwargs["hazard"] = "flood"
        with pytest.raises(DataUnavailableError):
            spec.fn(**kwargs)


def test_satellite_provenance_reports_scene_identifiers_and_age() -> None:
    result = get_satellite_data("bihar_ganga", "flood", FakeStore(flood=FLOOD_ROW))
    env = result.envelopes[0]

    assert env.temporal.observed_at == NOW
    assert "revisit" in " ".join(env.caveats).lower() or "batch" in " ".join(env.caveats).lower()


# --- nullable columns are absent, not zero -----------------------------------


def test_a_susceptibility_run_without_an_extent_omits_the_key_entirely() -> None:
    """`flooded_area_km2` and `population_exposed` are nullable in the schema.

    Indexing them directly raised; defaulting them to 0.0 would be worse, because
    the value would enter `groundable_values()` and license the agent to state
    "0 km² flooded" — a measurement nobody made, arriving pre-authorised by the
    very mechanism meant to prevent it.
    """
    row = {**FLOOD_ROW, "flooded_area_km2": None, "population_exposed": None}
    result = get_flood_prediction("bihar_ganga", FakeStore(flood=row))
    env = result.envelopes[0]

    assert "flooded_area_km2" not in env.value
    assert "population_exposed" not in env.value
    assert env.value["risk_index"] == pytest.approx(0.8734)
    assert 0.0 not in env.groundable_values()
    assert "susceptibility only" in result.summary


def test_a_damage_row_without_a_delineated_area_still_reports_severity() -> None:
    row = {**FLOOD_ROW, "flooded_area_km2": None, "changed_structures": None}
    store = FakeStore(flood=None)
    store.latest_hazard = lambda region, hazard: row if hazard == "damage" else None  # type: ignore[method-assign]

    result = get_tool("get_damage_assessment").fn(region="bihar_ganga", store=store)
    env = result.envelopes[0]

    assert "affected_area_km2" not in env.value
    assert env.value["severity_index"] == pytest.approx(0.8734)


# --- partial FRP reporting ----------------------------------------------------


def _fire_store(rows: list[dict[str, Any]]) -> FakeStore:
    store = FakeStore()
    store.observations = lambda region, kind, days: rows  # type: ignore[method-assign]
    return store


def test_mean_frp_is_averaged_over_the_detections_that_report_it() -> None:
    """Dividing by len(rows) deflated the mean in proportion to what was missing.

    Three detections, two with FRP: the mean is (10+20)/2, not (10+20)/3.
    """
    rows = [
        {"frp_mw": 10.0, "observed_at": NOW},
        {"frp_mw": 20.0, "observed_at": NOW},
        {"observed_at": NOW},
    ]
    result = get_tool("get_active_fires").fn(
        region="bihar_ganga", days="7", store=_fire_store(rows)
    )
    env = result.envelopes[0]

    assert env.value["mean_frp_mw"] == pytest.approx(15.0)
    assert env.value["frp_reported_count"] == pytest.approx(2.0)
    assert env.value["detection_count"] == pytest.approx(3.0)
    assert any("2 of 3 detections" in c for c in env.caveats)


def test_no_mean_frp_is_reported_when_nothing_measures_it() -> None:
    rows = [{"observed_at": NOW}, {"observed_at": NOW}]
    result = get_tool("get_active_fires").fn(
        region="bihar_ganga", days="7", store=_fire_store(rows)
    )
    env = result.envelopes[0]

    assert "mean_frp_mw" not in env.value
    assert any("no detection" in c.lower() for c in env.caveats)


def test_fire_detections_are_labelled_observations_not_predictions() -> None:
    rows = [{"frp_mw": 12.0, "observed_at": NOW}]
    result = get_tool("get_active_fires").fn(
        region="bihar_ganga", days="7", store=_fire_store(rows)
    )
    assert result.envelopes[0].source.kind is SourceKind.OBSERVATION
    assert any("not predictions" in c for c in result.envelopes[0].caveats)
