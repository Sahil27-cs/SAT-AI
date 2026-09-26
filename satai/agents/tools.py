"""Agent tools: the only source of project-specific values.

Every tool returns a :class:`~satai.provenance.ProvenanceEnvelope`, never a
bare number. That is what makes the grounding measurement (C1) possible: the
union of the turn's envelopes defines exactly what the language model is
permitted to state, and anything else in its output is a violation.

Tools read from the **serving plane** -- precomputed artifacts in the database
and object storage (ADR-001). No tool triggers model inference or raster
processing. A tool call is a lookup, which is why the agent layer can answer in
seconds while the underlying analysis is a revisit-limited batch job.

When a tool has no data for a request it raises
:class:`~satai.errors.DataUnavailableError`. It does **not** return a default,
an estimate, or an empty envelope that reads like zero. "We have no observation
for that area" is a correct and useful answer; a fabricated zero is neither.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from satai.errors import DataUnavailableError, ValidationError
from satai.logging import get_logger
from satai.provenance import (
    DataSourceRef,
    HazardType,
    ProvenanceEnvelope,
    SourceKind,
    SpatialRef,
    TemporalValidity,
    envelope,
)

log = get_logger(__name__)

__all__ = ["TOOL_REGISTRY", "ToolResult", "ToolSpec", "get_tool", "tool"]


@dataclass(frozen=True)
class ToolResult:
    """What a tool returns: envelopes plus a short factual summary."""

    envelopes: list[ProvenanceEnvelope[Any]]
    summary: str
    tool_name: str

    @property
    def is_empty(self) -> bool:
        return not self.envelopes


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool, with the metadata the router needs."""

    name: str
    description: str
    fn: Callable[..., ToolResult]
    parameters: dict[str, str]
    hazards: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Anthropic tool-use schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    key: {"type": "string", "description": desc}
                    for key, desc in self.parameters.items()
                },
                "required": [
                    name
                    for name, param in inspect.signature(self.fn).parameters.items()
                    if param.default is inspect.Parameter.empty and name != "store"
                ],
            },
        }


TOOL_REGISTRY: dict[str, ToolSpec] = {}


def tool(
    *,
    description: str,
    parameters: dict[str, str],
    hazards: tuple[str, ...] = (),
    agents: tuple[str, ...] = (),
) -> Callable[[Callable[..., ToolResult]], Callable[..., ToolResult]]:
    """Register a function as an agent tool."""

    def decorator(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        TOOL_REGISTRY[fn.__name__] = ToolSpec(
            name=fn.__name__,
            description=description,
            fn=fn,
            parameters=parameters,
            hazards=hazards,
            agents=agents,
        )
        return fn

    return decorator


def get_tool(name: str) -> ToolSpec:
    if name not in TOOL_REGISTRY:
        raise ValidationError(
            f"unknown tool {name!r}; registered: {', '.join(sorted(TOOL_REGISTRY))}"
        )
    return TOOL_REGISTRY[name]


# ---------------------------------------------------------------------------
# Store protocol. The serving plane injects a concrete implementation; the
# tools stay testable without a database.
# ---------------------------------------------------------------------------


class ResultStore:
    """Read-only access to precomputed results. Implemented by the API layer."""

    def latest_hazard(self, region: str, hazard: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def explanation(self, region: str, hazard: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def observations(self, region: str, kind: str, days: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    def history(self, region: str, hazard: str, months: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    def region_info(self, region: str) -> dict[str, Any] | None:
        raise NotImplementedError


def _spatial(info: dict[str, Any] | None) -> SpatialRef | None:
    if not info:
        return None
    return SpatialRef(
        bbox=tuple(info["bbox"]) if info.get("bbox") else None,
        region_id=info.get("id"),
        region_name=info.get("name"),
        resolution_m=info.get("resolution_m"),
    )


def _require_region(store: ResultStore, region: str) -> dict[str, Any]:
    info = store.region_info(region)
    if info is None:
        raise DataUnavailableError(
            f"no region matching {region!r} is configured in SAT-AI",
            region=region,
        )
    return info


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    description=(
        "Current SAT-AI prototype flood risk for a region: risk index, band, "
        "affected-area estimate and the contributing factors. Returns model and "
        "index outputs, never an official warning."
    ),
    parameters={"region": "Region identifier or name, e.g. 'bihar_ganga'"},
    hazards=("flood",),
    agents=("risk_analyst", "emergency"),
)
def get_flood_prediction(region: str, store: ResultStore) -> ToolResult:
    info = _require_region(store, region)
    row = store.latest_hazard(region, "flood")
    if row is None:
        raise DataUnavailableError(
            f"no flood analysis has been computed for {info['name']}. SAT-AI covers "
            f"only its configured study areas, and results are produced on a batch "
            f"schedule tied to Sentinel-1 revisit.",
            region=region,
            hazard="flood",
        )

    # `flooded_area_km2` and `population_exposed` are nullable in the schema: a
    # susceptibility run produces a risk index with no mapped extent at all.
    # Omitting a missing key is the honest encoding -- a 0.0 would enter the
    # envelope's groundable values and license the model to state "0 km2
    # flooded", which is a measurement nobody made.
    values: dict[str, float] = {"risk_index": float(row["risk_index"])}
    for key in ("flooded_area_km2", "population_exposed"):
        if row.get(key) is not None:
            values[key] = float(row[key])

    env = envelope(
        values,
        quantity="flood_risk",
        unit="index",
        kind=SourceKind.MODEL,
        source_id=row["model_id"],
        version=row["model_version"],
        confidence=row.get("confidence"),
        spatial=_spatial(info),
        temporal=TemporalValidity(
            observed_at=row.get("observed_at"), computed_at=row["computed_at"]
        ),
        hazard=HazardType.FLOOD,
        inputs=[DataSourceRef(**s) for s in row.get("sources", [])],
        caveats=[
            "SAT-AI prototype risk level, not an official warning. Official flood "
            "warnings for India come from IMD, NDMA and State Disaster Management "
            "Authorities.",
            *row.get("caveats", []),
        ],
        risk_band=row["risk_band"],
    )
    extent = (
        f"{values['flooded_area_km2']:.1f} km² flood extent"
        if "flooded_area_km2" in values
        else "no mapped extent (susceptibility only)"
    )
    return ToolResult(
        envelopes=[env],
        summary=(
            f"{info['name']}: flood risk index {row['risk_index']:.2f} "
            f"(band {row['risk_band']}), {extent} from "
            f"{row['model_id']} v{row['model_version']}."
        ),
        tool_name="get_flood_prediction",
    )


@tool(
    description=(
        "Which satellite scenes and datasets produced the current analysis for a "
        "region: scene identifiers, acquisition times and data age."
    ),
    parameters={"region": "Region identifier", "hazard": "flood | wildfire | cyclone"},
    agents=("risk_analyst", "recovery"),
)
def get_satellite_data(region: str, hazard: str, store: ResultStore) -> ToolResult:
    info = _require_region(store, region)
    row = store.latest_hazard(region, hazard)
    if row is None or not row.get("sources"):
        raise DataUnavailableError(
            f"no scene provenance recorded for {hazard} over {info['name']}",
            region=region,
            hazard=hazard,
        )

    sources = [DataSourceRef(**s) for s in row["sources"]]
    observed = row.get("observed_at")
    age_hours = (datetime.now(UTC) - observed).total_seconds() / 3600.0 if observed else None

    env = envelope(
        {
            "scene_count": float(len(sources)),
            **({"data_age_hours": age_hours} if age_hours else {}),
        },
        quantity="satellite_provenance",
        unit="count",
        kind=SourceKind.CATALOGUE,
        source_id="scene_registry",
        version="1.0.0",
        spatial=_spatial(info),
        temporal=TemporalValidity(observed_at=observed, computed_at=row["computed_at"]),
        inputs=sources,
        caveats=[
            "Sentinel-1 revisit is 6-12 days, so satellite-derived extent is "
            "revisit-limited batch analysis, not real-time monitoring."
        ],
    )
    listed = "; ".join(f"{s.dataset} {s.scene_id}" for s in sources[:4])
    return ToolResult(
        envelopes=[env],
        summary=f"{len(sources)} scene(s) for {hazard} over {info['name']}: {listed}",
        tool_name="get_satellite_data",
    )


@tool(
    description=(
        "Why the model assigned the risk it did: per-feature attributions from "
        "SHAP or modality ablation, with the actual feature values."
    ),
    parameters={"region": "Region identifier", "hazard": "flood | wildfire | cyclone"},
    agents=("risk_analyst",),
)
def get_model_explanation(region: str, hazard: str, store: ResultStore) -> ToolResult:
    info = _require_region(store, region)
    row = store.explanation(region, hazard)
    if row is None:
        raise DataUnavailableError(
            f"no explanation artifact for {hazard} over {info['name']}. "
            f"Explanations are computed in the batch plane alongside the prediction.",
            region=region,
            hazard=hazard,
        )

    drivers = row["drivers"]
    env = envelope(
        {d["feature"]: float(d["value"]) for d in drivers}
        | {f"{d['feature']}_contribution": float(d["contribution"]) for d in drivers},
        quantity=f"{hazard}_explanation",
        unit="mixed",
        kind=SourceKind.DERIVED,
        source_id=row["method"],
        version=row["version"],
        spatial=_spatial(info),
        hazard=HazardType(hazard) if hazard in {h.value for h in HazardType} else None,
        caveats=[
            "Attributions describe what the model used, not physical causation.",
            *row.get("caveats", []),
        ],
        method=row["method"],
    )
    top = ", ".join(
        f"{d['feature']}={d['value']:g} (contribution {d['contribution']:+.3f})"
        for d in drivers[:4]
    )
    return ToolResult(
        envelopes=[env],
        summary=f"Top drivers for {hazard} in {info['name']}: {top}",
        tool_name="get_model_explanation",
    )


@tool(
    description=(
        "Active fire detections from NASA FIRMS for a region. These are satellite "
        "OBSERVATIONS at overpass time, not predictions of fire."
    ),
    parameters={"region": "Region identifier", "days": "Look-back window in days"},
    hazards=("wildfire",),
    agents=("risk_analyst", "emergency"),
)
def get_active_fires(region: str, days: str, store: ResultStore) -> ToolResult:
    info = _require_region(store, region)
    window = int(days)
    rows = store.observations(region, "active_fire", window)
    if not rows:
        raise DataUnavailableError(
            f"no FIRMS detections recorded for {info['name']} in the last {window} days. "
            f"Absence of detections is not evidence of no fire: cloud cover, small "
            f"fires and fires between overpasses are all undetected.",
            region=region,
        )

    # FRP is not reported for every detection. Averaging the sum over ALL rows
    # rather than over the rows that carry a value silently deflates the mean in
    # proportion to how many are missing it -- and the deflated figure then
    # enters the envelope as a groundable value the agent may state as fact.
    # Averaged over its own denominator, and omitted entirely when nothing
    # reports it.
    frp = [float(r["frp_mw"]) for r in rows if r.get("frp_mw") is not None]

    counts: dict[str, float] = {
        "detection_count": float(len(rows)),
        "high_confidence_count": float(sum(1 for r in rows if r.get("high_confidence"))),
    }
    if frp:
        counts["mean_frp_mw"] = sum(frp) / len(frp)
        counts["frp_reported_count"] = float(len(frp))

    env = envelope(
        counts,
        quantity="active_fire_detections",
        unit="count",
        kind=SourceKind.OBSERVATION,
        source_id="firms_viirs",
        version="1.0.0",
        spatial=_spatial(info),
        temporal=TemporalValidity(observed_at=rows[0].get("observed_at")),
        hazard=HazardType.WILDFIRE,
        caveats=[
            "Active fire detections are satellite observations at overpass time, "
            "not predictions, and not a complete census of fires.",
            "In India a large share of October-November detections are "
            "agricultural residue burning rather than forest fire.",
            "FIRMS near-real-time latency is approximately 3 hours.",
            *(
                [
                    f"Fire radiative power is reported for {len(frp)} of "
                    f"{len(rows)} detections; the mean covers only those."
                ]
                if frp and len(frp) < len(rows)
                else []
            ),
            *([] if frp else ["No detection in this window reports fire radiative power."]),
        ],
    )
    return ToolResult(
        envelopes=[env],
        summary=(
            f"{len(rows)} FIRMS detections over {info['name']} in {window} days "
            f"(observations, not predictions)."
        ),
        tool_name="get_active_fires",
    )


@tool(
    description=(
        "Multi-hazard risk vector for a region: per-hazard indices and the dominant hazard."
    ),
    parameters={"region": "Region identifier"},
    agents=("risk_analyst", "emergency"),
)
def get_risk_map(region: str, store: ResultStore) -> ToolResult:
    info = _require_region(store, region)
    hazards = ["flood", "wildfire", "cyclone"]
    values: dict[str, float] = {}
    caveats: list[str] = []

    for hazard in hazards:
        row = store.latest_hazard(region, hazard)
        if row:
            values[f"{hazard}_risk"] = float(row["risk_index"])
            caveats.extend(row.get("caveats", []))

    if not values:
        raise DataUnavailableError(
            f"no hazard analyses available for {info['name']}", region=region
        )

    dominant = max(values.items(), key=lambda kv: kv[1])
    env = envelope(
        values,
        quantity="multi_hazard_risk_vector",
        unit="index",
        kind=SourceKind.INDEX,
        source_id="risk_engine",
        version="1.0.0",
        spatial=_spatial(info),
        hazard=HazardType.MULTI,
        caveats=[
            "SAT-AI prototype risk levels, not official warnings.",
            "Per-hazard scores are NOT averaged: they come from different models "
            "with different base rates and are not commensurable.",
            *dict.fromkeys(caveats),
        ],
        dominant_hazard=dominant[0].replace("_risk", ""),
    )
    listed = ", ".join(f"{k.replace('_risk', '')} {v:.2f}" for k, v in sorted(values.items()))
    return ToolResult(
        envelopes=[env],
        summary=(
            f"{info['name']} risk vector: {listed}. Dominant: {dominant[0].replace('_risk', '')}."
        ),
        tool_name="get_risk_map",
    )


@tool(
    description="Historical hazard record for a region over the past N months.",
    parameters={"region": "Region identifier", "hazard": "Hazard type", "months": "Months back"},
    agents=("risk_analyst", "recovery"),
)
def get_historical_risk(region: str, hazard: str, months: str, store: ResultStore) -> ToolResult:
    info = _require_region(store, region)
    window = int(months)
    rows = store.history(region, hazard, window)
    if not rows:
        raise DataUnavailableError(
            f"no {hazard} history for {info['name']} over {window} months", region=region
        )

    indices = [float(r["risk_index"]) for r in rows]
    env = envelope(
        {
            "observation_count": float(len(rows)),
            "mean_risk": sum(indices) / len(indices),
            "max_risk": max(indices),
            "min_risk": min(indices),
        },
        quantity=f"{hazard}_history",
        unit="index",
        kind=SourceKind.CATALOGUE,
        source_id="result_archive",
        version="1.0.0",
        spatial=_spatial(info),
        temporal=TemporalValidity(
            valid_from=datetime.now(UTC) - timedelta(days=30 * window),
            valid_to=datetime.now(UTC),
        ),
        caveats=[
            "Historical values are previous SAT-AI model outputs, not independent "
            "observations of what occurred."
        ],
    )
    return ToolResult(
        envelopes=[env],
        summary=(
            f"{info['name']} {hazard} over {window} months: {len(rows)} analyses, "
            f"mean index {sum(indices) / len(indices):.2f}, max {max(indices):.2f}."
        ),
        tool_name="get_historical_risk",
    )


@tool(
    description="Post-event damage assessment: affected area, changed structures and severity.",
    parameters={"region": "Region identifier"},
    agents=("recovery",),
)
def get_damage_assessment(region: str, store: ResultStore) -> ToolResult:
    info = _require_region(store, region)
    row = store.latest_hazard(region, "damage")
    if row is None:
        raise DataUnavailableError(
            f"no damage assessment for {info['name']}. Damage assessment requires a "
            f"pre/post image pair around a specific event.",
            region=region,
        )

    # Same nullability as the flood row: a damage run may carry a severity index
    # without a delineated area, or count changed structures without either. A
    # key that is absent from the envelope is a quantity the agent cannot state,
    # which is the behaviour we want; a defaulted 0.0 is a fabricated count.
    damage: dict[str, float] = {"severity_index": float(row["risk_index"])}
    if row.get("flooded_area_km2") is not None:
        damage["affected_area_km2"] = float(row["flooded_area_km2"])
    if row.get("changed_structures") is not None:
        damage["changed_structures"] = float(row["changed_structures"])

    env = envelope(
        damage,
        quantity="damage_assessment",
        unit="mixed",
        kind=SourceKind.MODEL,
        source_id=row["model_id"],
        version=row["model_version"],
        confidence=row.get("confidence"),
        spatial=_spatial(info),
        temporal=TemporalValidity(
            observed_at=row.get("observed_at"), computed_at=row["computed_at"]
        ),
        caveats=[
            "Change detected from imagery, not verified on the ground.",
            "No monetary damage estimate is produced: SAT-AI has no asset-value data.",
            *(row.get("caveats") or []),
        ],
    )
    affected = (
        f"{damage['affected_area_km2']:.1f} km² affected"
        if "affected_area_km2" in damage
        else "no delineated area"
    )
    return ToolResult(
        envelopes=[env],
        summary=f"{info['name']}: {affected}, severity {row['risk_index']:.2f}.",
        tool_name="get_damage_assessment",
    )


@tool(
    description="Weather observations and reanalysis driving the current risk estimate.",
    parameters={"region": "Region identifier", "days": "Look-back window in days"},
    agents=("risk_analyst", "emergency"),
)
def get_weather_data(region: str, days: str, store: ResultStore) -> ToolResult:
    info = _require_region(store, region)
    window = int(days)
    rows = store.observations(region, "weather", window)
    if not rows:
        raise DataUnavailableError(
            f"no weather records for {info['name']} in the last {window} days", region=region
        )

    latest = rows[0]
    env = envelope(
        {k: float(v) for k, v in latest.items() if isinstance(v, (int, float))},
        quantity="weather_observations",
        unit="mixed",
        kind=SourceKind.REANALYSIS,
        source_id=latest.get("source", "era5_land"),
        version="1.0.0",
        spatial=_spatial(info),
        temporal=TemporalValidity(observed_at=latest.get("observed_at")),
        caveats=[
            "ERA5 reanalysis has roughly a five-day lag and is retrospective.",
            "GPM IMERG Early is near-real-time at approximately 4 hours latency.",
            "Precipitation at ~11 km cannot resolve urban convective cells.",
        ],
    )
    return ToolResult(
        envelopes=[env],
        summary=f"Latest weather for {info['name']} from {latest.get('source', 'reanalysis')}.",
        tool_name="get_weather_data",
    )


@tool(
    description=(
        "Static facts about a configured region: bounding box, area, population, study role."
    ),
    parameters={"region": "Region identifier or name"},
    agents=("risk_analyst", "emergency", "recovery"),
)
def get_location_statistics(region: str, store: ResultStore) -> ToolResult:
    info = _require_region(store, region)
    env = envelope(
        {
            "area_km2": float(info["area_km2"]),
            "population": float(info.get("population", 0)),
            "tile_count": float(info.get("tile_count", 0)),
        },
        quantity="region_statistics",
        unit="mixed",
        kind=SourceKind.CATALOGUE,
        source_id="aoi_registry",
        version="1.0.0",
        spatial=_spatial(info),
        caveats=info.get("caveats", []),
        study_role=info.get("study_role", "unspecified"),
    )
    return ToolResult(
        envelopes=[env],
        summary=(
            f"{info['name']}: {info['area_km2']:,.0f} km², role in study: "
            f"{info.get('study_role', 'unspecified')}."
        ),
        tool_name="get_location_statistics",
    )
