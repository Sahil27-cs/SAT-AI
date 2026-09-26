"""The provenance contract.

This module is the scientific backbone of SAT-AI. Every quantitative value that
crosses a plane boundary -- from the ML plane to the serving plane, and from the
serving plane to the agent plane -- must be wrapped in a
:class:`ProvenanceEnvelope`.

Why this exists
---------------
A conversational interface over scientific data has exactly one catastrophic
failure mode: producing a plausible number that no model ever computed. A user
asking "what is the flood risk in Mumbai?" and receiving a confident invented
0.82 is worse than receiving nothing, because it is indistinguishable from a
real answer.

The defence is structural rather than aspirational. The language model never
receives a bare float. It receives an envelope carrying the value, the model
that produced it, the satellite scenes that fed that model, the validity window,
and the caveats. It may rephrase an envelope. It may not author one. A
validator in the agent graph (Phase 11) re-extracts every number from the
generated text and rejects the response if a number is not traceable to an
envelope in that turn.

This also gives us, for free, the audit trail a research prototype needs: any
figure in the final report can be traced back to the scene IDs it came from.

See ``docs/adr/ADR-003-provenance-and-config-contract.md``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "DataSourceRef",
    "HazardType",
    "ProvenanceEnvelope",
    "SourceKind",
    "SourceRef",
    "SpatialRef",
    "TemporalValidity",
    "envelope",
]

T = TypeVar("T")


class SourceKind(StrEnum):
    """What kind of process produced a value.

    The distinction matters scientifically and is surfaced to the user. An
    active-fire detection from NASA FIRMS is an *observation*; a fire-danger
    score is a *model* output. Presenting them identically would be misleading.
    """

    OBSERVATION = "observation"
    """Measured by an instrument. Not predicted. e.g. FIRMS fire detection."""

    MODEL = "model"
    """Output of a trained statistical or ML model. e.g. flood segmentation."""

    DERIVED = "derived"
    """Deterministic computation over other values. e.g. dNBR, slope, NDVI."""

    INDEX = "index"
    """A documented, configurable composite score. Not a trained prediction.
    e.g. the multi-hazard risk score. Must never be described as a forecast."""

    REANALYSIS = "reanalysis"
    """Model-assimilated historical fields. e.g. ERA5. Retrospective, not live."""

    CATALOGUE = "catalogue"
    """A curated historical record. e.g. IBTrACS best tracks, EM-DAT events."""


class HazardType(StrEnum):
    """Hazards SAT-AI handles.

    ``EARTHQUAKE`` appears here for *post-event damage assessment only*.
    SAT-AI does not and will not predict earthquakes.
    """

    FLOOD = "flood"
    WILDFIRE = "wildfire"
    CYCLONE = "cyclone"
    EARTHQUAKE = "earthquake"
    MULTI = "multi"


class SpatialRef(BaseModel):
    """Where a value applies."""

    model_config = ConfigDict(frozen=True)

    crs: str = Field(default="EPSG:4326", description="Coordinate reference system.")
    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description="(min_lon, min_lat, max_lon, max_lat) in ``crs``."
    )
    region_id: str | None = Field(default=None, description="Admin or AOI identifier.")
    region_name: str | None = Field(default=None)
    resolution_m: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Native resolution of the value. Critical for honesty: a risk score "
            "derived from 11 km IMERG carries no sub-kilometre information even "
            "when rendered on a 10 m grid."
        ),
    )


class TemporalValidity(BaseModel):
    """When a value is valid, and how stale it is.

    ``observed_at`` is when the underlying measurement was taken;
    ``computed_at`` is when SAT-AI processed it. The gap between the two is the
    latency the user must be shown. A Sentinel-1 flood extent computed a minute
    ago from a scene acquired nine days ago is nine days old, not one minute old,
    and the interface must say so (requirement 32).
    """

    model_config = ConfigDict(frozen=True)

    observed_at: datetime | None = Field(
        default=None, description="Acquisition time of the underlying observation (UTC)."
    )
    computed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When SAT-AI computed this."
    )
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    @property
    def observation_age_hours(self) -> float | None:
        """Hours between the observation and now. ``None`` if not observational."""
        if self.observed_at is None:
            return None
        return (datetime.now(UTC) - self.observed_at).total_seconds() / 3600.0

    @model_validator(mode="after")
    def _check_ordering(self) -> TemporalValidity:
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("valid_from must not be after valid_to")
        return self


class DataSourceRef(BaseModel):
    """A specific dataset -- and where possible a specific scene -- used as input.

    Recording the scene identifier, not merely "Sentinel-1", is what makes a
    result reproducible. "We used Sentinel-1" is not a method;
    ``S1A_IW_GRDH_1SDV_20240715T003112_...`` is.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str = Field(description="Collection, e.g. 'COPERNICUS/S1_GRD'.")
    provider: str = Field(description="Access route, e.g. 'gee', 'cdse', 'firms'.")
    scene_id: str | None = Field(default=None, description="Granule / product identifier.")
    acquired_at: datetime | None = None
    access_url: str | None = None
    notes: str | None = None


class SourceRef(BaseModel):
    """The process that produced the value."""

    model_config = ConfigDict(frozen=True)

    kind: SourceKind
    id: str = Field(description="Model or method identifier, e.g. 'flood_unet'.")
    version: str = Field(default="0.0.0", description="Semantic version of that model.")
    run_id: str | None = Field(default=None, description="Batch run that produced it.")
    git_sha: str | None = Field(default=None, description="Commit the run was executed at.")
    config_hash: str | None = Field(default=None, description="Hash of the run configuration.")


class ProvenanceEnvelope(BaseModel, Generic[T]):
    """A value plus everything needed to defend it.

    Example
    -------
    >>> from satai.provenance import envelope, SourceKind
    >>> env = envelope(
    ...     0.87,
    ...     quantity="flood_extent_confidence",
    ...     unit="probability",
    ...     kind=SourceKind.MODEL,
    ...     source_id="flood_unet",
    ...     version="0.3.1",
    ... )
    >>> env.value
    0.87
    >>> env.is_model_output
    True
    """

    model_config = ConfigDict(frozen=True)

    envelope_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    value: T = Field(description="The value itself.")
    quantity: str = Field(description="What is measured, e.g. 'flood_probability'.")
    unit: str = Field(
        default="dimensionless",
        description="Physical unit or 'probability', 'index', 'count', 'km2'.",
    )

    source: SourceRef
    inputs: list[DataSourceRef] = Field(default_factory=list)
    spatial: SpatialRef | None = None
    temporal: TemporalValidity = Field(default_factory=TemporalValidity)

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Model confidence where the model provides a calibrated one. Left "
            "None rather than invented -- an unknown confidence is not 1.0."
        ),
    )
    uncertainty: dict[str, float] | None = Field(
        default=None, description="e.g. {'ci_low': 0.71, 'ci_high': 0.93}."
    )

    caveats: list[str] = Field(
        default_factory=list,
        description=(
            "Machine-generated honesty. e.g. 'optical modality unavailable "
            "(cloud cover 78%)', 'trained on Sen1Floods11; India generalisation "
            "untested'. Surfaced verbatim in the UI and given to the agent."
        ),
    )
    hazard: HazardType | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    # -- convenience -------------------------------------------------------

    @property
    def is_model_output(self) -> bool:
        """True if a trained model produced this, rather than an instrument."""
        return self.source.kind is SourceKind.MODEL

    @property
    def is_observation(self) -> bool:
        """True if an instrument measured this. Must not be called a prediction."""
        return self.source.kind is SourceKind.OBSERVATION

    def with_caveat(self, caveat: str) -> ProvenanceEnvelope[T]:
        """Return a copy with an extra caveat (envelopes are immutable)."""
        return self.model_copy(update={"caveats": [*self.caveats, caveat]})

    def groundable_values(self) -> list[float]:
        """Numeric values a generated response is permitted to state.

        Consumed by the agent plane's grounding validator (Phase 11): any number
        appearing in generated text that is not in the union of these lists,
        across this turn's envelopes, causes the response to be regenerated.
        """
        out: list[float] = []
        if isinstance(self.value, bool):
            pass
        elif isinstance(self.value, (int, float)):
            out.append(float(self.value))
        elif isinstance(self.value, dict):
            out.extend(
                float(v)
                for v in self.value.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            )
        elif isinstance(self.value, (list, tuple)):
            out.extend(
                float(v)
                for v in self.value
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            )
        if self.confidence is not None:
            out.append(self.confidence)
        if self.uncertainty:
            out.extend(float(v) for v in self.uncertainty.values())
        return out

    def summary_line(self) -> str:
        """One-line human summary, used in logs and citation cards."""
        age = self.temporal.observation_age_hours
        age_str = f", observed {age:.1f}h ago" if age is not None else ""
        return (
            f"{self.quantity}={self.value} {self.unit} "
            f"[{self.source.kind}:{self.source.id}@{self.source.version}{age_str}]"
        )


def envelope(
    value: T,
    *,
    quantity: str,
    kind: SourceKind,
    source_id: str,
    unit: str = "dimensionless",
    version: str = "0.0.0",
    inputs: list[DataSourceRef] | None = None,
    spatial: SpatialRef | None = None,
    temporal: TemporalValidity | None = None,
    confidence: float | None = None,
    caveats: list[str] | None = None,
    hazard: HazardType | None = None,
    **extra: Any,
) -> ProvenanceEnvelope[T]:
    """Construct a :class:`ProvenanceEnvelope` with less ceremony.

    Keyword-only by design: a positional call site would make it easy to attach
    the wrong source to a value, which is precisely the error this module exists
    to prevent.
    """
    return ProvenanceEnvelope[T](
        value=value,
        quantity=quantity,
        unit=unit,
        source=SourceRef(kind=kind, id=source_id, version=version),
        inputs=inputs or [],
        spatial=spatial,
        temporal=temporal or TemporalValidity(),
        confidence=confidence,
        caveats=caveats or [],
        hazard=hazard,
        extra=extra,
    )
