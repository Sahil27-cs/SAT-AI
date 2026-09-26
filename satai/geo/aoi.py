"""Areas of interest: definition, validation, and CRS selection.

Deliberately dependency-free (no GDAL, rasterio or geopandas). AOI definition
and bookkeeping must work in CI, in a bare test environment, and on a machine
where the geospatial stack has not been installed yet. Heavy geometry
operations belong in the preprocessing layer (Phase 3), not here.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from satai.errors import ValidationError

__all__ = [
    "AOI",
    "AOIRegistry",
    "HazardEvent",
    "StudyRole",
    "load_aoi_registry",
    "utm_epsg_for",
]

# India's approximate bounding box, used as a soft sanity check on study areas.
INDIA_BBOX: tuple[float, float, float, float] = (68.0, 6.0, 97.5, 37.5)


def utm_epsg_for(lon: float, lat: float) -> str:
    """Return the EPSG code of the UTM zone containing ``(lon, lat)``.

    Analysis happens in a projected CRS, not EPSG:4326. Computing area, slope,
    or a 512x512 pixel tile in degrees is wrong -- a degree of longitude is
    ~111 km at the equator and ~96 km at 30 degrees N, so a "10 m" grid defined
    in degrees is not square and not 10 m. Mumbai and Bihar both fall in UTM
    zones 43N/45N, hence EPSG:326xx.

    Examples
    --------
    >>> utm_epsg_for(72.87, 19.07)     # Mumbai
    'EPSG:32643'
    >>> utm_epsg_for(85.14, 25.60)     # Patna, Bihar
    'EPSG:32645'
    """
    if not -180.0 <= lon <= 180.0:
        raise ValidationError(f"longitude out of range: {lon}")
    if not -90.0 <= lat <= 90.0:
        raise ValidationError(f"latitude out of range: {lat}")
    zone = math.floor((lon + 180.0) / 6.0) + 1
    zone = min(zone, 60)
    base = 32600 if lat >= 0 else 32700
    return f"EPSG:{base + zone}"


class StudyRole(StrEnum):
    """Why an AOI is in the study. Surfaced in the interface.

    The distinction is what separates a research instrument from a dashboard: a
    region held out to measure a transfer gap is doing different work from one
    the model was fitted on, and reporting them together without saying which is
    which is how an inflated number gets published.
    """

    TRAINING = "training"
    """Model is fitted on chips from this region."""

    TRANSFER_EVALUATION = "transfer_evaluation"
    """Held out entirely. Measures generalisation to an unseen setting."""

    CANDIDATE = "candidate"
    """Configured, not yet promoted on measured evidence (ADR-007)."""


class HazardEvent(BaseModel):
    """A specific, dated hazard event with verifiable satellite coverage.

    ``verified`` is the load-bearing field and it is never set from a news
    report. It means: a named sensor, a specific acquisition, and at least one
    publicly retrievable reference product have each been checked. An event that
    is real but whose imagery has not been confirmed stays ``verified: false``
    and the interface labels it CANDIDATE rather than showing it as analysis.

    Reference products are other organisations' delineations -- UNOSAT, Copernicus
    EMS. They are **label and validation data**, never SAT-AI outputs, and the
    serving layer must not present their figures as this system's results.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    hazard: str
    occurred_on: date = Field(description="Event onset date (UTC).")
    sensor: str = Field(description="Instrument behind the reference analysis.")
    acquisition_utc: datetime | None = Field(
        default=None, description="Scene acquisition time, where a product states it."
    )
    reference_products: list[str] = Field(
        default_factory=list,
        description="Public delineations covering this event. Not SAT-AI outputs.",
    )
    verified: bool = Field(
        default=False,
        description="True only when sensor, date and a retrievable product were checked.",
    )
    verification_note: str = Field(
        min_length=20, description="What was checked, and against what source."
    )

    @property
    def display_status(self) -> str:
        """What the interface is permitted to call this event."""
        return "HISTORICAL EVENT ANALYSIS" if self.verified else "CANDIDATE — UNVERIFIED"


class AOI(BaseModel):
    """A study area.

    ``selection_rationale`` is mandatory. Study areas in this project are chosen
    on evidence -- label availability, acquisition density, hazard recurrence,
    exposure -- not on familiarity. Forcing the justification into the data
    model means it ends up in the methodology chapter instead of being
    reconstructed afterwards.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[a-z0-9_]+$", description="Stable slug, e.g. 'bihar_ganga'.")
    name: str
    country: str = "India"
    bbox: tuple[float, float, float, float] = Field(
        description="(min_lon, min_lat, max_lon, max_lat) in EPSG:4326."
    )
    primary_hazards: list[str] = Field(default_factory=list)
    selection_rationale: str = Field(
        min_length=20, description="Why this area was chosen. Required."
    )
    label_sources: list[str] = Field(
        default_factory=list,
        description="Ground-truth datasets covering this AOI. Empty means supervised "
        "training here is not possible -- only transfer from elsewhere.",
    )
    status: str = Field(default="candidate", description="candidate | selected | rejected")
    study_role: StudyRole = Field(
        default=StudyRole.CANDIDATE,
        description="Training, held-out transfer target, or not yet promoted.",
    )
    events: list[HazardEvent] = Field(
        default_factory=list,
        description="Dated events with checked satellite coverage over this AOI.",
    )
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_bbox(self) -> AOI:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        if min_lon >= max_lon:
            raise ValueError(f"bbox min_lon ({min_lon}) must be < max_lon ({max_lon})")
        if min_lat >= max_lat:
            raise ValueError(f"bbox min_lat ({min_lat}) must be < max_lat ({max_lat})")
        if not (min_lon >= -180 and max_lon <= 180 and min_lat >= -90 and max_lat <= 90):
            raise ValueError(f"bbox outside valid lon/lat ranges: {self.bbox}")
        return self

    @property
    def centroid(self) -> tuple[float, float]:
        """Approximate centre as ``(lon, lat)``."""
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return ((min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0)

    @property
    def utm_epsg(self) -> str:
        """Projected CRS used for analysis over this AOI."""
        lon, lat = self.centroid
        return utm_epsg_for(lon, lat)

    @property
    def approx_area_km2(self) -> float:
        """Rough area, using a cosine-latitude correction.

        Approximate by design -- it sizes compute budgets and tile counts. Exact
        area comes from a projected geometry in the preprocessing layer.
        """
        min_lon, min_lat, max_lon, max_lat = self.bbox
        mean_lat_rad = math.radians((min_lat + max_lat) / 2.0)
        km_per_deg_lat = 110.574
        km_per_deg_lon = 111.320 * math.cos(mean_lat_rad)
        return (max_lon - min_lon) * km_per_deg_lon * (max_lat - min_lat) * km_per_deg_lat

    def approx_tile_count(self, tile_px: int = 512, resolution_m: float = 10.0) -> int:
        """Number of tiles needed to cover the AOI. Sizes the compute budget.

        At 10 m with 512 px tiles, one tile is ~5.12 x 5.12 km = ~26 km2.
        """
        tile_km2 = (tile_px * resolution_m / 1000.0) ** 2
        return max(1, math.ceil(self.approx_area_km2 / tile_km2))

    @property
    def label_sources_declared(self) -> bool:
        """Whether a ground-truth source is *named* for this AOI.

        Declared, not obtained. `configs/aoi.yaml` names UNOSAT products for the
        Nepal AOI and Sen1Floods11 for the Indian ones; only the latter has been
        downloaded. A True here means supervised work is conceivable, not that
        anything is on disk -- which is why the name says "declared" and why the
        served payload does not call it `hasLabels`.
        """
        return bool(self.label_sources)

    @property
    def verified_events(self) -> list[HazardEvent]:
        """Events whose sensor, date and reference product were actually checked.

        The interface may only offer historical analysis for these. An event in
        ``events`` with ``verified: false`` is a research lead, not a result.
        """
        return [e for e in self.events if e.verified]

    def hazard_coverage(self) -> dict[str, bool]:
        """Which of this AOI's declared hazards have a verified event behind them.

        Drives the honest empty state: a hazard listed with no verified event
        renders as "no analysis" with the reason, never as a zero.
        """
        covered = {e.hazard for e in self.verified_events}
        return {hazard: hazard in covered for hazard in self.primary_hazards}


class AOIRegistry(BaseModel):
    """All configured study areas."""

    model_config = ConfigDict(frozen=True)

    version: str = "0.1.0"
    aois: list[AOI]

    def get(self, aoi_id: str) -> AOI:
        for aoi in self.aois:
            if aoi.id == aoi_id:
                return aoi
        known = ", ".join(a.id for a in self.aois)
        raise ValidationError(f"unknown AOI {aoi_id!r}; known AOIs: {known}")

    @property
    def selected(self) -> list[AOI]:
        """AOIs confirmed for use, as opposed to candidates under evaluation."""
        return [a for a in self.aois if a.status == "selected"]

    def __len__(self) -> int:
        return len(self.aois)


def load_aoi_registry(path: Path | None = None) -> AOIRegistry:
    """Load and validate ``configs/aoi.yaml``."""
    from satai.config import get_settings

    resolved = path or (get_settings().configs_dir / "aoi.yaml")
    if not resolved.exists():
        raise ValidationError(f"AOI config not found at {resolved}")
    raw: dict[str, Any] = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    return AOIRegistry.model_validate(raw)
