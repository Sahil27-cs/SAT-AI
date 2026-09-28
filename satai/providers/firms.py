"""NASA FIRMS provider -- active fire detections.

**These are observations, not predictions**, and SAT-AI must never present them
otherwise. FIRMS reports thermal anomalies detected by VIIRS and MODIS at
satellite overpass times. A detection means "this pixel was hot when the
satellite looked"; the absence of a detection means only that nothing was
detected -- it may have been cloudy, or the fire may have started after the
overpass and burnt out before the next one.

Accordingly, every envelope built from this provider carries
``SourceKind.OBSERVATION`` and the caveats defined below.

Free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/
Latency: approximately 3 hours (near-real-time).

Used from Phase 16 (wildfire). Implemented now because it is small, and because
it is the case that proves the provider abstraction generalises beyond
catalogue search -- FIRMS returns CSV rows, not STAC items.
"""

from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
from datetime import UTC, date, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from satai.config import get_settings
from satai.errors import ProviderError, ValidationError
from satai.logging import get_logger
from satai.providers.base import Provider, ProviderStatus, SearchQuery, SearchResult

log = get_logger(__name__)

__all__ = [
    "AREA_ARCHIVE_URL",
    "FIRMS_CAVEATS",
    "FIRMS_SOURCES",
    "OPEN_PRODUCTS",
    "OPEN_WINDOWS",
    "FIRMSProvider",
    "FireDetection",
    "fetch_open_detections",
    "open_archive_url",
]

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

#: Sensors. VIIRS at 375 m resolves substantially smaller fires than MODIS at
#: 1 km and is the default; MODIS is retained for the long historical record,
#: which VIIRS does not have.
FIRMS_SOURCES: dict[str, str] = {
    "VIIRS_SNPP_NRT": "VIIRS S-NPP, 375 m, near-real-time",
    "VIIRS_NOAA20_NRT": "VIIRS NOAA-20, 375 m, near-real-time",
    "VIIRS_NOAA21_NRT": "VIIRS NOAA-21, 375 m, near-real-time",
    "MODIS_NRT": "MODIS Terra+Aqua, 1 km, near-real-time",
    "MODIS_SP": "MODIS standard processing, 1 km, historical archive",
}

#: Attached to every envelope built from FIRMS data. Machine-generated honesty:
#: these are the things a user would otherwise have to know to read the number
#: correctly, and the third one is specific to the Indian context.
FIRMS_CAVEATS: tuple[str, ...] = (
    "Active fire detections are satellite observations at overpass time, not "
    "predictions, and not a complete census of fires.",
    "Fires beneath cloud, fires smaller than the sensor footprint, and fires "
    "that start and end between overpasses are not detected.",
    "In India, a large share of detections during October-November are "
    "agricultural residue burning rather than forest fire. Stratify by land "
    "cover before interpreting counts as wildfire activity.",
)

#: FIRMS limits a single request to 10 days.
_MAX_DAY_RANGE = 10


class FireDetection(BaseModel):
    """One thermal anomaly."""

    model_config = ConfigDict(frozen=True)

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    acquired_at: datetime
    brightness_k: float | None = Field(default=None, description="Brightness temperature (K).")
    frp_mw: float | None = Field(default=None, description="Fire radiative power (MW).")
    confidence: str | None = Field(
        default=None,
        description=(
            "VIIRS: 'l'/'n'/'h' (low/nominal/high). MODIS: 0-100. Low-confidence "
            "detections include sun glint and industrial heat sources and should "
            "normally be filtered out before analysis."
        ),
    )
    daynight: str | None = None
    satellite: str | None = None
    source: str = Field(description="FIRMS source key, e.g. 'VIIRS_SNPP_NRT'.")

    @property
    def is_high_confidence(self) -> bool:
        """True for VIIRS 'h'/'n' or MODIS confidence >= 50."""
        if self.confidence is None:
            return False
        value = self.confidence.strip().lower()
        if value in {"h", "n"}:
            return True
        try:
            return float(value) >= 50.0
        except ValueError:
            return False


class FIRMSProvider(Provider):
    """NASA FIRMS active fire detections. Returns points, not scenes."""

    name = "firms"
    collections = tuple(FIRMS_SOURCES)
    quota_note = "Free with a MAP_KEY. Max 10-day range per request; be polite with rates."

    def status(self) -> ProviderStatus:
        return (
            ProviderStatus.READY
            if get_settings().nasa.firms_configured
            else ProviderStatus.UNCONFIGURED
        )

    def search(self, query: SearchQuery) -> SearchResult:
        """Not applicable: FIRMS returns detections, not scenes.

        Use :meth:`fetch_detections`. Raising rather than faking a
        ``SearchResult`` keeps the observation/scene distinction honest -- a
        fire detection is not a satellite granule and should not masquerade as
        one in provenance records.
        """
        raise ValidationError(
            "FIRMS returns point detections, not scenes; call fetch_detections() instead"
        )

    def fetch_detections(
        self,
        bbox: tuple[float, float, float, float],
        start: date,
        end: date,
        source: str = "VIIRS_SNPP_NRT",
        *,
        high_confidence_only: bool = True,
    ) -> list[FireDetection]:
        """Fetch detections, chunking the window to respect the 10-day limit."""
        self.require_ready()
        if source not in FIRMS_SOURCES:
            raise ValidationError(
                f"unknown FIRMS source {source!r}; known: {', '.join(FIRMS_SOURCES)}"
            )
        if start > end:
            raise ValidationError(f"start {start} is after end {end}")

        settings = get_settings()
        assert settings.nasa.firms_map_key is not None  # noqa: S101 - guarded by require_ready
        map_key = settings.nasa.firms_map_key.get_secret_value()

        detections: list[FireDetection] = []
        window_start = start
        while window_start <= end:
            span = min(_MAX_DAY_RANGE, (end - window_start).days + 1)
            area = ",".join(str(v) for v in bbox)
            url = f"{FIRMS_BASE_URL}/{map_key}/{source}/{area}/{span}/{window_start.isoformat()}"
            detections.extend(self._fetch_csv(url, source))
            window_start += timedelta(days=span)

        if high_confidence_only:
            before = len(detections)
            detections = [d for d in detections if d.is_high_confidence]
            log.info(
                "filtered low-confidence detections",
                extra={"before": before, "after": len(detections)},
            )
        return detections

    def _fetch_csv(self, url: str, source: str) -> list[FireDetection]:
        """FIRMS returns CSV, so this bypasses the JSON fetch layer."""
        import urllib.request

        try:
            request = urllib.request.Request(  # noqa: S310
                url, headers={"User-Agent": "SAT-AI/0.1"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                text = response.read().decode("utf-8")
        except Exception as exc:
            raise ProviderError(f"FIRMS request failed: {exc}", provider=self.name) from exc

        if text.lstrip().lower().startswith(("invalid", "error")):
            raise ProviderError(f"FIRMS rejected the request: {text[:200]}", provider=self.name)

        rows = list(csv.DictReader(io.StringIO(text)))
        return [d for row in rows if (d := _parse_row(row, source)) is not None]


def _parse_row(row: dict[str, Any], source: str) -> FireDetection | None:
    try:
        acq_date = str(row["acq_date"])
        acq_time = str(row.get("acq_time", "0000")).zfill(4)
        return FireDetection(
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            acquired_at=datetime.strptime(f"{acq_date} {acq_time}", "%Y-%m-%d %H%M").replace(
                tzinfo=UTC
            ),
            brightness_k=_opt_float(row.get("bright_ti4") or row.get("brightness")),
            frp_mw=_opt_float(row.get("frp")),
            confidence=row.get("confidence"),
            daynight=row.get("daynight"),
            satellite=row.get("satellite"),
            source=source,
        )
    except (KeyError, ValueError, TypeError) as exc:
        log.debug("skipping unparseable FIRMS row", extra={"error": str(exc)})
        return None


def _opt_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The open regional archive
# ---------------------------------------------------------------------------
#
# The FIRMS *API* needs a free MAP_KEY, which needs an account. The regional
# near-real-time files do not: NASA publishes them as plain CSV over HTTPS, and
# they are the same detections the API serves. That distinction is what lets
# this project produce a real wildfire observation at all.
#
# The trade-off is stated rather than hidden: these files cover only a rolling
# recent window, so they support "what is burning now" and not "what burned
# during a named historical fire". A historical archive request does need the
# key, and is recorded as a blocker rather than approximated with recent data.

AREA_ARCHIVE_URL = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/{product}/csv/{prefix}_{region}_{window}.csv"

#: Products available without a key, and the filename prefix each one uses.
OPEN_PRODUCTS: dict[str, str] = {
    "noaa-20-viirs-c2": "J1_VIIRS_C2",
    "noaa-21-viirs-c2": "J2_VIIRS_C2",
    "suomi-npp-viirs-c2": "SUOMI_VIIRS_C2",
    "modis-c6.1": "MODIS_C6_1",
}

#: Rolling windows NASA publishes. Nothing longer is available without the key.
OPEN_WINDOWS: tuple[str, ...] = ("24h", "48h", "7d")


def open_archive_url(product: str, region: str = "South_Asia", window: str = "7d") -> str:
    if product not in OPEN_PRODUCTS:
        raise ValidationError(
            f"unknown FIRMS product {product!r}; open products: {', '.join(OPEN_PRODUCTS)}"
        )
    if window not in OPEN_WINDOWS:
        raise ValidationError(
            f"{window!r} is not published without a MAP_KEY; available: {', '.join(OPEN_WINDOWS)}"
        )
    return AREA_ARCHIVE_URL.format(
        product=product, prefix=OPEN_PRODUCTS[product], region=region, window=window
    )


def fetch_open_detections(
    bbox: tuple[float, float, float, float],
    *,
    product: str = "noaa-20-viirs-c2",
    region: str = "South_Asia",
    window: str = "7d",
    timeout: float = 120.0,
) -> tuple[list[FireDetection], dict[str, Any]]:
    """Active-fire detections inside `bbox`, from the keyless regional archive.

    Returns the detections and a provenance record. An empty list is a real
    answer -- "nothing was detected in this window" -- and emphatically not
    "nothing burned": the sensor sees a given pixel twice a day at best, cloud
    blocks the thermal band, and a fire below the detection limit is invisible.
    """
    url = open_archive_url(product, region, window)
    request = urllib.request.Request(url, headers={"User-Agent": "SAT-AI/0.5 (research)"})  # noqa: S310 - fixed https endpoint, not user input
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            text = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderError(f"could not fetch {url}: {exc}", provider="firms") from exc

    min_lon, min_lat, max_lon, max_lat = bbox
    detections: list[FireDetection] = []
    total = 0
    for row in csv.DictReader(io.StringIO(text)):
        total += 1
        parsed = _parse_row(row, source=product)
        if parsed is None:
            continue
        if min_lon <= parsed.longitude <= max_lon and min_lat <= parsed.latitude <= max_lat:
            detections.append(parsed)

    provenance = {
        "dataset": "NASA FIRMS active fire",
        "product": product,
        "region_file": region,
        "window": window,
        "source_kind": "observation",
        "provider": "nasa_firms",
        "url": url,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "rows_in_file": total,
        "rows_in_bbox": len(detections),
        "requires_key": False,
        "caveats": [
            *FIRMS_CAVEATS,
            "From the keyless rolling regional archive, which covers only a "
            "recent window. A named historical fire needs the archive API and a "
            "MAP_KEY, which this deployment does not have.",
        ],
    }
    return detections, provenance
