"""Google Earth Engine provider -- the primary data plane (ADR-002).

Earth Engine is primary because it moves the processing to Google's
infrastructure instead of a student laptop. Filtering, masking, index
computation and temporal compositing run server-side; only reduced,
analysis-ready arrays come down. On a machine with single-digit gigabytes of
RAM, this is the difference between a project that runs and one that does not.

Free noncommercial quota as documented on 2026-09-22: 150 EECU-hours/month
(Community tier), or 1,000 EECU-hours/month (Contributor tier -- free, billing
account needed only for identity verification; graduate students and
researchers qualify).

``earthengine-api`` is imported lazily so the core package still imports on a
machine where Earth Engine is not installed, which is what keeps CI light.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from satai.config import get_settings
from satai.errors import ProviderError
from satai.logging import get_logger
from satai.providers.base import (
    FetchJson,
    Provider,
    ProviderStatus,
    SceneRef,
    SearchQuery,
    SearchResult,
)

log = get_logger(__name__)

__all__ = ["GEE_COLLECTIONS", "EarthEngineProvider"]

#: Earth Engine collection identifiers used across SAT-AI, with the phase that
#: introduces each. Kept here so there is one place to check an asset id --
#: a mistyped collection id fails deep inside a server-side call with an
#: unhelpful message.
GEE_COLLECTIONS: dict[str, str] = {
    # Phase 2-5: flood
    "S1_GRD": "COPERNICUS/S1_GRD",
    "S2_SR": "COPERNICUS/S2_SR_HARMONIZED",
    "S2_CLOUD_PROB": "COPERNICUS/S2_CLOUD_PROBABILITY",
    "DEM_GLO30": "COPERNICUS/DEM/GLO30",
    "IMERG": "NASA/GPM_L3/IMERG_V07",
    "ERA5_LAND_HOURLY": "ECMWF/ERA5_LAND/HOURLY",
    "WORLDCOVER": "ESA/WorldCover/v200",
    "JRC_WATER": "JRC/GSW1_4/GlobalSurfaceWater",
    # Phase 8: exposure
    "WORLDPOP": "WorldPop/GP/100m/pop",
    "GHSL_BUILT": "JRC/GHSL/P2023A/GHS_BUILT_S",
    # Phase 16-17: wildfire, cyclone
    "MODIS_BURNED": "MODIS/061/MCD64A1",
    "VIIRS_THERMAL": "NOAA/VIIRS/001/VNP14A1",
    "IBTRACS": "NOAA/IBTrACS/v4",
}


class EarthEngineProvider(Provider):
    """Earth Engine catalogue access and server-side reduction.

    Only catalogue search is implemented in Phase 2. Server-side preprocessing
    (speckle filtering, terrain correction, index stacks) arrives in Phase 3,
    once the AOI has actually been chosen -- building a preprocessing pipeline
    before knowing the study area would be guessing at its requirements.
    """

    name = "gee"
    collections = tuple(GEE_COLLECTIONS.values())
    quota_note = (
        "Free for noncommercial use: 150 EECU-hours/month (Community) or 1,000 "
        "(Contributor, free, billing account for identity verification only). "
        "Exceeding the quota degrades performance rather than cutting access."
    )

    def __init__(self, fetch: FetchJson | None = None) -> None:
        super().__init__(fetch)
        self._initialised = False

    # -- lifecycle ---------------------------------------------------------

    def status(self) -> ProviderStatus:
        """READY only if the library is importable and a project is configured."""
        settings = get_settings()
        if not settings.gee.is_configured:
            return ProviderStatus.UNCONFIGURED
        try:
            import ee  # noqa: F401
        except ImportError:
            return ProviderStatus.UNAVAILABLE
        return ProviderStatus.READY

    def initialise(self) -> None:
        """Authenticate and initialise the Earth Engine client. Idempotent.

        Two credential paths: a service-account key for unattended batch jobs,
        and the interactive credentials written by ``earthengine authenticate``
        for development. The service-account path is preferred for anything
        scheduled, because interactive credentials expire without warning
        halfway through an overnight run.
        """
        if self._initialised:
            return

        settings = get_settings()
        if not settings.gee.is_configured:
            raise ProviderError(
                "GEE_PROJECT_ID is not set. Register a Cloud project at "
                "https://code.earthengine.google.com, then set GEE_PROJECT_ID "
                "in .env and run: earthengine authenticate",
                provider=self.name,
            )

        try:
            import ee
        except ImportError as exc:
            raise ProviderError(
                "earthengine-api is not installed. Install it with: pip install earthengine-api",
                provider=self.name,
            ) from exc

        try:
            key_path = settings.gee.service_account_key_path
            if settings.gee.service_account_email and key_path:
                credentials = ee.ServiceAccountCredentials(
                    settings.gee.service_account_email, str(key_path)
                )
                ee.Initialize(credentials, project=settings.gee.project_id)
                log.info("earth engine initialised (service account)")
            else:
                ee.Initialize(project=settings.gee.project_id)
                log.info("earth engine initialised (user credentials)")
        except Exception as exc:  # ee raises its own bare Exception subclasses
            raise ProviderError(
                f"Earth Engine initialisation failed: {exc}. If this mentions "
                f"credentials, run: earthengine authenticate",
                provider=self.name,
            ) from exc

        self._initialised = True

    # -- search ------------------------------------------------------------

    def search(self, query: SearchQuery) -> SearchResult:
        """List granules in an Earth Engine collection over the AOI and window.

        Note on what this costs: metadata listing is cheap in EECU terms, but
        it is not free. Prefer CDSE for pure catalogue counting (it needs no
        credentials and no quota) and reserve Earth Engine quota for the
        processing it is actually good at.
        """
        self.require_ready()
        self.initialise()
        self.check_collection(query.collection)

        import ee

        min_lon, min_lat, max_lon, max_lat = query.bbox
        region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

        collection = (
            ee.ImageCollection(query.collection)
            .filterBounds(region)
            .filterDate(query.start.isoformat(), query.end.isoformat())
        )
        if query.max_cloud_cover is not None:
            collection = collection.filter(
                ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", query.max_cloud_cover)
            )

        try:
            info = collection.limit(query.limit).getInfo()
        except Exception as exc:
            raise ProviderError(
                f"Earth Engine query failed for {query.collection}: {exc}",
                provider=self.name,
                collection=query.collection,
            ) from exc

        features: list[dict[str, Any]] = (info or {}).get("features", [])
        scenes = [s for f in features if (s := self._to_scene(f, query.collection))]

        return SearchResult(
            provider=self.name,
            query=query,
            scenes=scenes,
            truncated=len(features) >= query.limit,
        )

    def _to_scene(self, feature: dict[str, Any], collection: str) -> SceneRef | None:
        props: dict[str, Any] = feature.get("properties") or {}
        scene_id = feature.get("id")
        if not scene_id:
            return None

        acquired = None
        millis = props.get("system:time_start")
        if isinstance(millis, (int, float)):
            acquired = datetime.fromtimestamp(millis / 1000.0, tz=UTC)

        return SceneRef(
            scene_id=str(scene_id),
            collection=collection,
            provider=self.name,
            acquired_at=acquired,
            cloud_cover=_as_float(props.get("CLOUDY_PIXEL_PERCENTAGE")),
            platform=props.get("platform_number") or props.get("SPACECRAFT_NAME"),
            orbit_direction=props.get("orbitProperties_pass"),
            relative_orbit=_as_int(props.get("relativeOrbitNumber_start")),
            properties={
                "instrument": props.get("instrumentMode"),
                "polarisation": props.get("transmitterReceiverPolarisation"),
            },
        )


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
