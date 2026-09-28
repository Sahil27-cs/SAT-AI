"""Microsoft Planetary Computer: analysis-ready Sentinel-1, with no account.

This provider exists because the project spent a long time recording "Earth
Engine / CDSE credentials" as the blocker on every acquisition, and that was
only true of *those two routes*. The Planetary Computer serves the same
Copernicus data through an anonymous STAC API, and its `sentinel-1-rtc`
collection is already radiometrically terrain corrected — which is the
processing step that would otherwise need a DEM and an SNAP graph.

What makes it usable here specifically:

* **Search needs nothing.** The STAC endpoint is open.
* **Reads need only a short-lived token** that the SAS endpoint issues to
  anonymous callers, per collection. No account, no key, nothing to leak.
* **The assets are Cloud-Optimized GeoTIFFs**, so an AOI is a windowed range
  read rather than a 1.9 GB download.

The radiometry caveat, stated here because it decides whether a result means
anything
-----------------------------------------------------------------------------
`sentinel-1-rtc` is **gamma0**, terrain-flattened. Sen1Floods11's chips — what
the flood model was trained on — are Earth Engine `COPERNICUS/S1_GRD`, which is
**sigma0** with ellipsoid correction. The two differ by the local incidence
angle term, and over terrain that difference is not small.

So this provider does not pretend the two are interchangeable. It records which
one a scene is (`radiometry` on the SceneRef), and the inference path runs the
project's own Track A/B distribution gate (`satai.ml.distribution_gate`) before
reporting anything. That gate exists for exactly this question, and its verdict
belongs in the artifact next to the prediction rather than in a footnote.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any

from satai.errors import ProviderError, ValidationError
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

__all__ = [
    "PC_COLLECTIONS",
    "RADIOMETRY",
    "PlanetaryComputerProvider",
    "sign_href",
]

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token"

#: Collections this provider searches, and what each one actually contains.
PC_COLLECTIONS: tuple[str, ...] = ("sentinel-1-rtc", "sentinel-1-grd", "sentinel-2-l2a")

#: The radiometric convention per collection. Carried into provenance because a
#: backscatter number without it cannot be compared with any other backscatter
#: number.
RADIOMETRY: dict[str, str] = {
    "sentinel-1-rtc": "gamma0_rtc_linear",
    "sentinel-1-grd": "dn_uncalibrated",
    "sentinel-2-l2a": "boa_reflectance",
}


#: Transient failures this service actually produces. A dropped connection part
#: way through a token request is common enough that not retrying makes the
#: pipeline look broken when the network merely hiccuped.
_TRANSIENT = (urllib.error.URLError, TimeoutError, ConnectionError, OSError)


def _request(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
    retries: int = 4,
    backoff: float = 1.5,
) -> bytes:
    """GET or POST with bounded exponential backoff.

    A 4xx is returned to the caller immediately: retrying a refusal wastes time
    and hides the reason. Only transport failures and 5xx are retried.
    """
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"User-Agent": "SAT-AI/0.5 (research)"}
    if data is not None:
        headers["Content-Type"] = "application/json"

    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, data=data, headers=headers)  # noqa: S310 - fixed https endpoint, not user input
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                body: bytes = response.read()
                return body
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise ProviderError(
                    f"Planetary Computer returned HTTP {exc.code} for {url.split('?')[0]}",
                    provider="planetary_computer",
                    status=str(exc.code),
                ) from exc
            last = exc
        except _TRANSIENT as exc:
            last = exc
        if attempt < retries - 1:
            delay = backoff**attempt
            log.warning(
                "planetary computer request failed, retrying",
                extra={"attempt": attempt + 1, "delay_s": delay, "error": str(last)[:120]},
            )
            time.sleep(delay)

    raise ProviderError(
        f"Planetary Computer unreachable after {retries} attempts: {last}",
        provider="planetary_computer",
    )


def _post_json(url: str, payload: dict[str, Any], *, timeout: float = 60.0) -> Any:
    try:
        return json.loads(_request(url, payload=payload, timeout=timeout))
    except ValueError as exc:
        raise ProviderError(
            f"Planetary Computer STAC returned unparseable JSON: {exc}",
            provider="planetary_computer",
        ) from exc


def sign_href(href: str, collection: str, *, timeout: float = 30.0) -> str:
    """Attach a short-lived anonymous SAS token to an asset URL.

    The token is issued per collection and expires within the hour, which is why
    nothing caches it to disk: a stale token in a manifest is a confusing 403
    somewhere far from here. Callers sign at the moment of reading.
    """
    try:
        token = json.loads(_request(f"{SAS_URL}/{collection}", timeout=timeout))["token"]
    except (ValueError, KeyError) as exc:
        raise ProviderError(
            f"could not obtain an anonymous SAS token for {collection!r}: {exc}",
            provider="planetary_computer",
        ) from exc
    separator = "&" if "?" in href else "?"
    return f"{href}{separator}{token}"


class PlanetaryComputerProvider(Provider):
    """Anonymous STAC access to Copernicus archives.

    Status is UNAVAILABLE rather than UNCONFIGURED when the endpoint cannot be
    reached: there is nothing to configure, so "not configured" would send a
    reader looking for a credential that does not exist.
    """

    name = "planetary_computer"
    collections = PC_COLLECTIONS
    quota_note = (
        "Anonymous, rate-limited, no account. Asset reads need a SAS token that "
        "the same service issues to anonymous callers and that expires within "
        "the hour."
    )

    def __init__(self, fetch: FetchJson | None = None) -> None:
        super().__init__(fetch)

    def status(self) -> ProviderStatus:
        try:
            _post_json(STAC_URL, {"collections": ["sentinel-1-rtc"], "limit": 1}, timeout=20.0)
        except ProviderError:
            return ProviderStatus.UNAVAILABLE
        return ProviderStatus.READY

    def search(self, query: SearchQuery) -> SearchResult:
        self.check_collection(query.collection)

        payload: dict[str, Any] = {
            "collections": [query.collection],
            "bbox": list(query.bbox),
            "datetime": f"{query.start.isoformat()}T00:00:00Z/{query.end.isoformat()}T23:59:59Z",
            "limit": min(query.limit, 250),
        }
        if query.max_cloud_cover is not None and query.collection.startswith("sentinel-2"):
            payload["query"] = {"eo:cloud_cover": {"lt": query.max_cloud_cover}}

        document = _post_json(STAC_URL, payload)
        features = document.get("features", [])
        if not isinstance(features, list):
            raise ProviderError(
                "Planetary Computer returned a STAC document with no feature list",
                provider=self.name,
            )

        return SearchResult(
            provider=self.name,
            query=query,
            scenes=[self._to_scene(f, query.collection) for f in features],
            truncated=len(features) >= payload["limit"],
        )

    def _to_scene(self, feature: dict[str, Any], collection: str) -> SceneRef:
        properties = feature.get("properties", {})
        acquired_at = None
        if properties.get("datetime"):
            acquired_at = datetime.fromisoformat(
                properties["datetime"].replace("Z", "+00:00")
            ).astimezone(UTC)

        bbox = feature.get("bbox")
        return SceneRef(
            scene_id=feature["id"],
            collection=collection,
            provider=self.name,
            acquired_at=acquired_at,
            bbox=tuple(bbox[:4]) if bbox and len(bbox) >= 4 else None,
            cloud_cover=properties.get("eo:cloud_cover"),
            platform=properties.get("platform"),
            orbit_direction=(properties.get("sat:orbit_state") or "").upper() or None,
            relative_orbit=properties.get("sat:relative_orbit"),
            access_url=(feature.get("assets", {}).get("vv", {}) or {}).get("href"),
            properties={
                # Everything a reader needs to judge whether this scene is
                # comparable with another one, and nothing decorative.
                "radiometry": RADIOMETRY.get(collection, "unknown"),
                "polarizations": properties.get("sar:polarizations"),
                "instrument_mode": properties.get("sar:instrument_mode"),
                "product_type": properties.get("sar:product_type"),
                "epsg": properties.get("proj:epsg"),
                "shape": properties.get("proj:shape"),
                "assets": {
                    name: asset.get("href")
                    for name, asset in (feature.get("assets") or {}).items()
                    if name in {"vv", "vh", "hh", "hv"}
                },
            },
        )

    def asset_href(self, scene: SceneRef, band: str, *, signed: bool = True) -> str:
        """The URL for one band of a scene, ready for a windowed COG read."""
        assets = scene.properties.get("assets") or {}
        href = assets.get(band)
        if not href:
            raise ValidationError(
                f"scene {scene.scene_id} has no {band!r} asset; it has: "
                f"{', '.join(sorted(assets)) or 'none'}"
            )
        return sign_href(href, scene.collection) if signed else href
