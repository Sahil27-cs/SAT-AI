"""Copernicus Data Space Ecosystem provider (STAC catalogue).

**This provider needs no credentials to search.** The CDSE STAC catalogue is
publicly readable; authentication is required only to *download* products. That
distinction is what makes the Phase 2 AOI decision runnable immediately, before
any account exists: scene counts, measured revisit intervals and cloud
statistics all come from catalogue metadata.

Free-tier limits for authenticated download, as documented on 2026-09-22:
10,000 Sentinel Hub processing units/month, 10,000 SH requests/month, 12 TB per
rolling 30 days, then throttling to 1 MB/s.

Role in the architecture (ADR-002): secondary / verification plane. Earth
Engine is primary for processing; CDSE is used for catalogue truth, for
full-resolution scenes needing custom SAR processing, and as the documented
fallback when GEE is unavailable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

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

__all__ = ["CDSE_STAC_URL", "S1_IW_GRD", "S2_L2A", "CDSEProvider"]

CDSE_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"

#: Product types. Sentinel-1 IW GRDH is the standard flood-mapping product;
#: Sentinel-2 L2A is surface reflectance with a scene-classification mask.
S1_IW_GRD = "IW_GRDH_1S"
S2_L2A = "S2MSI2A"

#: Page size. CDSE caps a single page well below the totals we need, so the
#: provider always paginates and reports truncation explicitly rather than
#: silently returning a partial count.
_PAGE_SIZE = 100
_MAX_PAGES = 100


class CDSEProvider(Provider):
    """Searches the CDSE STAC catalogue. No credentials needed for search."""

    name = "cdse"
    collections = ("SENTINEL-1", "SENTINEL-2", "SENTINEL-3", "SENTINEL-5P")
    quota_note = (
        "Search: free, unauthenticated. Download (authenticated): 10,000 SH "
        "processing units/month, 12 TB per rolling 30 days (as of 2026-09-22)."
    )

    def __init__(self, fetch: FetchJson | None = None, base_url: str = CDSE_STAC_URL) -> None:
        super().__init__(fetch)
        self.base_url = base_url.rstrip("/")

    def status(self) -> ProviderStatus:
        """Always READY: catalogue search requires no configuration.

        Deliberately does not probe the network. A status check that makes an
        HTTP call turns every health check into a dependency on an external
        service; reachability failures surface as ``ProviderError`` at the point
        of actual use, where the error message can be specific.
        """
        return ProviderStatus.READY

    def search(self, query: SearchQuery) -> SearchResult:
        """Search the catalogue, paginating until exhausted or capped."""
        self.check_collection(query.collection)

        params: dict[str, Any] = {
            "collections": query.collection,
            "bbox": ",".join(str(v) for v in query.bbox),
            "datetime": f"{query.start.isoformat()}T00:00:00Z/{query.end.isoformat()}T23:59:59Z",
            "limit": min(_PAGE_SIZE, query.limit),
        }

        features: list[dict[str, Any]] = []
        url: str = f"{self.base_url}/search"
        page_params: dict[str, Any] | None = params
        truncated = False

        for page in range(_MAX_PAGES):
            payload = self._fetch(url, page_params, None)
            if not isinstance(payload, dict):
                raise ProviderError("CDSE STAC returned a non-object response", provider=self.name)

            batch = payload.get("features") or []
            features.extend(batch)
            log.debug(
                "cdse page fetched",
                extra={"page": page, "batch": len(batch), "total": len(features)},
            )

            if len(features) >= query.limit:
                features = features[: query.limit]
                truncated = True
                break

            next_url = _next_link(payload)
            if not next_url or not batch:
                break
            url, page_params = next_url, None
        else:
            truncated = True

        scenes = [
            scene
            for feature in features
            if (scene := self._to_scene(feature, query.collection)) is not None
            and _passes_filters(scene, query)
        ]

        log.info(
            "cdse search complete",
            extra={
                "collection": query.collection,
                "raw_features": len(features),
                "after_filter": len(scenes),
                "truncated": truncated,
            },
        )
        return SearchResult(provider=self.name, query=query, scenes=scenes, truncated=truncated)

    def _to_scene(self, feature: dict[str, Any], collection: str) -> SceneRef | None:
        """Convert a STAC item. Returns ``None`` for an unparseable item.

        Skipping a malformed item is preferred over failing the whole search:
        one bad record in a catalogue of thousands should not cost the user the
        query. The count of skipped items is logged.
        """
        try:
            props: dict[str, Any] = feature.get("properties") or {}
            scene_id = feature.get("id") or props.get("identifier")
            if not scene_id:
                return None

            return SceneRef(
                scene_id=str(scene_id),
                collection=collection,
                provider=self.name,
                acquired_at=_parse_dt(
                    props.get("datetime") or props.get("startTimeFromAscendingNode")
                ),
                bbox=_bbox_of(feature),
                cloud_cover=_as_float(props.get("cloudCover", props.get("eo:cloud_cover"))),
                platform=props.get("platformShortName") or props.get("platform"),
                orbit_direction=_upper(props.get("orbitDirection") or props.get("sat:orbit_state")),
                relative_orbit=_as_int(
                    props.get("relativeOrbitNumber") or props.get("sat:relative_orbit")
                ),
                access_url=_self_link(feature),
                properties={
                    "productType": props.get("productType"),
                    "processingLevel": props.get("processingLevel"),
                    "polarisation": props.get("polarisationChannels"),
                },
            )
        except (TypeError, ValueError) as exc:
            log.debug("skipping unparseable STAC item", extra={"error": str(exc)})
            return None


def _passes_filters(scene: SceneRef, query: SearchQuery) -> bool:
    """Post-filter on product type and cloud cover.

    Filtering client-side rather than through STAC's query extension, whose
    support on CDSE has varied. Costs some bandwidth, gains reliability.
    """
    wanted_type = query.extra.get("product_type")
    if wanted_type and scene.properties.get("productType") != wanted_type:
        return False
    if query.max_cloud_cover is not None and scene.cloud_cover is not None:
        return scene.cloud_cover <= query.max_cloud_cover
    return True


def _next_link(payload: dict[str, Any]) -> str | None:
    for link in payload.get("links") or []:
        if isinstance(link, dict) and link.get("rel") == "next" and link.get("href"):
            return str(link["href"])
    return None


def _self_link(feature: dict[str, Any]) -> str | None:
    for link in feature.get("links") or []:
        if isinstance(link, dict) and link.get("rel") == "self" and link.get("href"):
            return str(link["href"])
    return None


def _bbox_of(feature: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = feature.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return None


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if 0.0 <= result <= 100.0 else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _upper(value: Any) -> str | None:
    return value.upper() if isinstance(value, str) else None
