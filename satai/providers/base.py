"""Provider abstraction for Earth-observation data sources.

Every external data source in SAT-AI sits behind this interface. The reason is
not abstraction for its own sake -- it is that Earth-observation APIs are
unreliable in specific, predictable ways (quota exhaustion, scheduled outages,
silent schema changes, rate limits), and the system must be able to fail over
between them and to *say* which one it used.

Three rules the interface enforces:

1. **A provider always reports its own status.** ``status()`` distinguishes
   "not configured" (no credential) from "unavailable" (configured but
   unreachable). Those require different messages to the user and different
   fallback behaviour.
2. **A search result names its scenes.** Not "we used Sentinel-1", but a list of
   granule identifiers with acquisition times. This is what feeds
   :class:`satai.provenance.DataSourceRef` and makes a result reproducible.
3. **The HTTP layer is injectable.** Providers take a ``fetch`` callable, so the
   test suite runs against recorded fixtures with no network. A test suite that
   depends on a satellite catalogue being up is a test suite that fails for
   reasons unrelated to the code.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, date, datetime
from enum import StrEnum
from itertools import pairwise
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from satai.errors import ProviderError, ValidationError
from satai.logging import get_logger
from satai.provenance import DataSourceRef

log = get_logger(__name__)

__all__ = [
    "FetchJson",
    "Provider",
    "ProviderStatus",
    "SceneRef",
    "SearchQuery",
    "SearchResult",
    "default_fetch_json",
]

#: Signature of the injectable HTTP layer: (url, params, headers) -> parsed JSON.
FetchJson = Callable[[str, dict[str, Any] | None, dict[str, str] | None], Any]


class ProviderStatus(StrEnum):
    """Whether a provider can be used right now, and if not, why not."""

    READY = "ready"
    """Configured (or needs no configuration) and expected to work."""

    UNCONFIGURED = "unconfigured"
    """Missing a credential. The user can fix this; tell them which one."""

    UNAVAILABLE = "unavailable"
    """Configured but unreachable: outage, network failure, quota exhausted."""


class SceneRef(BaseModel):
    """One satellite granule.

    Deliberately flat and provider-agnostic. Provider-specific fields live in
    ``properties`` rather than accreting onto the model, so that adding a
    provider never requires changing this class.
    """

    model_config = ConfigDict(frozen=True)

    scene_id: str = Field(description="Granule / product identifier.")
    collection: str = Field(description="Collection, e.g. 'SENTINEL-1'.")
    provider: str = Field(description="Access route, e.g. 'cdse'.")
    acquired_at: datetime | None = None
    bbox: tuple[float, float, float, float] | None = None

    cloud_cover: float | None = Field(default=None, ge=0.0, le=100.0)
    platform: str | None = Field(default=None, description="e.g. 'SENTINEL-1A'.")
    orbit_direction: str | None = Field(default=None, description="ASCENDING / DESCENDING.")
    relative_orbit: int | None = Field(
        default=None,
        description=(
            "Relative orbit number. Matters more than it looks: a pre/post SAR "
            "pair from different relative orbits has different incidence "
            "geometry, so their backscatter difference mixes flooding with "
            "viewing angle. Change detection should compare within one orbit."
        ),
    )
    access_url: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)

    def to_source_ref(self) -> DataSourceRef:
        """Convert to the provenance record carried alongside derived values."""
        return DataSourceRef(
            dataset=self.collection,
            provider=self.provider,
            scene_id=self.scene_id,
            acquired_at=self.acquired_at,
            access_url=self.access_url,
        )


class SearchQuery(BaseModel):
    """A catalogue query. Validated before it leaves the machine."""

    model_config = ConfigDict(frozen=True)

    bbox: tuple[float, float, float, float] = Field(
        description="(min_lon, min_lat, max_lon, max_lat) in EPSG:4326."
    )
    start: date
    end: date
    collection: str
    max_cloud_cover: float | None = Field(default=None, ge=0.0, le=100.0)
    limit: int = Field(default=1000, gt=0, le=10_000)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        min_lon, min_lat, max_lon, max_lat = v
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError(f"degenerate or inverted bbox: {v}")
        if not (min_lon >= -180 and max_lon <= 180 and min_lat >= -90 and max_lat <= 90):
            raise ValueError(f"bbox outside valid lon/lat range: {v}")
        return v

    @model_validator(mode="after")
    def _check_dates(self) -> SearchQuery:
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        return self

    @property
    def days(self) -> int:
        """Inclusive length of the query window, in days."""
        return (self.end - self.start).days + 1

    def cache_key(self) -> str:
        """Stable key for the on-disk cache.

        Preprocessing gets re-run dozens of times during development; caching
        catalogue responses is the difference between a two-second iteration and
        a two-minute one.
        """
        payload = self.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True)


class SearchResult(BaseModel):
    """The scenes a query found, plus enough context to reproduce it."""

    model_config = ConfigDict(frozen=True)

    provider: str
    query: SearchQuery
    scenes: list[SceneRef]
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    truncated: bool = Field(
        default=False,
        description="True when the result hit the limit, so counts are a floor.",
    )

    def __len__(self) -> int:
        return len(self.scenes)

    @property
    def acquisition_dates(self) -> list[date]:
        """Sorted unique acquisition dates."""
        return sorted({s.acquired_at.date() for s in self.scenes if s.acquired_at})

    def revisit_gaps_days(self) -> list[int]:
        """Gaps between consecutive acquisition dates.

        This is the **measured** revisit interval over the area, which is what
        should be reported -- not the nominal mission figure. Coverage over a
        given footprint depends on the observation scenario, not on the
        satellite's orbital period.
        """
        dates = self.acquisition_dates
        return [(b - a).days for a, b in pairwise(dates)]

    def usable(self, max_cloud: float = 20.0) -> list[SceneRef]:
        """Scenes below a cloud threshold.

        Scenes with no cloud metadata (all SAR) count as usable, since cloud
        cover is not a meaningful attribute of a radar acquisition.
        """
        return [s for s in self.scenes if s.cloud_cover is None or s.cloud_cover <= max_cloud]

    def to_source_refs(self) -> list[DataSourceRef]:
        return [s.to_source_ref() for s in self.scenes]


class Provider(ABC):
    """Base class for an Earth-observation data source."""

    #: Short stable identifier recorded in provenance, e.g. ``"cdse"``.
    name: str = "provider"

    #: Collections this provider can search.
    collections: tuple[str, ...] = ()

    #: Human-readable note on cost/quota, surfaced by ``describe()``.
    quota_note: str = ""

    def __init__(self, fetch: FetchJson | None = None) -> None:
        self._fetch: FetchJson = fetch or default_fetch_json

    @abstractmethod
    def status(self) -> ProviderStatus:
        """Whether this provider is usable right now."""

    @abstractmethod
    def search(self, query: SearchQuery) -> SearchResult:
        """Find scenes matching ``query``.

        Raises
        ------
        ProviderError
            The upstream service failed, refused, or returned something
            unparseable.
        ValidationError
            The query asks for a collection this provider does not serve.
        """

    def require_ready(self) -> None:
        """Raise a useful error if the provider is not usable."""
        status = self.status()
        if status is ProviderStatus.UNCONFIGURED:
            raise ProviderError(
                f"{self.name} is not configured; see .env.example for the credentials it needs",
                provider=self.name,
                status=status.value,
            )
        if status is ProviderStatus.UNAVAILABLE:
            raise ProviderError(
                f"{self.name} is configured but unreachable",
                provider=self.name,
                status=status.value,
            )

    def check_collection(self, collection: str) -> None:
        if self.collections and collection not in self.collections:
            raise ValidationError(
                f"{self.name} does not serve collection {collection!r}; "
                f"it serves: {', '.join(self.collections)}"
            )

    def describe(self) -> dict[str, Any]:
        """Self-description, used by ``/health`` and the diagnostic script."""
        return {
            "name": self.name,
            "status": self.status().value,
            "collections": list(self.collections),
            "quota_note": self.quota_note,
        }


def default_fetch_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    *,
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 2.0,
) -> Any:
    """Fetch JSON over HTTPS with bounded exponential backoff.

    Uses the standard library rather than ``requests``/``httpx`` so that the
    core package has no HTTP dependency and imports cleanly in a minimal
    environment (see ADR-004 on why CI installs only the light stack).

    Retries on 429 and 5xx, which is what EO catalogues return under load, and
    does not retry on 4xx, which means the query itself is wrong.
    """
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"

    request_headers = {"Accept": "application/json", "User-Agent": "SAT-AI/0.1"}
    if headers:
        request_headers.update(headers)

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=request_headers)  # noqa: S310
            if not url.lower().startswith("https://"):
                raise ValidationError(f"refusing non-HTTPS request to {url!r}")
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))

        except urllib.error.HTTPError as exc:
            last_error = exc
            retryable = exc.code == 429 or 500 <= exc.code < 600
            log.warning(
                "http error from provider",
                extra={
                    "url": url[:200],
                    "code": exc.code,
                    "retryable": retryable,
                    "attempt": attempt + 1,
                },
            )
            if not retryable:
                raise ProviderError(
                    f"HTTP {exc.code} from {urllib.parse.urlsplit(url).netloc}: {exc.reason}",
                    provider=urllib.parse.urlsplit(url).netloc,
                    status_code=exc.code,
                ) from exc

        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            log.warning(
                "provider request failed",
                extra={"url": url[:200], "error": str(exc), "attempt": attempt + 1},
            )

        if attempt < retries - 1:
            time.sleep(backoff**attempt)

    raise ProviderError(
        f"request failed after {retries} attempts: {last_error}",
        provider=urllib.parse.urlsplit(url).netloc,
    )
