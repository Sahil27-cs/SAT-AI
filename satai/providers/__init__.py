"""Earth-observation data providers.

Each external data source sits behind :class:`~satai.providers.base.Provider`,
so that a failure in one becomes a documented fallback rather than a broken
pipeline (requirement 39).

Roles, per ADR-002:

- **Earth Engine** — primary processing plane. Server-side reduction; only
  analysis-ready arrays come down.
- **CDSE** — secondary/verification plane, and the authoritative catalogue.
  **Search needs no credentials**, which is what makes the Phase 2 AOI
  measurement runnable before any account exists.
- **FIRMS** — active fire *observations*, from Phase 16.
"""

from __future__ import annotations

from satai.providers.base import (
    Provider,
    ProviderStatus,
    SceneRef,
    SearchQuery,
    SearchResult,
)
from satai.providers.cdse import S1_IW_GRD, S2_L2A, CDSEProvider
from satai.providers.firms import FIRMS_CAVEATS, FireDetection, FIRMSProvider
from satai.providers.gee import GEE_COLLECTIONS, EarthEngineProvider
from satai.providers.manifest import Manifest, ManifestEntry

__all__ = [
    "FIRMS_CAVEATS",
    "GEE_COLLECTIONS",
    "S1_IW_GRD",
    "S2_L2A",
    "CDSEProvider",
    "EarthEngineProvider",
    "FIRMSProvider",
    "FireDetection",
    "Manifest",
    "ManifestEntry",
    "Provider",
    "ProviderStatus",
    "SceneRef",
    "SearchQuery",
    "SearchResult",
    "available_providers",
]


def available_providers() -> dict[str, dict[str, object]]:
    """Status of every provider, for ``/health`` and the diagnostic script.

    Reports capability honestly: a provider that is unconfigured is reported as
    unconfigured, never silently skipped. The user should be able to see why an
    answer is unavailable.
    """
    return {
        provider.name: provider.describe()
        for provider in (CDSEProvider(), EarthEngineProvider(), FIRMSProvider())
    }
