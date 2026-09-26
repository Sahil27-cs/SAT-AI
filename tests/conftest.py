"""Shared test fixtures.

Everything here is offline. The provider tests run against recorded catalogue
responses rather than the live CDSE API, because a test suite that depends on a
satellite catalogue being up is a test suite that fails for reasons unrelated to
the code — and it cannot run in CI, which has no credentials and, increasingly,
no outbound access to those hosts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


def _s1_feature(
    index: int,
    day: int,
    *,
    relative_orbit: int = 63,
    direction: str = "DESCENDING",
    product_type: str = "IW_GRDH_1S",
) -> dict[str, Any]:
    """One Sentinel-1 STAC item, shaped like a real CDSE response."""
    return {
        "id": f"S1A_IW_GRDH_1SDV_20240{day // 30 + 6}{day % 30 + 1:02d}T003112_{index:06d}",
        "bbox": [85.0, 25.2, 86.5, 26.2],
        "properties": {
            "datetime": f"2024-0{day // 30 + 6}-{day % 30 + 1:02d}T00:31:12.000Z",
            "productType": product_type,
            "platformShortName": "SENTINEL-1A",
            "orbitDirection": direction,
            "relativeOrbitNumber": relative_orbit,
            "polarisationChannels": "VV&VH",
        },
        "links": [{"rel": "self", "href": f"https://example.invalid/items/{index}"}],
    }


def _s2_feature(index: int, day: int, cloud: float) -> dict[str, Any]:
    return {
        "id": f"S2A_MSIL2A_2024060{day}T052641_{index:06d}",
        "bbox": [85.0, 25.2, 86.5, 26.2],
        "properties": {
            "datetime": f"2024-06-{day:02d}T05:26:41.000Z",
            "productType": "S2MSI2A",
            "platformShortName": "SENTINEL-2A",
            "cloudCover": cloud,
        },
        "links": [{"rel": "self", "href": f"https://example.invalid/items/{index}"}],
    }


@pytest.fixture
def s1_stac_page() -> dict[str, Any]:
    """A single-page Sentinel-1 STAC response: 6 scenes, 12-day cadence."""
    return {
        "type": "FeatureCollection",
        "features": [_s1_feature(i, day=i * 12) for i in range(6)],
        "links": [{"rel": "self", "href": "https://example.invalid/search"}],
    }


@pytest.fixture
def s2_stac_page() -> dict[str, Any]:
    """Sentinel-2 response with a realistic monsoon cloud distribution."""
    clouds = [95.0, 88.0, 12.0, 99.0, 73.0, 4.0, 100.0, 61.0]
    return {
        "type": "FeatureCollection",
        "features": [_s2_feature(i, day=i + 1, cloud=c) for i, c in enumerate(clouds)],
        "links": [],
    }


@pytest.fixture
def fake_fetch() -> Callable[[list[dict[str, Any]]], Any]:
    """Build a fetch callable that replays a list of recorded responses.

    Returns a factory so a test can define its own page sequence, which is how
    pagination and truncation get tested without a network.
    """

    def factory(pages: list[dict[str, Any]]) -> Any:
        calls: list[tuple[str, dict[str, Any] | None]] = []

        def fetch(
            url: str,
            params: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> Any:
            calls.append((url, params))
            index = min(len(calls) - 1, len(pages) - 1)
            return pages[index]

        fetch.calls = calls  # type: ignore[attr-defined]
        return fetch

    return factory
