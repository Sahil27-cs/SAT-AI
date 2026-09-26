"""Tests for the provider layer. All offline."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import pytest
from pydantic import ValidationError as PydanticValidationError

from satai.errors import ProviderError, ValidationError
from satai.providers.base import (
    Provider,
    ProviderStatus,
    SceneRef,
    SearchQuery,
    SearchResult,
)
from satai.providers.cdse import S1_IW_GRD, CDSEProvider
from satai.providers.firms import FIRMS_CAVEATS, FireDetection, FIRMSProvider

BIHAR_BBOX = (85.0, 25.2, 86.5, 26.2)


class TestSearchQuery:
    def _query(self, **kw: Any) -> SearchQuery:
        base: dict[str, Any] = {
            "bbox": BIHAR_BBOX,
            "start": date(2024, 6, 1),
            "end": date(2024, 10, 31),
            "collection": "SENTINEL-1",
        }
        base.update(kw)
        return SearchQuery(**base)

    def test_rejects_inverted_bbox(self) -> None:
        with pytest.raises(PydanticValidationError):
            self._query(bbox=(86.5, 25.2, 85.0, 26.2))

    def test_rejects_out_of_range_bbox(self) -> None:
        with pytest.raises(PydanticValidationError):
            self._query(bbox=(85.0, 25.2, 200.0, 26.2))

    def test_rejects_reversed_dates(self) -> None:
        with pytest.raises(PydanticValidationError):
            self._query(start=date(2024, 10, 31), end=date(2024, 6, 1))

    def test_day_count_is_inclusive(self) -> None:
        q = self._query(start=date(2024, 6, 1), end=date(2024, 6, 10))
        assert q.days == 10

    def test_cache_key_is_stable_and_discriminating(self) -> None:
        assert self._query().cache_key() == self._query().cache_key()
        assert self._query().cache_key() != self._query(collection="SENTINEL-2").cache_key()


class TestSearchResult:
    def _result(self, scenes: list[SceneRef]) -> SearchResult:
        return SearchResult(
            provider="cdse",
            query=SearchQuery(
                bbox=BIHAR_BBOX,
                start=date(2024, 6, 1),
                end=date(2024, 10, 31),
                collection="SENTINEL-1",
            ),
            scenes=scenes,
        )

    def _scene(self, day: int, **kw: Any) -> SceneRef:
        return SceneRef(
            scene_id=f"S1A_{day}",
            collection="SENTINEL-1",
            provider="cdse",
            acquired_at=datetime.fromisoformat(f"2024-06-{day:02d}T00:31:12+00:00"),
            **kw,
        )

    def test_measured_revisit_gaps(self) -> None:
        """Revisit is measured from acquisitions, not quoted from the mission spec.

        Coverage over a footprint follows the observation scenario, not the
        orbital period, so the nominal figure can be badly wrong locally.
        """
        result = self._result([self._scene(d) for d in (1, 13, 25)])
        assert result.revisit_gaps_days() == [12, 12]

    def test_duplicate_dates_collapse(self) -> None:
        """Two scenes on the same day are one acquisition date, not a 0-day revisit."""
        result = self._result([self._scene(1), self._scene(1), self._scene(13)])
        assert result.acquisition_dates == [date(2024, 6, 1), date(2024, 6, 13)]
        assert result.revisit_gaps_days() == [12]

    def test_gaps_empty_for_single_scene(self) -> None:
        assert self._result([self._scene(1)]).revisit_gaps_days() == []

    def test_sar_scenes_count_as_usable(self) -> None:
        """Cloud cover is not a meaningful attribute of a radar acquisition."""
        result = self._result([self._scene(1), self._scene(13)])
        assert len(result.usable(max_cloud=0.0)) == 2

    def test_cloud_filter_applies_to_optical(self) -> None:
        scenes = [self._scene(1, cloud_cover=5.0), self._scene(13, cloud_cover=95.0)]
        assert len(self._result(scenes).usable(max_cloud=20.0)) == 1

    def test_scene_converts_to_provenance_ref(self) -> None:
        """'We used Sentinel-1' is not a method; a scene id is."""
        ref = self._scene(1).to_source_ref()
        assert ref.scene_id == "S1A_1"
        assert ref.provider == "cdse"
        assert ref.dataset == "SENTINEL-1"


class TestCDSEProvider:
    def test_search_needs_no_credentials(self) -> None:
        """The whole Phase 2 AOI decision depends on this being true."""
        assert CDSEProvider().status() is ProviderStatus.READY

    def test_status_does_not_touch_the_network(self) -> None:
        def exploding_fetch(*_: Any, **__: Any) -> Any:
            raise AssertionError("status() must not make a network call")

        assert CDSEProvider(fetch=exploding_fetch).status() is ProviderStatus.READY

    def test_parses_scenes(
        self, s1_stac_page: dict[str, Any], fake_fetch: Callable[..., Any]
    ) -> None:
        provider = CDSEProvider(fetch=fake_fetch([s1_stac_page]))
        result = provider.search(
            SearchQuery(
                bbox=BIHAR_BBOX,
                start=date(2024, 6, 1),
                end=date(2024, 10, 31),
                collection="SENTINEL-1",
            )
        )
        assert len(result) == 6
        first = result.scenes[0]
        assert first.platform == "SENTINEL-1A"
        assert first.orbit_direction == "DESCENDING"
        assert first.relative_orbit == 63
        assert first.acquired_at is not None

    def test_product_type_filter(
        self, s1_stac_page: dict[str, Any], fake_fetch: Callable[..., Any]
    ) -> None:
        s1_stac_page["features"][0]["properties"]["productType"] = "IW_SLC__1S"
        provider = CDSEProvider(fetch=fake_fetch([s1_stac_page]))
        result = provider.search(
            SearchQuery(
                bbox=BIHAR_BBOX,
                start=date(2024, 6, 1),
                end=date(2024, 10, 31),
                collection="SENTINEL-1",
                extra={"product_type": S1_IW_GRD},
            )
        )
        assert len(result) == 5

    def test_cloud_filter(
        self, s2_stac_page: dict[str, Any], fake_fetch: Callable[..., Any]
    ) -> None:
        provider = CDSEProvider(fetch=fake_fetch([s2_stac_page]))
        result = provider.search(
            SearchQuery(
                bbox=BIHAR_BBOX,
                start=date(2024, 6, 1),
                end=date(2024, 6, 30),
                collection="SENTINEL-2",
                max_cloud_cover=20.0,
            )
        )
        assert len(result) == 2  # only the 12% and 4% scenes

    def test_unparseable_item_is_skipped_not_fatal(
        self, s1_stac_page: dict[str, Any], fake_fetch: Callable[..., Any]
    ) -> None:
        """One bad record should not cost the user the whole query."""
        s1_stac_page["features"].append({"properties": {}})  # no id
        provider = CDSEProvider(fetch=fake_fetch([s1_stac_page]))
        result = provider.search(
            SearchQuery(
                bbox=BIHAR_BBOX,
                start=date(2024, 6, 1),
                end=date(2024, 10, 31),
                collection="SENTINEL-1",
            )
        )
        assert len(result) == 6

    def test_truncation_is_reported(
        self, s1_stac_page: dict[str, Any], fake_fetch: Callable[..., Any]
    ) -> None:
        """A capped result must say so: the count is then a lower bound.

        Silently returning a partial count would make the AOI comparison wrong
        in a way nobody would notice.
        """
        provider = CDSEProvider(fetch=fake_fetch([s1_stac_page]))
        result = provider.search(
            SearchQuery(
                bbox=BIHAR_BBOX,
                start=date(2024, 6, 1),
                end=date(2024, 10, 31),
                collection="SENTINEL-1",
                limit=3,
            )
        )
        assert result.truncated is True
        assert len(result) == 3

    def test_follows_pagination(self, fake_fetch: Callable[..., Any]) -> None:
        page1 = {
            "features": [
                {
                    "id": f"S1A_{i}",
                    "properties": {"datetime": f"2024-06-{i + 1:02d}T00:00:00Z"},
                }
                for i in range(3)
            ],
            "links": [{"rel": "next", "href": "https://example.invalid/search?page=2"}],
        }
        page2 = {
            "features": [
                {
                    "id": f"S1A_{i}",
                    "properties": {"datetime": f"2024-07-{i + 1:02d}T00:00:00Z"},
                }
                for i in range(3, 5)
            ],
            "links": [],
        }
        provider = CDSEProvider(fetch=fake_fetch([page1, page2]))
        result = provider.search(
            SearchQuery(
                bbox=BIHAR_BBOX,
                start=date(2024, 6, 1),
                end=date(2024, 10, 31),
                collection="SENTINEL-1",
            )
        )
        assert len(result) == 5

    def test_rejects_unknown_collection(self) -> None:
        with pytest.raises(ValidationError, match="does not serve"):
            CDSEProvider().search(
                SearchQuery(
                    bbox=BIHAR_BBOX,
                    start=date(2024, 6, 1),
                    end=date(2024, 6, 30),
                    collection="LANDSAT-9",
                )
            )

    def test_non_object_response_raises(self, fake_fetch: Callable[..., Any]) -> None:
        provider = CDSEProvider(fetch=fake_fetch([["not", "an", "object"]]))  # type: ignore[list-item]
        with pytest.raises(ProviderError):
            provider.search(
                SearchQuery(
                    bbox=BIHAR_BBOX,
                    start=date(2024, 6, 1),
                    end=date(2024, 6, 30),
                    collection="SENTINEL-1",
                )
            )


class TestFIRMS:
    def test_unconfigured_without_a_map_key(self) -> None:
        assert FIRMSProvider().status() is ProviderStatus.UNCONFIGURED

    def test_unconfigured_provider_raises_a_helpful_error(self) -> None:
        with pytest.raises(ProviderError, match="not configured"):
            FIRMSProvider().require_ready()

    def test_search_refuses_to_pretend_detections_are_scenes(self) -> None:
        """A fire detection is not a satellite granule and must not pose as one."""
        with pytest.raises(ValidationError, match="point detections"):
            FIRMSProvider().search(
                SearchQuery(
                    bbox=BIHAR_BBOX,
                    start=date(2024, 6, 1),
                    end=date(2024, 6, 30),
                    collection="VIIRS_SNPP_NRT",
                )
            )

    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [("h", True), ("n", True), ("l", False), ("80", True), ("30", False), (None, False)],
    )
    def test_confidence_classification(self, confidence: str | None, expected: bool) -> None:
        detection = FireDetection(
            latitude=25.6,
            longitude=85.1,
            acquired_at=datetime.fromisoformat("2024-06-01T05:00:00+00:00"),
            confidence=confidence,
            source="VIIRS_SNPP_NRT",
        )
        assert detection.is_high_confidence is expected

    def test_caveats_state_the_observation_boundary(self) -> None:
        """FIRMS data must never be presented as prediction."""
        joined = " ".join(FIRMS_CAVEATS).lower()
        assert "not predictions" in joined
        assert "residue burning" in joined  # the India-specific confound


def test_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        Provider()  # type: ignore[abstract]
