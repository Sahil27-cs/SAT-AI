"""Tests for AOI definitions and CRS selection."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from satai.errors import ValidationError
from satai.geo.aoi import AOI, load_aoi_registry, utm_epsg_for


class TestUTMZone:
    @pytest.mark.parametrize(
        ("lon", "lat", "expected"),
        [
            (72.87, 19.07, "EPSG:32643"),  # Mumbai
            (85.14, 25.60, "EPSG:32645"),  # Patna, Bihar
            (91.75, 26.14, "EPSG:32646"),  # Guwahati, Assam
            (76.27, 9.93, "EPSG:32643"),  # Kochi, Kerala
            (-58.38, -34.60, "EPSG:32721"),  # southern hemisphere -> 327xx
        ],
    )
    def test_known_locations(self, lon: float, lat: float, expected: str) -> None:
        assert utm_epsg_for(lon, lat) == expected

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            utm_epsg_for(200.0, 0.0)
        with pytest.raises(ValidationError):
            utm_epsg_for(0.0, 95.0)


class TestAOI:
    def _aoi(self, **kw: object) -> AOI:
        base = {
            "id": "test_area",
            "name": "Test",
            "bbox": (85.0, 25.2, 86.5, 26.2),
            "selection_rationale": "A rationale long enough to satisfy the minimum length.",
        }
        base.update(kw)
        return AOI.model_validate(base)

    def test_rejects_inverted_bbox(self) -> None:
        with pytest.raises(PydanticValidationError):
            self._aoi(bbox=(86.5, 25.2, 85.0, 26.2))

    def test_rejects_out_of_range_bbox(self) -> None:
        with pytest.raises(PydanticValidationError):
            self._aoi(bbox=(85.0, 25.2, 200.0, 26.2))

    def test_requires_a_substantive_rationale(self) -> None:
        """Study areas are chosen on evidence; the model refuses to let that go
        unrecorded."""
        with pytest.raises(PydanticValidationError):
            self._aoi(selection_rationale="because")

    def test_id_must_be_a_slug(self) -> None:
        with pytest.raises(PydanticValidationError):
            self._aoi(id="Test Area!")

    def test_utm_is_derived_from_the_centroid(self) -> None:
        assert self._aoi().utm_epsg == "EPSG:32645"

    def test_area_is_plausible(self) -> None:
        # 1.5 deg lon x 1.0 deg lat near 25.7N is roughly 150 x 110 km.
        area = self._aoi().approx_area_km2
        assert 14_000 < area < 19_000

    def test_tile_count_sizes_the_compute_budget(self) -> None:
        # One 512 px tile at 10 m covers ~26.2 km2.
        aoi = self._aoi()
        expected = aoi.approx_area_km2 / ((512 * 10 / 1000) ** 2)
        assert abs(aoi.approx_tile_count() - expected) <= 1

    def test_has_labels_reflects_supervised_feasibility(self) -> None:
        assert self._aoi(label_sources=[]).has_labels is False
        assert self._aoi(label_sources=["Sen1Floods11"]).has_labels is True

    def test_is_immutable(self) -> None:
        with pytest.raises(PydanticValidationError):
            self._aoi().id = "other"  # type: ignore[misc]


class TestRegistry:
    def test_shipped_config_is_valid(self) -> None:
        registry = load_aoi_registry()
        assert len(registry) >= 4
        assert {a.id for a in registry.aois} >= {"bihar_ganga", "mumbai_mmr"}

    def test_every_aoi_states_why_it_was_chosen(self) -> None:
        for aoi in load_aoi_registry().aois:
            assert len(aoi.selection_rationale) > 50, aoi.id

    def test_no_aoi_is_selected_before_phase_2_measures_availability(self) -> None:
        """Phase 2 promotes a candidate on evidence. Until then, none is selected.

        Delete this test in Phase 2 when the measurement has actually been made.
        """
        assert load_aoi_registry().selected == []

    def test_unknown_aoi_raises_a_helpful_error(self) -> None:
        registry = load_aoi_registry()
        with pytest.raises(ValidationError, match="known AOIs"):
            registry.get("atlantis")

    def test_tile_counts_are_within_a_student_compute_budget(self) -> None:
        """A guard against quietly widening a bbox into an untrainable AOI."""
        for aoi in load_aoi_registry().aois:
            assert aoi.approx_tile_count() < 3000, (
                f"{aoi.id} needs ~{aoi.approx_tile_count()} tiles - too large; "
                "narrow the bbox or justify the compute in an ADR"
            )
