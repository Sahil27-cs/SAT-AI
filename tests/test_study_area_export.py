"""The frontend's study-area catalogue must match `configs/aoi.yaml`.

The frontend imports a generated JSON snapshot of the AOI registry rather than
fetching it, because study areas are configuration and change on a pull request
rather than on a satellite revisit. The cost of that choice is a second copy,
and the thing that keeps a second copy honest is this module: if someone edits
the YAML and forgets to regenerate, the build goes red here instead of the
interface quietly serving a stale region list.

Beyond the sync check, these tests enforce the rules that make the catalogue
trustworthy — chiefly that no event claims to be verified without a sensor, a
date and a retrievable product behind it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_study_areas import OUTPUT, build_payload  # noqa: E402
from satai.geo.aoi import load_aoi_registry  # noqa: E402


@pytest.fixture(scope="module")
def exported() -> dict[str, Any]:
    if not OUTPUT.exists():
        pytest.fail(
            f"{OUTPUT.relative_to(REPO_ROOT)} is missing. Run: python scripts/export_study_areas.py"
        )
    data: dict[str, Any] = json.loads(OUTPUT.read_text(encoding="utf-8"))
    return data


def test_the_generated_file_is_not_stale(exported: dict[str, Any]) -> None:
    """The whole reason a generated file is acceptable."""
    assert exported == build_payload(), (
        "frontend/lib/study-areas.generated.json is out of date with "
        "configs/aoi.yaml. Run: python scripts/export_study_areas.py"
    )


def test_every_configured_aoi_reaches_the_frontend(exported: dict[str, Any]) -> None:
    registry = load_aoi_registry()
    assert {a["id"] for a in exported["studyAreas"]} == {a.id for a in registry.aois}


def test_the_catalogue_covers_more_than_one_country(exported: dict[str, Any]) -> None:
    """Cross-border transfer is a stated contribution; it needs a second country."""
    countries = {a["country"] for a in exported["studyAreas"]}
    assert len(countries) >= 2
    assert "Nepal" in countries


def test_every_declared_hazard_has_an_aoi(exported: dict[str, Any]) -> None:
    hazards = {h for a in exported["studyAreas"] for h in a["primaryHazards"]}
    assert {"flood", "wildfire", "cyclone"} <= hazards


class TestEventVerification:
    """`verified: true` is a claim about what was checked, so it is constrained."""

    def _events(self, exported: dict[str, Any]) -> list[dict[str, Any]]:
        return [e for a in exported["studyAreas"] for e in a["events"]]

    def test_a_verified_event_names_its_sensor_and_date(self, exported: dict[str, Any]) -> None:
        for event in self._events(exported):
            if event["verified"]:
                assert event["sensor"], f"{event['id']} is verified with no sensor"
                assert event["occurredOn"], f"{event['id']} is verified with no date"

    def test_a_verified_event_carries_a_retrievable_product(self, exported: dict[str, Any]) -> None:
        """Without a product there is nothing to validate against, so the event
        is a news report rather than a dataset."""
        for event in self._events(exported):
            if event["verified"]:
                assert event["referenceProducts"], f"{event['id']} has no reference product"
                for url in event["referenceProducts"]:
                    assert url.startswith("https://"), f"{event['id']}: {url!r}"

    def test_every_event_explains_what_was_checked(self, exported: dict[str, Any]) -> None:
        for event in self._events(exported):
            assert len(event["verificationNote"]) > 60, event["id"]

    def test_an_unverified_event_is_never_offered_as_analysis(
        self, exported: dict[str, Any]
    ) -> None:
        """The interface keys off displayStatus; it must not read as a result."""
        for event in self._events(exported):
            if not event["verified"]:
                assert "CANDIDATE" in event["displayStatus"]
                assert "ANALYSIS" not in event["displayStatus"]

    def test_hazard_coverage_only_counts_verified_events(self, exported: dict[str, Any]) -> None:
        for area in exported["studyAreas"]:
            verified_hazards = {e["hazard"] for e in area["events"] if e["verified"]}
            for hazard, covered in area["hazardCoverage"].items():
                assert covered == (hazard in verified_hazards), (
                    f"{area['id']}: {hazard} coverage disagrees with its verified events"
                )


class TestNepalStudyArea:
    """The cross-border transfer target, specifically requested and specifically
    required to be real rather than a dropdown entry."""

    @pytest.fixture
    def nepal(self, exported: dict[str, Any]) -> dict[str, Any]:
        return next(a for a in exported["studyAreas"] if a["id"] == "nepal_koshi_terai")

    def test_nepal_is_a_held_out_transfer_target_not_a_training_region(
        self, nepal: dict[str, Any]
    ) -> None:
        assert nepal["studyRole"] == "transfer_evaluation"

    def test_nepal_carries_a_verified_sentinel_1_event(self, nepal: dict[str, Any]) -> None:
        event = next(e for e in nepal["events"] if e["verified"])
        assert "Sentinel-1" in event["sensor"]
        assert event["occurredOn"] == "2024-09-27"
        assert event["displayStatus"] == "HISTORICAL EVENT ANALYSIS"

    def test_nepal_shares_a_utm_zone_with_the_training_region(
        self, exported: dict[str, Any], nepal: dict[str, Any]
    ) -> None:
        """The rationale for this AOI is that it holds physiography fixed while
        changing the country. Same UTM zone as Bihar is a cheap check that the
        two really are the same alluvial system rather than merely adjacent.
        """
        bihar = next(a for a in exported["studyAreas"] if a["id"] == "bihar_ganga")
        assert nepal["utmEpsg"] == bihar["utmEpsg"]

    def test_no_reference_figure_is_stored_as_a_satai_result(self, nepal: dict[str, Any]) -> None:
        """UNOSAT's reported flood area and exposure counts are validation data.

        Storing them here would put them one refactor away from being rendered
        as this system's output, so the config records the products and not
        their numbers.
        """
        blob = json.dumps(nepal)
        for figure in ("19,000", "19000", "50 km", "7,000 km"):
            assert figure not in blob, f"a UNOSAT reference figure leaked into config: {figure}"
