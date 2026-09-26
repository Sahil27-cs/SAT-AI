"""Tests for the AOI selection scoring in ``scripts/measure_aoi_availability.py``.

This scoring decides the study area, which decides everything downstream, so it
is tested rather than eyeballed. The first test below exists because an offline
dry run caught a real defect: the original scorer normalised Sentinel-1 scene
count by AOI area, which rewarded an AOI for being small and would have ranked
Mumbai above Bihar for the wrong reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from measure_aoi_availability import (  # noqa: E402
    DATA_WEIGHTS,
    AOIMeasurement,
)

WINDOW_DAYS = 153  # 2024-06-01 to 2024-10-31


def measurement(**kw: object) -> AOIMeasurement:
    base: dict[str, object] = {
        "aoi_id": "test",
        "name": "Test",
        "bbox": (85.0, 25.2, 86.5, 26.2),
        "area_km2": 16_637.0,
        "tile_count": 635,
        "utm_epsg": "EPSG:32645",
        "window_days": WINDOW_DAYS,
    }
    base.update(kw)
    return AOIMeasurement(**base)  # type: ignore[arg-type]


class TestCoverageScore:
    def test_does_not_reward_small_area(self) -> None:
        """Two AOIs with identical temporal coverage must score identically.

        A Sentinel-1 IW swath is ~250 km wide, so a small AOI is fully covered
        by fewer distinct scenes. Normalising scene count by area would make a
        small AOI look data-rich when it is merely small -- and
        ``score_feasibility`` already accounts for size, so area would be
        double-counted.
        """
        small = measurement(area_km2=2_617.0, tile_count=100, s1_dates=25, s1_scenes=40)
        large = measurement(area_km2=19_837.0, tile_count=757, s1_dates=25, s1_scenes=400)
        assert small.score_s1_coverage() == large.score_s1_coverage()

    def test_saturates_at_the_six_day_ideal(self) -> None:
        """~25 acquisition dates over 153 days is the constellation's best case."""
        assert measurement(s1_dates=26).score_s1_coverage() == 1.0
        assert measurement(s1_dates=100).score_s1_coverage() == 1.0

    def test_scales_with_observations(self) -> None:
        sparse = measurement(s1_dates=6).score_s1_coverage()
        dense = measurement(s1_dates=18).score_s1_coverage()
        assert 0.0 < sparse < dense < 1.0

    def test_zero_without_data(self) -> None:
        assert measurement(s1_dates=0).score_s1_coverage() == 0.0


class TestLabelScore:
    def test_declared_labels_do_not_earn_full_marks(self) -> None:
        """A config asserting labels exist is not evidence that they do.

        Full marks require ``verify_sen1floods11.py`` to have confirmed it.
        """
        declared = measurement(declared_label_sources=["Sen1Floods11"])
        assert declared.score_labels() == pytest.approx(0.6)

    def test_verification_earns_full_marks(self) -> None:
        verified = measurement(declared_label_sources=["Sen1Floods11"], labels_verified=True)
        assert verified.score_labels() == 1.0

    def test_no_labels_scores_zero(self) -> None:
        assert measurement(declared_label_sources=[]).score_labels() == 0.0


class TestRevisitScore:
    @pytest.mark.parametrize(
        ("days", "expected"),
        [(6.0, 1.0), (12.0, 0.5), (18.0, 0.0), (30.0, 0.0)],
    )
    def test_shorter_revisit_scores_higher(self, days: float, expected: float) -> None:
        assert measurement(s1_median_revisit_days=days).score_revisit() == pytest.approx(expected)

    def test_unmeasured_revisit_scores_zero(self) -> None:
        assert measurement(s1_median_revisit_days=None).score_revisit() == 0.0


class TestFeasibilityScore:
    def test_within_budget_scores_full(self) -> None:
        assert measurement(tile_count=100).score_feasibility() == 1.0

    def test_oversized_aoi_is_penalised(self) -> None:
        big = measurement(tile_count=3200).score_feasibility()
        assert big == pytest.approx(0.25)


class TestTotalScore:
    def test_data_weights_sum_to_one(self) -> None:
        assert sum(DATA_WEIGHTS.values()) == pytest.approx(1.0)

    def test_labels_are_multiplicative_not_additive(self) -> None:
        """No labels means a training score of exactly zero, not a reduced one.

        Same argument as the risk engine (ADR-008): an AOI without labels
        cannot support supervised training at all, so no amount of imagery
        quality compensates. An additive weighting would let a data-rich
        unlabelled AOI outrank a labelled one, which is backwards.
        """
        perfect_but_unlabelled = measurement(
            declared_label_sources=[],
            s1_dates=30,
            s1_median_revisit_days=6.0,
            s2_usable_fraction=0.9,
            tile_count=50,
        )
        assert perfect_but_unlabelled.total_score() == 0.0
        assert perfect_but_unlabelled.data_score() == pytest.approx(1.0)
        assert perfect_but_unlabelled.can_train is False

    def test_unlabelled_aoi_cannot_win_on_coverage_alone(self) -> None:
        """A data-rich AOI with no labels must not outrank a labelled one.

        This is the ranking property that matters: coverage is worthless
        without ground truth to train against.
        """
        unlabelled = measurement(
            declared_label_sources=[],
            s1_dates=26,
            s1_median_revisit_days=6.0,
            s2_usable_fraction=0.5,
            tile_count=100,
        )
        labelled = measurement(
            declared_label_sources=["Sen1Floods11"],
            s1_dates=13,
            s1_median_revisit_days=12.0,
            s2_usable_fraction=0.1,
            tile_count=635,
        )
        assert labelled.total_score() > unlabelled.total_score()

    def test_data_score_survives_for_transfer_targets(self) -> None:
        """A zero training score must not erase the AOI's evaluation value.

        Mumbai's role in this project is exactly this: unlabelled, so useless
        for training, but a valid and important transfer-evaluation target
        (ADR-007).
        """
        transfer_target = measurement(
            declared_label_sources=[],
            s1_dates=20,
            s1_median_revisit_days=9.0,
            s2_usable_fraction=0.2,
            tile_count=100,
        )
        assert transfer_target.total_score() == 0.0
        assert transfer_target.data_score() > 0.5

    def test_score_is_bounded(self) -> None:
        best = measurement(
            declared_label_sources=["x"],
            labels_verified=True,
            s1_dates=30,
            s1_median_revisit_days=6.0,
            s2_usable_fraction=0.9,
            tile_count=50,
        )
        assert 0.0 <= measurement().total_score() <= best.total_score() <= 1.0
