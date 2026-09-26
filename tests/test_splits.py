"""Tests for the evaluation split machinery (ADR-009).

These tests encode the protocol. A leaked chip raises nothing by itself — it
just makes every reported metric better than it is — so the guarantees have to
be asserted rather than inspected.

The region names and chip counts below are the real ones, verified against the
Sen1Floods11 bucket on 2026-09-22.
"""

from __future__ import annotations

import pytest

from satai.errors import ValidationError
from satai.preprocessing.splits import (
    BOLIVIA,
    ChipRef,
    Fold,
    SplitProtocol,
    leave_one_region_out,
    official_fold,
    parse_chip_name,
)

# Verified 2026-09-22 from the public bucket.
REAL_COUNTS = {
    "Bolivia": 15,
    "Ghana": 53,
    "India": 68,
    "Mekong": 30,
    "Nigeria": 18,
    "Pakistan": 28,
    "Paraguay": 67,
    "Somalia": 26,
    "Spain": 30,
    "Sri-Lanka": 42,
    "USA": 69,
}


def make_chips(counts: dict[str, int] | None = None) -> list[ChipRef]:
    counts = counts or REAL_COUNTS
    return [
        ChipRef(region=region, chip_id=f"{i:06d}") for region, n in counts.items() for i in range(n)
    ]


class TestParsing:
    def test_full_filename(self) -> None:
        chip = parse_chip_name("India_902184_S1Hand.tif")
        assert chip.region == "India"
        assert chip.chip_id == "902184"

    def test_path_prefix_is_stripped(self) -> None:
        chip = parse_chip_name("v1.1/data/flood_events/HandLabeled/S1Hand/Ghana_12345_S1Hand.tif")
        assert chip.region == "Ghana"

    def test_hyphenated_region(self) -> None:
        """Sri-Lanka is hyphenated, not underscored — the split must survive it."""
        assert parse_chip_name("Sri-Lanka_1234_S1Hand.tif").region == "Sri-Lanka"

    def test_bare_name(self) -> None:
        assert parse_chip_name("India_902184").chip_id == "902184"

    def test_rejects_unparseable(self) -> None:
        with pytest.raises(ValidationError, match="cannot parse"):
            parse_chip_name("nonsense.tif")

    def test_filename_round_trip(self) -> None:
        name = "India_902184_LabelHand.tif"
        assert parse_chip_name(name).filename("LabelHand") == name


class TestLORO:
    def test_one_fold_per_region_excluding_bolivia(self) -> None:
        folds = list(leave_one_region_out(make_chips()))
        assert len(folds) == 10  # 11 regions minus Bolivia
        assert {f.test_regions[0] for f in folds} == set(REAL_COUNTS) - {BOLIVIA}

    def test_every_fold_is_region_disjoint(self) -> None:
        """The property the whole protocol exists to provide."""
        for fold in leave_one_region_out(make_chips()):
            assert fold.is_region_disjoint, fold.name

    def test_no_chip_leaks_between_partitions(self) -> None:
        for fold in leave_one_region_out(make_chips()):
            train, val, test = set(fold.train), set(fold.val), set(fold.test)
            assert not train & val
            assert not train & test
            assert not val & test

    def test_test_partition_is_exactly_one_region(self) -> None:
        for fold in leave_one_region_out(make_chips()):
            assert {c.region for c in fold.test} == {fold.test_regions[0]}

    def test_validation_is_a_held_out_region_not_sampled_from_training(self) -> None:
        """Validation must not share a region with training.

        Sampling validation chips at random from the training regions would
        make early stopping tune on data statistically identical to the
        training set, which is model selection on a leak.
        """
        for fold in leave_one_region_out(make_chips()):
            train_regions = {c.region for c in fold.train}
            val_regions = {c.region for c in fold.val}
            assert not train_regions & val_regions
            assert val_regions != {fold.test_regions[0]}

    def test_bolivia_appears_nowhere(self) -> None:
        """Reserved as the final untouched hold-out (ADR-009)."""
        for fold in leave_one_region_out(make_chips()):
            for partition in (fold.train, fold.val, fold.test):
                assert all(c.region != BOLIVIA for c in partition), fold.name

    def test_india_fold_tests_on_all_68_indian_chips(self) -> None:
        """The fold that C3's domain-transfer claim rests on."""
        india = next(f for f in leave_one_region_out(make_chips()) if f.test_regions == ("India",))
        assert len(india.test) == 68
        assert all(c.region == "India" for c in india.test)
        assert all(c.region != "India" for c in india.train)

    def test_chip_totals_are_conserved(self) -> None:
        total_non_bolivia = sum(n for r, n in REAL_COUNTS.items() if r != BOLIVIA)
        for fold in leave_one_region_out(make_chips()):
            assert sum(fold.sizes.values()) == total_non_bolivia

    def test_folds_are_deterministic(self) -> None:
        """Same input, same split — a precondition for reproducibility."""
        first = [(f.name, f.train, f.val, f.test) for f in leave_one_region_out(make_chips())]
        second = [(f.name, f.train, f.val, f.test) for f in leave_one_region_out(make_chips())]
        assert first == second

    def test_too_few_regions_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least 3 regions"):
            list(leave_one_region_out(make_chips({"India": 10, "Ghana": 10})))

    def test_exclusion_list_is_configurable(self) -> None:
        folds = list(leave_one_region_out(make_chips(), exclude_regions=()))
        assert len(folds) == 11
        assert BOLIVIA in {f.test_regions[0] for f in folds}


class TestOfficialFold:
    def test_reports_itself_as_not_region_disjoint(self) -> None:
        """The verified finding, encoded so it cannot be forgotten.

        Every region except Bolivia appears in train, valid and test at ~58/21/21.
        """
        fold = official_fold(
            train_names=["India_1_S1Hand.tif", "Ghana_1_S1Hand.tif"],
            val_names=["India_2_S1Hand.tif"],
            test_names=["India_3_S1Hand.tif", "Ghana_2_S1Hand.tif"],
        )
        assert fold.is_region_disjoint is False
        assert fold.protocol == SplitProtocol.OFFICIAL

    def test_carries_a_note_pointing_at_the_adr(self) -> None:
        fold = official_fold(["India_1.tif"], ["India_2.tif"], ["India_3.tif"])
        assert any("ADR-009" in note for note in fold.notes)

    def test_still_refuses_actual_chip_overlap(self) -> None:
        """Leaky protocol or not, the same chip in two partitions is a bug."""
        with pytest.raises(ValidationError, match="appear in both"):
            official_fold(
                train_names=["India_1_S1Hand.tif"],
                val_names=[],
                test_names=["India_1_S1Hand.tif"],
            )


class TestFold:
    def test_disjointness_is_checked_at_construction(self) -> None:
        chip = ChipRef("India", "1")
        with pytest.raises(ValidationError, match="appear in both"):
            Fold(name="bad", protocol="test", train=(chip,), val=(), test=(chip,))

    def test_summary_names_the_protocol_and_disjointness(self) -> None:
        india = next(f for f in leave_one_region_out(make_chips()) if f.test_regions == ("India",))
        summary = india.summary()
        assert "loro_india" in summary
        assert "region-disjoint" in summary
        assert "NOT" not in summary
