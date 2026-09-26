"""Tests for leakage-safe normalisation.

The leak these guard against is invisible in the metrics: fitting statistics
across all partitions makes every reported number slightly better and nothing
looks wrong. So the constraints are asserted rather than trusted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from satai.errors import ValidationError
from satai.preprocessing.normalize import BandStats, StackNormalizer, fit_band_stats


@pytest.fixture
def sar_samples() -> np.ndarray:
    """Plausible Sentinel-1 VV in dB: land around -10, water around -20."""
    rng = np.random.default_rng(42)
    land = rng.normal(-10.0, 2.5, 9_000)
    water = rng.normal(-20.0, 1.5, 1_000)
    return np.concatenate([land, water])


class TestFitBandStats:
    def test_records_fold_and_band(self, sar_samples: np.ndarray) -> None:
        """Statistics must carry their provenance or they can be misapplied."""
        stats = fit_band_stats(sar_samples, band="VV", fold="loro_india")
        assert stats.band == "VV"
        assert stats.fitted_on_fold == "loro_india"
        assert stats.fitted_on_partition == "train"

    def test_percentile_clipping_bounds(self, sar_samples: np.ndarray) -> None:
        stats = fit_band_stats(sar_samples, band="VV", fold="f")
        assert stats.lower == pytest.approx(np.percentile(sar_samples, 2.0))
        assert stats.upper == pytest.approx(np.percentile(sar_samples, 98.0))

    def test_outliers_do_not_set_the_scale(self) -> None:
        """A handful of bright targets must not define the band's range."""
        clean = np.random.default_rng(0).normal(-10.0, 2.0, 10_000)
        contaminated = np.concatenate([clean, np.full(5, 5_000.0)])

        a = fit_band_stats(clean, band="VV", fold="f")
        b = fit_band_stats(contaminated, band="VV", fold="f")
        assert b.std == pytest.approx(a.std, rel=0.05)
        assert b.upper < 100.0

    def test_nan_is_ignored_and_counted(self) -> None:
        values = np.array([1.0, 2.0, np.nan, 3.0, np.nan] * 40)
        stats = fit_band_stats(values, band="B", fold="f")
        assert stats.n_samples == 200
        assert stats.n_valid == 120
        assert stats.valid_fraction == pytest.approx(0.6)

    def test_all_nan_raises(self) -> None:
        with pytest.raises(ValidationError, match="NaN or infinite"):
            fit_band_stats(np.full(100, np.nan), band="B", fold="f")

    def test_constant_band_raises_rather_than_dividing_by_zero(self) -> None:
        with pytest.raises(ValidationError, match="constant"):
            fit_band_stats(np.full(1_000, 7.0), band="B", fold="f")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValidationError, match="no samples"):
            fit_band_stats(np.array([]), band="B", fold="f")

    def test_bad_percentiles_raise(self) -> None:
        with pytest.raises(ValidationError, match="percentiles must satisfy"):
            fit_band_stats(
                np.arange(100.0), band="B", fold="f", lower_percentile=90.0, upper_percentile=10.0
            )

    def test_non_train_partition_is_refused(self) -> None:
        """Fitting on validation or test is the leak; make it an explicit refusal."""
        with pytest.raises(ValidationError, match="Fitting on anything but"):
            fit_band_stats(np.arange(1_000.0), band="B", fold="f", partition="test")


class TestApply:
    def test_standardises_training_data(self, sar_samples: np.ndarray) -> None:
        stats = fit_band_stats(sar_samples, band="VV", fold="f")
        transformed = stats.apply(sar_samples)
        assert float(transformed.mean()) == pytest.approx(0.0, abs=0.05)
        assert float(transformed.std()) == pytest.approx(1.0, abs=0.05)

    def test_clips_beyond_the_fitted_range(self, sar_samples: np.ndarray) -> None:
        stats = fit_band_stats(sar_samples, band="VV", fold="f")
        extreme = stats.apply(np.array([1e6]))
        at_upper = stats.apply(np.array([stats.upper]))
        assert extreme[0] == pytest.approx(at_upper[0])

    def test_nan_stays_nan(self, sar_samples: np.ndarray) -> None:
        stats = fit_band_stats(sar_samples, band="VV", fold="f")
        assert np.isnan(stats.apply(np.array([np.nan, -10.0]))[0])

    def test_invert_round_trips_within_the_clip_range(self, sar_samples: np.ndarray) -> None:
        stats = fit_band_stats(sar_samples, band="VV", fold="f")
        values = np.array([-12.0, -10.0, -8.0])
        assert np.allclose(stats.invert(stats.apply(values)), values)


class TestLeakagePrevention:
    def test_test_partition_does_not_shift_the_transform(self) -> None:
        """The core guarantee.

        Under LORO the test region is a different flood in a different country,
        so its radiometry genuinely differs. Fitting across both partitions
        would hand the model information about the test distribution.
        """
        rng = np.random.default_rng(1)
        train = rng.normal(-10.0, 2.0, 10_000)
        test = rng.normal(-25.0, 5.0, 10_000)  # a visibly different distribution

        correct = fit_band_stats(train, band="VV", fold="f")
        leaked = fit_band_stats(np.concatenate([train, test]), band="VV", fold="f")

        assert abs(correct.mean - leaked.mean) > 1.0
        assert correct.lower > leaked.lower

    def test_applying_another_folds_statistics_is_refused(self) -> None:
        normalizer = StackNormalizer(fold="loro_india")
        normalizer.fit_band("VV", np.random.default_rng(0).normal(-10, 2, 1_000))
        with pytest.raises(ValidationError, match="would leak one fold"):
            normalizer.assert_fold("loro_ghana")

    def test_matching_fold_passes(self) -> None:
        normalizer = StackNormalizer(fold="loro_india")
        normalizer.assert_fold("loro_india")


class TestStackNormalizer:
    def _fitted(self) -> StackNormalizer:
        rng = np.random.default_rng(7)
        normalizer = StackNormalizer(fold="loro_india")
        normalizer.fit_band("VV", rng.normal(-10.0, 2.0, 5_000))
        normalizer.fit_band("VH", rng.normal(-17.0, 2.5, 5_000))
        return normalizer

    def test_transform_stack(self) -> None:
        normalizer = self._fitted()
        stack = np.stack([np.full((4, 4), -10.0), np.full((4, 4), -17.0)])
        out = normalizer.transform_stack(stack, ["VV", "VH"])
        assert out.shape == (2, 4, 4)
        assert abs(float(out[0].mean())) < 0.2

    def test_refitting_a_band_is_refused(self) -> None:
        """Re-fitting usually means two sample sets got mixed."""
        normalizer = self._fitted()
        with pytest.raises(ValidationError, match="already has statistics"):
            normalizer.fit_band("VV", np.random.default_rng(0).normal(-10, 2, 1_000))

    def test_unknown_band_raises_with_the_known_list(self) -> None:
        with pytest.raises(ValidationError, match="VH, VV"):
            self._fitted().transform("NDWI", np.zeros((4, 4)))

    def test_band_count_mismatch_raises(self) -> None:
        normalizer = self._fitted()
        with pytest.raises(ValidationError, match="2 bands but 1"):
            normalizer.transform_stack(np.zeros((2, 4, 4)), ["VV"])

    def test_non_3d_stack_raises(self) -> None:
        with pytest.raises(ValidationError, match="3-D"):
            self._fitted().transform_stack(np.zeros((4, 4)), ["VV"])

    def test_round_trip_through_disk(self, tmp_path: Path) -> None:
        """Statistics travel with the model (ADR-010), so they must serialise."""
        original = self._fitted()
        path = original.save(tmp_path / "norm.json")

        loaded = StackNormalizer.load(path)
        assert loaded.fold == original.fold
        assert loaded.bands == original.bands
        assert loaded["VV"].mean == pytest.approx(original["VV"].mean)

        values = np.array([-9.0, -11.0])
        assert np.allclose(loaded.transform("VV", values), original.transform("VV", values))

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="not found"):
            StackNormalizer.load(tmp_path / "nope.json")


class TestBandStatsValidation:
    def test_inverted_clip_bounds_raise(self) -> None:
        with pytest.raises(ValidationError, match="must exceed"):
            BandStats(
                band="B",
                lower=5.0,
                upper=1.0,
                mean=3.0,
                std=1.0,
                lower_percentile=2.0,
                upper_percentile=98.0,
                n_samples=10,
                n_valid=10,
                fitted_on_fold="f",
            )

    def test_zero_std_raises(self) -> None:
        with pytest.raises(ValidationError, match="constant band"):
            BandStats(
                band="B",
                lower=1.0,
                upper=5.0,
                mean=3.0,
                std=0.0,
                lower_percentile=2.0,
                upper_percentile=98.0,
                n_samples=10,
                n_valid=10,
                fitted_on_fold="f",
            )
