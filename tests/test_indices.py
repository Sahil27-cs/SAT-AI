"""Tests for band math.

Every function here can fail silently — a sign error or a unit mistake produces
a plausible-looking raster that poisons everything downstream without raising
anything. So the tests check physics and conventions, not just shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from satai.errors import ValidationError
from satai.preprocessing.indices import (
    db_to_linear,
    dnbr,
    linear_to_db,
    log_ratio_db,
    mndwi,
    nbr,
    ndvi,
    ndwi,
    normalized_difference,
    sar_ratio_db,
)


class TestUnitConversion:
    def test_round_trip(self) -> None:
        linear = np.array([0.001, 0.01, 0.1, 1.0])
        assert np.allclose(db_to_linear(linear_to_db(linear)), linear)

    def test_known_values(self) -> None:
        assert linear_to_db([1.0])[0] == pytest.approx(0.0)
        assert linear_to_db([0.1])[0] == pytest.approx(-10.0)
        assert linear_to_db([0.01])[0] == pytest.approx(-20.0)

    def test_zero_becomes_nan_not_negative_infinity(self) -> None:
        """-inf survives arithmetic and contaminates whatever it touches.

        NaN is designed to be detected; -inf quietly propagates through means
        and percentiles.
        """
        result = linear_to_db([0.0, 1.0])
        assert np.isnan(result[0])
        assert not np.isneginf(result[0])
        assert result[1] == pytest.approx(0.0)


class TestSARRatio:
    def test_ratio_in_db_is_a_difference(self) -> None:
        """The whole point of the module.

        10·log₁₀(VV/VH) in linear equals VV_dB − VH_dB. Dividing two dB arrays
        instead is a common silent error.
        """
        vv_linear = np.array([0.1, 0.05, 0.2])
        vh_linear = np.array([0.01, 0.01, 0.05])

        expected = 10.0 * np.log10(vv_linear / vh_linear)
        got = sar_ratio_db(linear_to_db(vv_linear), linear_to_db(vh_linear))
        assert np.allclose(got, expected)

    def test_differs_from_naive_db_division(self) -> None:
        """Confirm the wrong answer really is different, so the test has teeth."""
        vv_db = np.array([-8.0, -12.0])
        vh_db = np.array([-16.0, -20.0])
        correct = sar_ratio_db(vv_db, vh_db)
        naive = vv_db / vh_db
        assert not np.allclose(correct, naive)
        assert np.allclose(correct, [8.0, 8.0])

    def test_rejects_linear_input(self) -> None:
        """Linear power passed to a dB function must be refused, not computed."""
        with pytest.raises(ValidationError, match="linear power"):
            sar_ratio_db(np.array([0.1, 0.2]), np.array([0.01, 0.02]))

    def test_rejects_implausible_db_range(self) -> None:
        with pytest.raises(ValidationError, match="plausible dB range"):
            sar_ratio_db(np.array([-500.0, -8.0]), np.array([-16.0, -16.0]))

    def test_check_can_be_disabled(self) -> None:
        result = sar_ratio_db(np.array([0.5, 0.6]), np.array([0.1, 0.2]), check_units=False)
        assert np.allclose(result, [0.4, 0.4])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValidationError, match="same shape"):
            sar_ratio_db(np.zeros((2, 2)) - 10, np.zeros((3, 3)) - 16)

    def test_open_water_has_high_ratio(self) -> None:
        """Surface scattering (water) gives VV ≫ VH; volume scattering does not."""
        water = sar_ratio_db(np.array([-18.0]), np.array([-28.0]))
        forest = sar_ratio_db(np.array([-8.0]), np.array([-11.0]))
        assert water[0] > forest[0]


class TestLogRatio:
    def test_sign_convention_new_water_is_negative(self) -> None:
        """Flooding darkens SAR, so the change must be negative.

        Flipping this inverts every downstream interpretation while leaving the
        magnitudes intact, so a histogram check would not catch it.
        """
        dry_before = np.array([-8.0])
        flooded_after = np.array([-20.0])
        change = log_ratio_db(dry_before, flooded_after)
        assert change[0] < 0
        assert change[0] == pytest.approx(-12.0)

    def test_drying_is_positive(self) -> None:
        assert log_ratio_db(np.array([-20.0]), np.array([-8.0]))[0] == pytest.approx(12.0)

    def test_no_change_is_zero(self) -> None:
        assert log_ratio_db(np.array([-12.0]), np.array([-12.0]))[0] == pytest.approx(0.0)

    def test_rejects_linear_input(self) -> None:
        with pytest.raises(ValidationError, match="linear power"):
            log_ratio_db(np.array([0.2, 0.3]), np.array([0.05, 0.06]))


class TestNormalizedDifference:
    def test_basic(self) -> None:
        assert normalized_difference([3.0], [1.0])[0] == pytest.approx(0.5)

    def test_bounded(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.uniform(0.01, 1.0, 500)
        b = rng.uniform(0.01, 1.0, 500)
        result = normalized_difference(a, b)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_zero_denominator_is_nan_not_a_large_number(self) -> None:
        """Both bands near zero means no signal; no signal is missing data."""
        result = normalized_difference([0.0, 1.0], [0.0, 0.0])
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(1.0)

    def test_opposite_signs_cancelling(self) -> None:
        assert np.isnan(normalized_difference([1.0], [-1.0])[0])


class TestOpticalIndices:
    def test_ndwi_positive_over_water(self) -> None:
        """Water: high green, very low NIR."""
        assert ndwi(np.array([0.09]), np.array([0.02]))[0] > 0

    def test_ndwi_negative_over_vegetation(self) -> None:
        """Vegetation: low green, high NIR."""
        assert ndwi(np.array([0.05]), np.array([0.35]))[0] < 0

    def test_mndwi_separates_water_from_built_up_better_than_ndwi(self) -> None:
        """The reason MNDWI exists, and why it matters for the Mumbai AOI.

        Built-up surfaces have high SWIR, so MNDWI pushes them negative, while
        NDWI can leave them positive and confuse them with water.
        """
        built_green, built_nir, built_swir = 0.16, 0.20, 0.30
        assert ndwi(np.array([built_green]), np.array([built_nir]))[0] < 0.0
        assert (
            mndwi(np.array([built_green]), np.array([built_swir]))[0]
            < ndwi(np.array([built_green]), np.array([built_nir]))[0]
        )

    def test_ndvi_high_for_vegetation(self) -> None:
        assert ndvi(np.array([0.04]), np.array([0.40]))[0] > 0.7

    def test_ndvi_near_zero_for_bare_soil(self) -> None:
        assert abs(ndvi(np.array([0.22]), np.array([0.26]))[0]) < 0.2

    def test_nbr_drops_after_burning(self) -> None:
        healthy = nbr(np.array([0.40]), np.array([0.08]))
        burnt = nbr(np.array([0.12]), np.array([0.30]))
        assert healthy[0] > burnt[0]


class TestDNBR:
    def test_sign_convention_is_pre_minus_post(self) -> None:
        """Opposite to the SAR log ratio, and deliberately so.

        dNBR is conventionally pre−post so severity increases upward; the SAR
        log ratio is post−pre so darkening is negative. Assuming either one
        matches the other is how sign errors get in.
        """
        result = dnbr(np.array([0.6]), np.array([-0.2]))
        assert result[0] == pytest.approx(0.8)
        assert result[0] > 0

    def test_no_burn_is_near_zero(self) -> None:
        assert dnbr(np.array([0.55]), np.array([0.54]))[0] == pytest.approx(0.01)

    def test_regrowth_is_negative(self) -> None:
        assert dnbr(np.array([0.2]), np.array([0.5]))[0] < 0


class TestNaNPropagation:
    def test_nan_survives_sar_ratio(self) -> None:
        """Nodata must stay nodata, not become a number."""
        result = sar_ratio_db(np.array([-8.0, np.nan]), np.array([-16.0, -16.0]))
        assert result[0] == pytest.approx(8.0)
        assert np.isnan(result[1])

    def test_nan_survives_normalized_difference(self) -> None:
        result = normalized_difference([3.0, np.nan], [1.0, 1.0])
        assert result[0] == pytest.approx(0.5)
        assert np.isnan(result[1])

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            normalized_difference([], [])
