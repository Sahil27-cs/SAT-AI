"""The wildfire, extreme-weather and damage modules.

These modules are deterministic computations over observed inputs, so unlike a
trained model they can be tested against values worked out by hand. Most of what
follows checks arithmetic. A few tests check something more important: that the
modules refuse to produce a number when the data does not support one, which is
the behaviour the whole project is built around and the easiest to erode.
"""

from __future__ import annotations

import numpy as np
import pytest

from satai.errors import ValidationError
from satai.hazards.damage import assess_change, log_ratio, optical_change_magnitude
from satai.hazards.extreme_weather import (
    TrackPoint,
    rainfall_anomaly,
    rainfall_return_period,
    track_exposure,
    wind_risk_index,
)
from satai.hazards.wildfire import (
    classify_severity,
    dnbr,
    fire_danger_index,
    nbr,
    summarise_detections,
)
from satai.provenance import SourceKind

# ---------------------------------------------------------------------------
# Wildfire
# ---------------------------------------------------------------------------


class TestBurnSeverity:
    def test_nbr_matches_the_hand_computed_ratio(self) -> None:
        # (0.4 - 0.1) / (0.4 + 0.1) = 0.6
        assert nbr(np.array([0.4]), np.array([0.1]))[0] == pytest.approx(0.6)

    def test_a_zero_denominator_is_nan_not_zero(self) -> None:
        """0.0 is a legitimate NBR meaning equal reflectance.

        Returning it for a masked pixel would make missing data indistinguishable
        from a real measurement downstream.
        """
        assert np.isnan(nbr(np.array([0.0]), np.array([0.0]))[0])

    def test_dnbr_is_positive_where_vegetation_was_lost(self) -> None:
        healthy_then_burned = dnbr(np.array([0.6]), np.array([-0.2]))
        assert healthy_then_burned[0] == pytest.approx(800.0)

    def test_dnbr_scaling_matches_the_published_thresholds(self) -> None:
        """Unscaled dNBR against scaled break points classifies everything as
        unburned -- the units have to agree."""
        raw = dnbr(np.array([0.6]), np.array([-0.2]), scaled=False)
        assert raw[0] == pytest.approx(0.8)

    def test_severity_classes_span_regrowth_to_high(self) -> None:
        field = np.array([-150.0, 50.0, 200.0, 350.0, 550.0, 800.0])
        result = classify_severity(field)
        assert result.n_valid == 6
        assert result.fractions["high"] == pytest.approx(1 / 6)
        assert result.fractions["enhanced_regrowth"] == pytest.approx(1 / 6)

    def test_masked_pixels_are_excluded_from_the_denominator(self) -> None:
        """A cloudy scene must not read as a scene with no fire."""
        field = np.array([800.0, np.nan, np.nan, np.nan])
        result = classify_severity(field)

        assert result.n_valid == 1
        assert result.n_total == 4
        assert result.fractions["high"] == pytest.approx(1.0)
        assert result.burned_fraction == pytest.approx(1.0)

    def test_a_fully_masked_scene_raises_rather_than_reporting_no_burn(self) -> None:
        with pytest.raises(ValidationError, match="entirely masked"):
            classify_severity(np.full(9, np.nan))


class TestFireDanger:
    def test_hot_dry_windy_cured_conditions_score_high(self) -> None:
        index = fire_danger_index(
            temperature_c=np.array([42.0]),
            relative_humidity=np.array([20.0]),
            wind_speed_ms=np.array([12.0]),
            days_since_rain=np.array([28.0]),
        )
        assert index[0] > 0.75

    def test_recent_rain_suppresses_danger_however_hot_and_windy(self) -> None:
        """The multiplicative boundary condition, which an additive index gets
        wrong: saturated fuel means low danger regardless of the other drivers."""
        soaked = fire_danger_index(
            temperature_c=np.array([42.0]),
            relative_humidity=np.array([20.0]),
            wind_speed_ms=np.array([12.0]),
            days_since_rain=np.array([0.0]),
        )
        assert soaked[0] < 0.5

    def test_the_index_stays_in_range(self) -> None:
        index = fire_danger_index(
            temperature_c=np.array([-50.0, 100.0]),
            relative_humidity=np.array([0.0, 200.0]),
            wind_speed_ms=np.array([0.0, 99.0]),
            days_since_rain=np.array([0.0, 365.0]),
        )
        assert np.all((index >= 0.0) & (index <= 1.0))

    def test_ndvi_is_excluded_when_absent_rather_than_imputed(self) -> None:
        """Assuming average greenness would be inventing a fuel observation."""
        args = {
            "temperature_c": np.array([35.0]),
            "relative_humidity": np.array([30.0]),
            "wind_speed_ms": np.array([8.0]),
            "days_since_rain": np.array([20.0]),
        }
        without = fire_danger_index(**args)
        with_cured_fuel = fire_danger_index(**args, ndvi=np.array([0.15]))

        assert without[0] != with_cured_fuel[0]
        assert np.isfinite(without[0])

    def test_mismatched_shapes_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="mismatched shapes"):
            fire_danger_index(
                temperature_c=np.array([30.0, 31.0]),
                relative_humidity=np.array([40.0]),
                wind_speed_ms=np.array([5.0]),
                days_since_rain=np.array([10.0]),
            )


class TestFirmsSummary:
    def test_detections_are_labelled_observations(self) -> None:
        env = summarise_detections([{"frp_mw": 12.0, "confidence": "high"}], window_days=7)
        assert env.source.kind is SourceKind.OBSERVATION
        assert env.is_observation and not env.is_model_output

    def test_absence_of_detections_is_not_evidence_of_no_fire(self) -> None:
        """The caveat that stops an empty result being read as 'no fire'."""
        env = summarise_detections([], window_days=7)
        assert env.value["detection_count"] == 0.0
        assert any("not evidence of no fire" in c for c in env.caveats)

    def test_frp_is_averaged_over_the_records_that_report_it(self) -> None:
        env = summarise_detections([{"frp_mw": 10.0}, {"frp_mw": 20.0}, {}], window_days=7)
        assert env.value["mean_frp_mw"] == pytest.approx(15.0)
        assert env.value["frp_reported_count"] == pytest.approx(2.0)
        assert any("2 of 3" in c for c in env.caveats)

    def test_a_non_positive_window_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="window_days"):
            summarise_detections([], window_days=0)


# ---------------------------------------------------------------------------
# Extreme weather
# ---------------------------------------------------------------------------


class TestRainfall:
    def test_anomaly_is_the_percentile_within_the_record(self) -> None:
        climatology = np.arange(20, dtype=np.float64).reshape(20, 1, 1)
        current = np.array([[19.0]])
        assert rainfall_anomaly(current, climatology)[0, 0] == pytest.approx(95.0)

    def test_a_short_record_is_refused_rather_than_called_a_climatology(self) -> None:
        with pytest.raises(ValidationError, match="at least 10"):
            rainfall_anomaly(np.array([[5.0]]), np.zeros((5, 1, 1)))

    def test_return_period_is_capped_at_the_record_length(self) -> None:
        """An event at the top of a 20-year record is the largest in 20 years,
        not a 1-in-500-year event."""
        capped = rainfall_return_period(np.array([100.0]), record_years=20)
        assert capped[0] == pytest.approx(20.0)

    def test_return_period_rises_with_percentile(self) -> None:
        periods = rainfall_return_period(np.array([50.0, 90.0, 99.0]), record_years=100)
        assert periods[0] < periods[1] < periods[2]


class TestCycloneExposure:
    def test_no_track_means_no_output_because_satai_does_not_forecast_tracks(self) -> None:
        with pytest.raises(ValidationError, match="does not generate tracks"):
            track_exposure([], np.array([[85.0]]), np.array([[20.0]]))

    def test_wind_peaks_at_the_track_and_decays_with_distance(self) -> None:
        point = TrackPoint(lon=85.8, lat=19.8, max_wind_ms=60.0, radius_max_wind_km=40.0)
        lons = np.array([[85.8, 86.8, 88.0]])
        lats = np.array([[19.8, 19.8, 19.8]])

        wind = track_exposure([point], lons, lats)
        assert wind[0, 0] == pytest.approx(60.0)
        assert wind[0, 0] > wind[0, 1] > wind[0, 2]

    def test_a_cell_takes_the_worst_fix_not_the_sum(self) -> None:
        """Summing successive fixes would fabricate a wind speed nothing recorded."""
        track = [
            TrackPoint(lon=85.0, lat=20.0, max_wind_ms=40.0),
            TrackPoint(lon=85.0, lat=20.0, max_wind_ms=55.0),
        ]
        wind = track_exposure(track, np.array([[85.0]]), np.array([[20.0]]))
        assert wind[0, 0] == pytest.approx(55.0)

    def test_an_invalid_track_point_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="out of range"):
            TrackPoint(lon=999.0, lat=20.0, max_wind_ms=50.0)

    def test_wind_risk_is_zero_below_the_damage_threshold(self) -> None:
        """A breezy day must not carry nonzero risk."""
        assert wind_risk_index(np.array([10.0]))[0] == pytest.approx(0.0)

    def test_wind_risk_is_quadratic_not_linear(self) -> None:
        """Wind load scales with the square of speed; a linear index would
        understate the gap between a Category 1 and a Category 4."""
        midpoint = wind_risk_index(np.array([47.5]))[0]
        assert midpoint == pytest.approx(0.25, abs=0.01)

    def test_wind_risk_saturates_at_one(self) -> None:
        assert wind_risk_index(np.array([200.0]))[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Damage
# ---------------------------------------------------------------------------


class TestChangeDetection:
    def test_db_inputs_difference_rather_than_divide(self) -> None:
        assert log_ratio(np.array([-12.0]), np.array([-8.0]))[0] == pytest.approx(4.0)

    def test_linear_power_is_converted_before_differencing(self) -> None:
        # 10 * log10(2/1) ~ 3.0103 dB
        assert log_ratio(np.array([1.0]), np.array([2.0]), already_db=False)[0] == pytest.approx(
            3.0103, abs=1e-3
        )

    def test_both_brightening_and_darkening_count_as_change(self) -> None:
        """A collapsed building can raise backscatter through rubble scattering
        or lower it by removing a double-bounce corner. Looking only for a
        decrease would miss half the damage, so the sign must not matter.
        """
        darkened = assess_change(np.array([-8.0]), threshold_db=3.0)
        brightened = assess_change(np.array([8.0]), threshold_db=3.0)
        assert darkened.fractions == brightened.fractions

        # 8 dB sits in the third band: breaks are at 3, 6 and 9 dB.
        assert darkened.fractions["moderate"] == pytest.approx(1.0)
        assert darkened.changed_fraction == pytest.approx(1.0)

    def test_the_top_band_needs_more_than_three_times_the_threshold(self) -> None:
        result = assess_change(np.array([10.0]), threshold_db=3.0)
        assert result.fractions["high"] == pytest.approx(1.0)

    def test_change_below_the_threshold_is_treated_as_speckle(self) -> None:
        result = assess_change(np.array([0.5, -1.2, 2.0]), threshold_db=3.0)
        assert result.fractions["none"] == pytest.approx(1.0)
        assert result.changed_fraction == pytest.approx(0.0)

    def test_masked_pixels_are_excluded_and_flagged(self) -> None:
        field = np.array([10.0, np.nan, np.nan, np.nan, np.nan])
        result = assess_change(field, threshold_db=3.0)

        assert result.n_valid == 1
        assert result.coverage == pytest.approx(0.2)
        assert any("20% of the scene" in c for c in result.caveats())

    def test_the_result_never_calls_itself_a_damage_grade(self) -> None:
        """Backscatter change is not an EMS-98 damage state and must not be
        displayed as one."""
        result = assess_change(np.array([10.0, 0.0]), threshold_db=3.0)
        joined = " ".join(result.caveats()).lower()

        assert "not verified on the ground" in joined
        assert "not a building damage grade" in joined
        assert "no monetary damage estimate" in joined

    def test_a_fully_masked_pair_raises(self) -> None:
        with pytest.raises(ValidationError, match="entirely masked"):
            assess_change(np.full(4, np.nan))

    def test_a_non_positive_threshold_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="threshold must be positive"):
            assess_change(np.array([1.0]), threshold_db=0.0)

    def test_optical_change_needs_a_band_axis(self) -> None:
        with pytest.raises(ValidationError, match=r"expected \(band, y, x\)"):
            optical_change_magnitude(np.zeros((4, 4)), np.zeros((4, 4)))

    def test_optical_change_is_the_spectral_distance(self) -> None:
        pre = np.zeros((3, 1, 1))
        post = np.array([[[3.0]], [[4.0]], [[0.0]]])
        assert optical_change_magnitude(pre, post)[0, 0] == pytest.approx(5.0)
