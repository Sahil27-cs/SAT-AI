"""Extreme weather: rainfall anomaly and cyclone exposure along an observed track.

**SAT-AI does not forecast cyclone tracks or intensity.** Track forecasting
requires assimilated observations and numerical weather prediction, which IMD
operates. What this module does is assess *exposure* along a track that has
already been observed or has been supplied by the caller — a retrospective or
scenario computation, never a prediction of where a storm will go.

The distinction is enforced by the signature: :func:`track_exposure` takes the
track as an argument. There is no code path in this module that produces one.

Two computations:

``rainfall_anomaly``
    Where current rainfall sits in the historical distribution for that place.
    A percentile, not a threshold — "this is the wettest 3-day total in the
    record for this cell" is a defensible statement; "this is a flood" is not.

``track_exposure`` / ``wind_risk_index``
    Distance-decay of wind exposure from best-track points, combined with the
    exposure surface. Answers "what lay within the damaging-wind envelope of
    this storm", which is what post-event assessment needs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from satai.errors import ValidationError
from satai.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "SAFFIR_SIMPSON_MS",
    "TrackPoint",
    "rainfall_anomaly",
    "rainfall_return_period",
    "track_exposure",
    "wind_risk_index",
]

FloatArray = npt.NDArray[np.floating]

#: Saffir-Simpson category lower bounds in m/s, for labelling an observed
#: intensity. Included for description only -- SAT-AI never assigns a category
#: to anything it has not been given a measured wind speed for.
SAFFIR_SIMPSON_MS: dict[int, float] = {1: 33.0, 2: 43.0, 3: 50.0, 4: 58.0, 5: 70.0}


def rainfall_anomaly(
    current: npt.ArrayLike,
    climatology: npt.ArrayLike,
    *,
    min_years: int = 10,
) -> FloatArray:
    """Percentile of ``current`` within the per-cell ``climatology`` distribution.

    Parameters
    ----------
    current:
        Rainfall accumulation, shape ``(y, x)``.
    climatology:
        Historical accumulations for the same window and cells, shape
        ``(n_years, y, x)``.
    min_years:
        Refuse below this. A percentile computed against five years is not a
        climatology, it is a small sample wearing the word -- and the resulting
        "99th percentile" would be reported with a confidence the data cannot
        support.

    Returns
    -------
    Percentile in [0, 100] per cell. 95 means the current total exceeds 95 % of
    the historical record for that cell and that window.
    """
    now = np.asarray(current, dtype=np.float64)
    history = np.asarray(climatology, dtype=np.float64)

    if history.ndim != now.ndim + 1:
        raise ValidationError(
            f"climatology must have one more axis than current (got {history.ndim} and {now.ndim})"
        )
    if history.shape[1:] != now.shape:
        raise ValidationError(
            f"climatology cell grid {history.shape[1:]} does not match current {now.shape}"
        )
    if history.shape[0] < min_years:
        raise ValidationError(
            f"climatology has {history.shape[0]} years; at least {min_years} are "
            f"required before a percentile is meaningful"
        )

    # Fraction of historical years the current value exceeds, per cell.
    below = np.sum(history < now[np.newaxis, ...], axis=0)
    finite_years = np.sum(np.isfinite(history), axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        percentile = np.where(finite_years > 0, 100.0 * below / finite_years, np.nan)
    result: FloatArray = np.asarray(percentile, dtype=np.float64)
    return result


def rainfall_return_period(percentile: npt.ArrayLike, *, record_years: int) -> FloatArray:
    """Approximate return period in years from an empirical percentile.

    Capped at the record length. An event at the 100th percentile of a 20-year
    record is "the largest in 20 years", not a 1-in-500-year event: extrapolating
    a return period past the data is how a defensible statistic becomes an
    indefensible headline, and the cap is what stops it here.
    """
    if record_years < 1:
        raise ValidationError(f"record_years must be at least 1, got {record_years}")

    p = np.clip(np.asarray(percentile, dtype=np.float64), 0.0, 100.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        period = np.where(p >= 100.0, float(record_years), 1.0 / (1.0 - p / 100.0))
    result: FloatArray = np.clip(np.nan_to_num(period, nan=1.0), 1.0, float(record_years))
    return result


@dataclass(frozen=True)
class TrackPoint:
    """One best-track fix. Supplied, never predicted by SAT-AI."""

    lon: float
    lat: float
    max_wind_ms: float
    radius_max_wind_km: float = 50.0

    def __post_init__(self) -> None:
        if not -180.0 <= self.lon <= 180.0 or not -90.0 <= self.lat <= 90.0:
            raise ValidationError(f"track point out of range: ({self.lon}, {self.lat})")
        if self.max_wind_ms < 0:
            raise ValidationError(f"negative wind speed: {self.max_wind_ms}")
        if self.radius_max_wind_km <= 0:
            raise ValidationError(f"radius must be positive: {self.radius_max_wind_km}")


def _haversine_km(lon1: FloatArray, lat1: FloatArray, lon2: float, lat2: float) -> FloatArray:
    """Great-circle distance in km. Plane geometry on lon/lat would be wrong."""
    earth_radius_km = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * math.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    result: FloatArray = 2.0 * earth_radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return result


def track_exposure(
    track: Sequence[TrackPoint],
    lons: npt.ArrayLike,
    lats: npt.ArrayLike,
) -> FloatArray:
    """Peak modelled wind at each grid cell from an observed or supplied track.

    A modified Rankine vortex: wind holds near the maximum inside the radius of
    maximum wind and decays as ``r ** -0.5`` outside it. This is the standard
    parametric profile for exposure work and it is an approximation — it has no
    asymmetry from storm motion, no terrain roughness and no boundary-layer
    correction, so it will overstate wind over rough terrain and understate it
    on the dangerous semicircle.

    Each cell takes the **maximum** over all track points, not the sum: a cell
    experiences the worst the storm brought as it passed, and adding successive
    fixes would fabricate a wind speed no instrument recorded.
    """
    if not track:
        raise ValidationError("no track points supplied; SAT-AI does not generate tracks")

    lon_grid = np.asarray(lons, dtype=np.float64)
    lat_grid = np.asarray(lats, dtype=np.float64)
    if lon_grid.shape != lat_grid.shape:
        raise ValidationError(f"lon grid {lon_grid.shape} does not match lat grid {lat_grid.shape}")

    peak = np.zeros(lon_grid.shape, dtype=np.float64)
    for point in track:
        distance = _haversine_km(lon_grid, lat_grid, point.lon, point.lat)
        inside = distance <= point.radius_max_wind_km
        with np.errstate(divide="ignore", invalid="ignore"):
            decay = np.sqrt(point.radius_max_wind_km / np.maximum(distance, 1e-6))
        wind = np.where(inside, point.max_wind_ms, point.max_wind_ms * np.clip(decay, 0.0, 1.0))
        peak = np.maximum(peak, wind)

    result: FloatArray = peak
    return result


def wind_risk_index(
    peak_wind_ms: npt.ArrayLike,
    *,
    damage_threshold_ms: float = 25.0,
    saturation_ms: float = 70.0,
) -> FloatArray:
    """Normalise peak wind to a [0, 1] index for the risk engine.

    Zero below ``damage_threshold_ms``, which is deliberate rather than a
    convenience: structural damage to ordinary construction begins around
    gale-to-storm force, and a linear scale from zero would assign nonzero
    "risk" to a breezy day. The response between threshold and saturation is
    quadratic because wind load scales with the square of speed, so a linear
    index would badly understate the difference between a Category 1 and a
    Category 4.
    """
    if damage_threshold_ms >= saturation_ms:
        raise ValidationError(
            f"damage threshold ({damage_threshold_ms}) must be below saturation ({saturation_ms})"
        )

    wind = np.asarray(peak_wind_ms, dtype=np.float64)
    scaled = (wind - damage_threshold_ms) / (saturation_ms - damage_threshold_ms)
    result: FloatArray = np.clip(scaled, 0.0, 1.0) ** 2
    return result
