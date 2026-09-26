"""Wildfire: burn severity, fire-danger indices and FIRMS detection summaries.

Three things live here and they are not the same kind of thing:

``dnbr`` / ``classify_severity``
    Burn severity from pre- and post-fire reflectance. Deterministic arithmetic
    over observed bands, with published USGS break points. An index, not a
    prediction.

``fire_danger_index``
    A documented composite over weather and fuel-state proxies. It says
    conditions are conducive to fire spreading; it does not say a fire will
    start. SAT-AI performs no ignition prediction.

``summarise_detections``
    Aggregation of NASA FIRMS active-fire records. These are **instrument
    measurements at satellite overpass time** and are the one hazard stream in
    the project that is a direct observation. They are never a census: cloud,
    small fires and fires between overpasses are all missed, and in India a
    large share of October-November detections are agricultural residue burning
    rather than forest fire.

The three are kept apart because collapsing them is the specific way a wildfire
dashboard starts lying. A burned-area map, a danger score and a thermal anomaly
answer different questions, and a user who reads a danger score as a detection
has been misled by the interface rather than by the data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from satai.errors import ValidationError
from satai.logging import get_logger
from satai.provenance import (
    HazardType,
    ProvenanceEnvelope,
    SourceKind,
    SpatialRef,
    TemporalValidity,
    envelope,
)

log = get_logger(__name__)

__all__ = [
    "SEVERITY_BREAKS",
    "SEVERITY_CLASSES",
    "BurnSeverity",
    "classify_severity",
    "dnbr",
    "fire_danger_index",
    "nbr",
    "summarise_detections",
]

FloatArray = npt.NDArray[np.floating]

#: USGS / Key & Benson dNBR severity break points, scaled by 1000 as published.
#: These are the standard operational thresholds and they are **not tuned to
#: Indian vegetation**. Chir pine and dry deciduous forest differ from the North
#: American conifer stands the breaks were derived over, so a class boundary
#: here is a published convention applied out of domain, not a measured one.
#: That caveat travels on every envelope this module produces.
SEVERITY_BREAKS: tuple[float, ...] = (-100.0, 99.0, 269.0, 439.0, 659.0)

SEVERITY_CLASSES: tuple[str, ...] = (
    "enhanced_regrowth",
    "unburned",
    "low",
    "moderate_low",
    "moderate_high",
    "high",
)


def nbr(nir: npt.ArrayLike, swir: npt.ArrayLike) -> FloatArray:
    """Normalised Burn Ratio, ``(NIR - SWIR) / (NIR + SWIR)``.

    Live vegetation reflects strongly in the near infrared and weakly in the
    short-wave infrared; burning inverts that, so the ratio drops sharply. For
    Sentinel-2 this is bands 8 (or 8A) and 12.

    A zero denominator means no signal in either band -- cloud shadow, water, a
    masked pixel -- and yields NaN rather than 0.0, because 0.0 is a legitimate
    NBR value meaning "equal reflectance" and would be indistinguishable from
    missing data downstream.
    """
    nir_a = np.asarray(nir, dtype=np.float64)
    swir_a = np.asarray(swir, dtype=np.float64)
    if nir_a.shape != swir_a.shape:
        raise ValidationError(f"NIR shape {nir_a.shape} does not match SWIR {swir_a.shape}")

    denominator = nir_a + swir_a
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denominator == 0.0, np.nan, (nir_a - swir_a) / denominator)
    result: FloatArray = np.asarray(ratio, dtype=np.float64)
    return result


def dnbr(pre_nbr: npt.ArrayLike, post_nbr: npt.ArrayLike, *, scaled: bool = True) -> FloatArray:
    """Differenced NBR: ``pre - post``, positive where vegetation was lost.

    Parameters
    ----------
    scaled:
        Multiply by 1000, which is how the USGS break points in
        :data:`SEVERITY_BREAKS` are published. Left on by default so the output
        and the thresholds are in the same units -- mixing them is an easy and
        silent way to classify everything as unburned.
    """
    pre = np.asarray(pre_nbr, dtype=np.float64)
    post = np.asarray(post_nbr, dtype=np.float64)
    if pre.shape != post.shape:
        raise ValidationError(f"pre-fire shape {pre.shape} does not match post-fire {post.shape}")

    difference = pre - post
    result: FloatArray = difference * 1000.0 if scaled else difference
    return result


@dataclass(frozen=True)
class BurnSeverity:
    """A classified burn-severity field and the share of area in each class."""

    classes: npt.NDArray[np.str_]
    fractions: dict[str, float]
    n_valid: int
    n_total: int

    @property
    def burned_fraction(self) -> float:
        """Share of valid pixels in any class above ``unburned``."""
        burned = ("low", "moderate_low", "moderate_high", "high")
        return sum(self.fractions.get(c, 0.0) for c in burned)


def classify_severity(dnbr_values: npt.ArrayLike) -> BurnSeverity:
    """Bin a scaled dNBR field into the published severity classes.

    Pixels that are NaN -- cloud, shadow, water, no data -- are excluded from
    the denominator rather than counted as unburned. Counting them as unburned
    would make a cloudy scene look like a scene with no fire, which is the
    opposite of what the data supports.
    """
    array = np.asarray(dnbr_values, dtype=np.float64)
    valid = np.isfinite(array)
    n_total = int(array.size)
    n_valid = int(valid.sum())

    if n_valid == 0:
        raise ValidationError(
            "no finite dNBR values: the scene is entirely masked, so no severity "
            "statement can be made about it"
        )

    # np.digitize with the published breaks: index 0 is below the first break
    # (enhanced regrowth), the last index is above the final break (high).
    indices = np.digitize(array, SEVERITY_BREAKS)
    classes = np.asarray(SEVERITY_CLASSES, dtype="<U18")[indices]
    classes = np.where(valid, classes, "no_data")

    counts = {name: float(np.sum((classes == name) & valid) / n_valid) for name in SEVERITY_CLASSES}
    return BurnSeverity(
        classes=classes.astype("<U18"),
        fractions=counts,
        n_valid=n_valid,
        n_total=n_total,
    )


def fire_danger_index(
    *,
    temperature_c: npt.ArrayLike,
    relative_humidity: npt.ArrayLike,
    wind_speed_ms: npt.ArrayLike,
    days_since_rain: npt.ArrayLike,
    ndvi: npt.ArrayLike | None = None,
) -> FloatArray:
    """A documented composite fire-danger index in [0, 1].

    **This is not a prediction of ignition.** It says conditions are conducive
    to fire spreading if one starts. SAT-AI does not predict where or when a
    fire will start, and the caveat is attached to every envelope built from
    this.

    The four drivers are normalised onto [0, 1] against operationally reasonable
    ranges and combined multiplicatively, for the same reason the risk engine is
    multiplicative (ADR-008): the boundary condition has to be right. Saturated
    fuel after recent rain suppresses fire danger no matter how hot and windy it
    is, and an additive form gets that wrong -- it would return "moderate"
    danger for a hot windy day over soaked ground.

    ``ndvi`` is an optional fuel-state proxy. It is **excluded when absent, not
    imputed**: assuming average greenness would be inventing a fuel observation.

    Ranges, and why they are these:
        temperature   15-45 C   -- the operational span for the Indian fire season
        humidity      20-80 %   -- inverted; dry air raises danger
        wind          0-15 m/s  -- above this, spread is wind-driven regardless
        days dry      0-30 d    -- fuel moisture recovery timescale
    """

    def _normalise(
        values: npt.ArrayLike, low: float, high: float, *, invert: bool = False
    ) -> FloatArray:
        array = np.clip((np.asarray(values, dtype=np.float64) - low) / (high - low), 0.0, 1.0)
        out: FloatArray = 1.0 - array if invert else array
        return out

    temp = _normalise(temperature_c, 15.0, 45.0)
    humidity = _normalise(relative_humidity, 20.0, 80.0, invert=True)
    wind = _normalise(wind_speed_ms, 0.0, 15.0)
    dryness = _normalise(days_since_rain, 0.0, 30.0)

    shapes = {temp.shape, humidity.shape, wind.shape, dryness.shape}
    if len(shapes) != 1:
        raise ValidationError(f"fire-danger inputs have mismatched shapes: {sorted(shapes)}")

    # A floor keeps one zeroed driver from collapsing the whole index, the same
    # guard the risk engine applies for the same reason.
    floor = 0.05
    index = (
        np.maximum(temp, floor)
        * np.maximum(humidity, floor)
        * np.maximum(wind, floor)
        * np.maximum(dryness, floor)
    )

    if ndvi is not None:
        # Low NDVI means cured or sparse fuel, which raises danger -- hence the
        # inversion. Very low NDVI is bare rock or water and carries no fuel at
        # all, which this simple proxy does NOT distinguish; that limitation is
        # a caveat on the envelope.
        fuel = _normalise(ndvi, 0.1, 0.7, invert=True)
        if fuel.shape != index.shape:
            raise ValidationError(f"NDVI shape {fuel.shape} does not match {index.shape}")
        index = index * np.maximum(fuel, floor)

    # Rescale to [0, 1]: the product of four or five [0, 1] terms is heavily
    # compressed toward zero, and a raw product would render as "no danger
    # anywhere". The exponent is a presentation choice, recorded as such.
    exponent = 1.0 / (5.0 if ndvi is not None else 4.0)
    result: FloatArray = np.clip(np.power(index, exponent), 0.0, 1.0)
    return result


def summarise_detections(
    detections: Sequence[dict[str, Any]],
    *,
    region_id: str | None = None,
    spatial: SpatialRef | None = None,
    window_days: int,
) -> ProvenanceEnvelope[dict[str, float]]:
    """Aggregate FIRMS active-fire records into an OBSERVATION envelope.

    Every caveat here is load-bearing. An empty result means "no detection in
    this window", which is emphatically not "no fire": the sensor sees a pixel
    twice a day at best, cloud blocks the thermal band, and a fire smaller than
    the detection limit is invisible. Reporting zero detections as zero fire is
    the most likely way this module gets misread.

    Fire radiative power is averaged over the records that report it, not over
    all records -- dividing by the full count deflates the mean in proportion to
    how many are missing it, and that deflated number would then be stated as
    fact by the agent layer.
    """
    if window_days <= 0:
        raise ValidationError(f"window_days must be positive, got {window_days}")

    frp = [float(d["frp_mw"]) for d in detections if d.get("frp_mw") is not None]
    high_confidence = sum(
        1 for d in detections if str(d.get("confidence", "")).lower() in {"h", "high", "nominal"}
    )

    values: dict[str, float] = {
        "detection_count": float(len(detections)),
        "high_confidence_count": float(high_confidence),
        "window_days": float(window_days),
    }
    if frp:
        values["mean_frp_mw"] = sum(frp) / len(frp)
        values["max_frp_mw"] = max(frp)
        values["frp_reported_count"] = float(len(frp))

    # Typed explicitly: the comprehension filters out None, but that narrowing
    # is invisible to the checker, and `max` over a list that might contain None
    # is exactly the kind of thing worth keeping the checker able to see.
    observed: list[datetime] = [
        d["observed_at"] for d in detections if isinstance(d.get("observed_at"), datetime)
    ]
    caveats = [
        "Active fire detections are satellite observations at overpass time, "
        "not predictions, and not a complete census of fires.",
        "Absence of detections is not evidence of no fire: cloud cover, fires "
        "smaller than the detection limit, and fires between overpasses are all "
        "undetected.",
        "In India a large share of October-November detections are agricultural "
        "residue burning rather than forest fire.",
        "FIRMS near-real-time latency is approximately 3 hours.",
    ]
    if frp and len(frp) < len(detections):
        caveats.append(
            f"Fire radiative power is reported for {len(frp)} of {len(detections)} "
            f"detections; the mean and maximum cover only those."
        )
    elif not frp and detections:
        caveats.append("No detection in this window reports fire radiative power.")

    return envelope(
        values,
        quantity="active_fire_detections",
        unit="count",
        kind=SourceKind.OBSERVATION,
        source_id="firms_viirs",
        version="1.0.0",
        spatial=spatial,
        temporal=TemporalValidity(observed_at=max(observed) if observed else None),
        hazard=HazardType.WILDFIRE,
        caveats=caveats,
        region_id=region_id,
    )
