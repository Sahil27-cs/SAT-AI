"""Post-event damage: change detection from a pre/post image pair.

Scope, stated precisely. This module detects **change**, not damage. A SAR
log-ratio tells you backscatter differed between two acquisitions; whether that
difference is a collapsed building, a harvested field, a flooded road or a
different incidence angle is not something the arithmetic knows. Calling the
output "damage" without that qualifier is the single most misleading thing this
project could do, so every envelope carries it and the class is named
:class:`ChangeAssessment` rather than ``DamageAssessment``.

SAT-AI produces **no monetary damage estimate**. It has no asset-value data, and
generating a figure without it would be fabrication.

Earthquake note: post-event damage assessment is the *only* earthquake-related
capability in the project. SAT-AI does not predict earthquakes under any
circumstances, and nothing here is a step toward it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from satai.errors import ValidationError
from satai.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "CHANGE_CLASSES",
    "ChangeAssessment",
    "assess_change",
    "log_ratio",
    "optical_change_magnitude",
]

FloatArray = npt.NDArray[np.floating]
BoolArray = npt.NDArray[np.bool_]

#: Ordered from no change to severe. Deliberately about *change magnitude*, not
#: about building damage states -- this is not an EMS-98 damage grade and must
#: not be displayed as one.
CHANGE_CLASSES: tuple[str, ...] = ("none", "low", "moderate", "high")


def log_ratio(pre: npt.ArrayLike, post: npt.ArrayLike, *, already_db: bool = True) -> FloatArray:
    """Change magnitude between two SAR acquisitions.

    In decibels the ratio is a difference, which is why SAR change detection is
    done in dB: speckle is multiplicative in linear power and becomes additive
    under the log, so a difference of dB values has roughly stationary noise
    while a ratio of linear powers does not.

    Parameters
    ----------
    already_db:
        True when the inputs are backscatter in dB (the normal case in this
        project -- see ``satai.preprocessing``). False converts from linear
        power first. Getting this wrong silently produces a field that looks
        plausible and is meaningless, so it is explicit rather than sniffed.
    """
    pre_a = np.asarray(pre, dtype=np.float64)
    post_a = np.asarray(post, dtype=np.float64)
    if pre_a.shape != post_a.shape:
        raise ValidationError(f"pre shape {pre_a.shape} does not match post {post_a.shape}")

    if already_db:
        return np.asarray(post_a - pre_a, dtype=np.float64)

    if np.any((pre_a <= 0) | (post_a <= 0)):
        log.warning("non-positive linear power values; those cells become NaN")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = 10.0 * np.log10(
            np.where(post_a > 0, post_a, np.nan) / np.where(pre_a > 0, pre_a, np.nan)
        )
    result: FloatArray = np.asarray(ratio, dtype=np.float64)
    return result


def optical_change_magnitude(pre: npt.ArrayLike, post: npt.ArrayLike) -> FloatArray:
    """Euclidean spectral distance between pre- and post-event optical stacks.

    Inputs are ``(band, y, x)``. Sensitive to illumination and phenology as well
    as to physical change, which is why optical change over a months-long gap
    says more about the season than about the event.
    """
    pre_a = np.asarray(pre, dtype=np.float64)
    post_a = np.asarray(post, dtype=np.float64)
    if pre_a.shape != post_a.shape:
        raise ValidationError(f"pre shape {pre_a.shape} does not match post {post_a.shape}")
    if pre_a.ndim != 3:
        raise ValidationError(f"expected (band, y, x); got shape {pre_a.shape}")

    result: FloatArray = np.sqrt(np.nansum((post_a - pre_a) ** 2, axis=0))
    return result


@dataclass(frozen=True)
class ChangeAssessment:
    """Classified change magnitude over a pre/post pair.

    ``changed_fraction`` is a share of *valid* pixels. Masked pixels are
    excluded from the denominator rather than counted as unchanged, because a
    half-clouded scene reported as "3 % changed" understates the change over the
    part that was actually seen.
    """

    classes: npt.NDArray[np.str_]
    magnitude: FloatArray
    fractions: dict[str, float]
    n_valid: int
    n_total: int
    threshold_db: float

    @property
    def changed_fraction(self) -> float:
        return sum(self.fractions.get(c, 0.0) for c in ("low", "moderate", "high"))

    @property
    def coverage(self) -> float:
        """Share of the scene that carried usable data."""
        return self.n_valid / self.n_total if self.n_total else 0.0

    def caveats(self) -> list[str]:
        """What must travel with this result wherever it is displayed."""
        out = [
            "Change detected from imagery, not verified on the ground.",
            "This is a change-magnitude field, not a building damage grade. "
            "Backscatter change can be structural damage, flooding, harvest, "
            "or a different acquisition geometry.",
            "No monetary damage estimate is produced: SAT-AI has no asset-value data.",
            f"Change threshold is {self.threshold_db} dB, a configurable assumption "
            f"from operational practice rather than a calibrated value.",
        ]
        if self.coverage < 0.8:
            out.append(
                f"Only {self.coverage:.0%} of the scene carried usable data; the "
                f"remainder was masked and is excluded from all fractions."
            )
        return out


def assess_change(
    change: npt.ArrayLike,
    *,
    threshold_db: float = 3.0,
    valid: BoolArray | None = None,
) -> ChangeAssessment:
    """Classify a change field into magnitude bands.

    ``threshold_db`` is the point below which change is treated as noise. 3 dB
    is a common operational starting point and it is an *assumption*, reported
    as one -- speckle in a single-look SAR pair routinely reaches a couple of
    dB, so a lower threshold would report speckle as damage.

    Bands are multiples of the threshold: below it is ``none``, up to 2x is
    ``low``, up to 3x ``moderate``, beyond ``high``. Absolute magnitude is used,
    so both brightening and darkening count -- a collapsed building can raise
    backscatter through rubble scattering or lower it by removing a double-bounce
    corner, and looking only for a decrease would miss half the damage.
    """
    if threshold_db <= 0:
        raise ValidationError(f"threshold must be positive, got {threshold_db}")

    array = np.abs(np.asarray(change, dtype=np.float64))
    mask = np.isfinite(array) if valid is None else (valid & np.isfinite(array))

    n_total = int(array.size)
    n_valid = int(mask.sum())
    if n_valid == 0:
        raise ValidationError(
            "no valid pixels in the change field: the pair is entirely masked, so "
            "no change statement can be made about it"
        )

    breaks = [threshold_db, 2.0 * threshold_db, 3.0 * threshold_db]
    indices = np.digitize(array, breaks)
    classes = np.asarray(CHANGE_CLASSES, dtype="<U10")[indices]
    classes = np.where(mask, classes, "no_data")

    fractions = {name: float(np.sum((classes == name) & mask) / n_valid) for name in CHANGE_CLASSES}
    return ChangeAssessment(
        classes=classes.astype("<U10"),
        magnitude=array,
        fractions=fractions,
        n_valid=n_valid,
        n_total=n_total,
        threshold_db=threshold_db,
    )
