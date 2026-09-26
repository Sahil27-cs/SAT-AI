"""Leakage-safe normalisation.

The rule this module enforces: **normalisation statistics are fitted on the
training partition only, and every other partition is transformed with those
same statistics.**

That sounds obvious and is violated constantly, because the convenient thing to
do is compute percentiles over the whole array stack before splitting it. Doing
so lets the test distribution influence the transform applied to the training
data — a small, invisible leak that inflates every reported metric and cannot
be detected from the metrics themselves.

The leak matters more than usual here. Under leave-one-region-out (ADR-009) the
test partition is a *different flood in a different country*, and its backscatter
distribution genuinely differs from the training regions'. Fitting statistics
across all regions would quietly hand the model information about the test
region's radiometry — precisely the domain shift the LORO protocol exists to
measure.

So :class:`BandStats` records the fold it was fitted on, refuses to be fitted
twice, and is serialised alongside model weights so inference applies the same
transform (ADR-010).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from satai.errors import ValidationError
from satai.logging import get_logger

log = get_logger(__name__)

__all__ = ["BandStats", "StackNormalizer", "fit_band_stats"]

FloatArray = npt.NDArray[np.floating]

#: Robust default percentiles. Clipping before standardising stops a handful of
#: extreme pixels — a radar bright target, a sensor artefact — from setting the
#: scale for an entire band.
DEFAULT_LOWER_PERCENTILE = 2.0
DEFAULT_UPPER_PERCENTILE = 98.0


@dataclass(frozen=True)
class BandStats:
    """Per-band statistics fitted on one fold's training partition."""

    band: str
    lower: float
    upper: float
    mean: float
    std: float
    lower_percentile: float
    upper_percentile: float
    n_samples: int
    n_valid: int
    fitted_on_fold: str
    fitted_on_partition: str = "train"

    def __post_init__(self) -> None:
        if self.upper <= self.lower:
            raise ValidationError(
                f"band {self.band!r}: upper clip {self.upper} must exceed lower {self.lower}"
            )
        if self.std <= 0:
            raise ValidationError(
                f"band {self.band!r}: std is {self.std}; a constant band carries no "
                "information and must be dropped rather than normalised"
            )
        if self.fitted_on_partition != "train":
            raise ValidationError(
                f"band {self.band!r}: statistics were fitted on "
                f"{self.fitted_on_partition!r}. Fitting on anything but the "
                "training partition leaks the evaluation distribution into the "
                "transform."
            )

    @property
    def valid_fraction(self) -> float:
        return self.n_valid / self.n_samples if self.n_samples else 0.0

    def apply(self, array: npt.ArrayLike) -> FloatArray:
        """Clip to the fitted range, then standardise. NaN stays NaN."""
        values = np.asarray(array, dtype=np.float64)
        clipped = np.clip(values, self.lower, self.upper)
        return (clipped - self.mean) / self.std

    def invert(self, array: npt.ArrayLike) -> FloatArray:
        """Undo :meth:`apply`, up to the clipping, for inspecting predictions."""
        values = np.asarray(array, dtype=np.float64)
        return values * self.std + self.mean


def fit_band_stats(
    values: npt.ArrayLike,
    *,
    band: str,
    fold: str,
    partition: str = "train",
    lower_percentile: float = DEFAULT_LOWER_PERCENTILE,
    upper_percentile: float = DEFAULT_UPPER_PERCENTILE,
) -> BandStats:
    """Fit clipping bounds and moments for one band.

    Parameters
    ----------
    values:
        Samples from the **training partition only**. NaN is ignored.
    band:
        Band name, recorded so statistics cannot be applied to the wrong band.
    fold:
        Fold identity, recorded so statistics cannot be reused across folds.
    partition:
        Must be ``"train"``. The parameter exists so that a misuse is an
        explicit, rejected argument rather than an undetectable mistake.
    """
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValidationError(
            f"percentiles must satisfy 0 <= lower < upper <= 100; "
            f"got {lower_percentile} and {upper_percentile}"
        )

    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValidationError(f"band {band!r}: no samples to fit statistics on")

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValidationError(f"band {band!r}: every sample is NaN or infinite")
    if finite.size < 100:
        log.warning(
            "fitting normalisation on very few samples",
            extra={"band": band, "fold": fold, "n_valid": int(finite.size)},
        )

    lower = float(np.percentile(finite, lower_percentile))
    upper = float(np.percentile(finite, upper_percentile))
    if upper <= lower:  # a near-constant band
        raise ValidationError(
            f"band {band!r}: the {lower_percentile}th and {upper_percentile}th "
            f"percentiles are both {lower}. The band is constant over the "
            "training partition and carries no information."
        )

    clipped = np.clip(finite, lower, upper)
    return BandStats(
        band=band,
        lower=lower,
        upper=upper,
        mean=float(clipped.mean()),
        std=float(clipped.std()),
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        n_samples=int(array.size),
        n_valid=int(finite.size),
        fitted_on_fold=fold,
        fitted_on_partition=partition,
    )


class StackNormalizer:
    """Per-band statistics for one fold's feature stack.

    Travels with the model: serialised beside the weights, reloaded at inference
    so that operational data is transformed exactly as the training data was
    (ADR-010). Recomputing statistics from AOI data at inference would erase the
    very domain shift the Phase 5 gate exists to detect.
    """

    def __init__(self, fold: str, stats: dict[str, BandStats] | None = None) -> None:
        self.fold = fold
        self._stats: dict[str, BandStats] = dict(stats or {})

    def fit_band(self, band: str, training_values: npt.ArrayLike, **kwargs: Any) -> BandStats:
        """Fit and store statistics for one band.

        Refuses to overwrite: re-fitting a band mid-experiment usually means
        statistics from two different sample sets have been mixed, which is
        exactly the kind of thing that is invisible afterwards.
        """
        if band in self._stats:
            raise ValidationError(
                f"band {band!r} already has statistics for fold {self.fold!r}. "
                "Refusing to overwrite — create a new normalizer instead."
            )
        stats = fit_band_stats(training_values, band=band, fold=self.fold, **kwargs)
        self._stats[band] = stats
        return stats

    def transform(self, band: str, array: npt.ArrayLike) -> FloatArray:
        """Apply the fitted transform for ``band``."""
        if band not in self._stats:
            known = ", ".join(sorted(self._stats)) or "(none)"
            raise ValidationError(
                f"no statistics fitted for band {band!r} in fold {self.fold!r}; have: {known}"
            )
        return self._stats[band].apply(array)

    def transform_stack(self, stack: npt.ArrayLike, bands: list[str]) -> FloatArray:
        """Transform a ``(bands, height, width)`` stack, band by band."""
        array = np.asarray(stack, dtype=np.float64)
        if array.ndim != 3:
            raise ValidationError(f"expected a 3-D (band, y, x) stack; got shape {array.shape}")
        if array.shape[0] != len(bands):
            raise ValidationError(
                f"stack has {array.shape[0]} bands but {len(bands)} names were given: "
                f"{', '.join(bands)}"
            )
        return np.stack([self.transform(b, array[i]) for i, b in enumerate(bands)])

    @property
    def bands(self) -> list[str]:
        return sorted(self._stats)

    def __contains__(self, band: str) -> bool:
        return band in self._stats

    def __getitem__(self, band: str) -> BandStats:
        return self._stats[band]

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "bands": {name: asdict(stats) for name, stats in sorted(self._stats.items())},
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        log.info(
            "normalisation statistics saved",
            extra={"path": str(path), "fold": self.fold, "n_bands": len(self._stats)},
        )
        return path

    @classmethod
    def load(cls, path: Path) -> StackNormalizer:
        if not path.is_file():
            raise ValidationError(f"normalisation statistics not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        stats = {name: BandStats(**values) for name, values in payload["bands"].items()}
        return cls(fold=payload["fold"], stats=stats)

    def assert_fold(self, expected_fold: str) -> None:
        """Fail if these statistics belong to a different fold.

        Called before training and before inference. Applying fold A's
        statistics to fold B's data leaks A's training distribution into B's
        evaluation, and nothing downstream would reveal it.
        """
        if self.fold != expected_fold:
            raise ValidationError(
                f"normalisation statistics were fitted on fold {self.fold!r} but "
                f"are being applied to fold {expected_fold!r}. This would leak one "
                "fold's training distribution into another's evaluation."
            )
