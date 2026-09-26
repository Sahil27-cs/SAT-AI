"""Classical flood-detection baselines.

**These are built and evaluated before any deep model, and that ordering is not
a formality.** Otsu thresholding of SAR backscatter with a terrain mask is the
standard operational method for flood mapping; it is fast, needs no training
data, and is genuinely strong. A U-Net that cannot beat it has not demonstrated
anything, and most student flood projects never find that out because they
never build the baseline.

Two baselines here:

``otsu_hand``
    Unsupervised. Otsu's method finds the threshold minimising within-class
    variance in the VV histogram; HAND (height above nearest drainage) then
    removes the false positives Otsu cannot avoid -- tarmac, sand and dry
    riverbeds are smooth, so they scatter radar away from the sensor exactly
    like water does. Terrain is what tells them apart: water does not sit high
    above drainage.

``pixel_ensemble``
    Supervised per-pixel Random Forest / gradient boosting over the band stack.
    No spatial context at all, which is precisely what makes it the right
    control for a U-Net: the difference between them is the value of spatial
    context, isolated from everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from satai.errors import ValidationError
from satai.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "OtsuHandBaseline",
    "PixelEnsembleBaseline",
    "otsu_threshold",
    "otsu_threshold_and_separability",
]

FloatArray = npt.NDArray[np.floating]
BoolArray = npt.NDArray[np.bool_]


def otsu_threshold_and_separability(
    values: npt.ArrayLike, *, bins: int = 256
) -> tuple[float, float]:
    """Otsu's threshold together with its separability measure eta.

    ``eta`` is the fraction of total variance that the chosen split explains,
    ``sigma^2_between / sigma^2_total``. It is the quantity Otsu maximises, so
    it comes out of the same computation for free, and unlike a statistic
    computed *after* splitting it is not high by construction.

    The reference value that makes it usable: splitting a single Gaussian at
    its own mean explains exactly ``2/pi ~ 0.637`` of the variance, since the
    two truncated halves sit ``sqrt(2/pi)*sigma`` either side of it, and no
    other cut of a unimodal Gaussian does much better. Two well-separated
    modes explain well over 0.9. A cut-off between those -- 0.75 in
    :class:`OtsuHandBaseline` -- therefore separates "there are two classes
    here" from "there is one class and Otsu bisected it".
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    finite = array[np.isfinite(array)]
    if finite.size < 2:
        raise ValidationError("Otsu needs at least two finite values")

    counts, edges = np.histogram(finite, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    total = counts.sum()
    if total == 0:
        raise ValidationError("empty histogram")

    weight_bg = np.cumsum(counts) / total
    weight_fg = 1.0 - weight_bg

    cum_mean = np.cumsum(counts * centres) / total
    global_mean = cum_mean[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        between = (global_mean * weight_bg - cum_mean) ** 2 / (weight_bg * weight_fg)
    between = np.nan_to_num(between, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    best = int(np.argmax(between))
    probabilities = counts / total
    total_variance = float(np.sum(probabilities * (centres - global_mean) ** 2))
    eta = float(between[best] / total_variance) if total_variance > 0 else 0.0

    return float(centres[best]), max(0.0, min(1.0, eta))


def otsu_threshold(values: npt.ArrayLike, *, bins: int = 256) -> float:
    """Otsu's threshold: the cut minimising intra-class variance.

    Implemented directly rather than taken from scikit-image so the histogram
    range can be controlled. SAR dB values span roughly [-35, +5] with a long
    low tail from border noise; letting the histogram auto-range lets that tail
    drag the threshold down until the water class vanishes.

    Assumes bimodality. A chip that is entirely dry has one mode, and Otsu will
    still return a threshold -- splitting the land distribution in half and
    reporting most of it as water. :class:`OtsuHandBaseline` guards against
    this using the separability returned by
    :func:`otsu_threshold_and_separability`.
    """
    return otsu_threshold_and_separability(values, bins=bins)[0]


@dataclass
class OtsuHandBaseline:
    """Unsupervised SAR thresholding with a terrain-based false-positive mask.

    Parameters
    ----------
    hand_threshold_m:
        Pixels more than this far above the nearest drainage cannot plausibly
        be inundated. 15 m is a common operational default; it is a
        *configurable assumption*, not a measurement, and is reported as such.
    min_separability:
        Otsu assumes two modes. Below this value of eta -- the fraction of
        total variance the chosen split explains -- the histogram is treated as
        unimodal (a dry chip) and the baseline predicts no water rather than
        splitting the land distribution in half.

        The reference points: bisecting a single Gaussian explains
        ``2/pi ~ 0.637``, while a land and a water mode separated by the usual
        ~10 dB explain over 0.9. The default sat at 0.75 on that reasoning
        alone until it was measured.

        **0.66 is now measured, not argued.** Scored over all 446 Sen1Floods11
        hand-labelled chips, pooled IoU is extremely sensitive to this value --
        0.184 with the guard off, peaking at 0.420 near 0.66, and collapsing to
        0.031 by 0.85. With the guard off, precision falls to 0.197: Otsu
        bisects every dry chip and reports half of it as water, which is exactly
        the failure this parameter exists to prevent, now observed rather than
        predicted.

        Selection was leave-one-region-out -- for each of the 11 regions the
        value was chosen on the other ten and applied to the held-out one -- so
        this is not the maximum of a curve fitted to the evaluation set. All
        eleven folds independently chose 0.66, so the region-disjoint estimate
        and the all-data optimum coincide to four decimal places. A parameter
        that stable across held-out regions is one worth shipping as a default.

        See ``ml/experiments/flood_baseline/threshold_selection.json``.
    """

    band_index: int = 0  # VV
    hand_threshold_m: float = 15.0
    min_separability: float = 0.66
    bins: int = 256

    fitted_threshold_: float | None = field(default=None, init=False)
    separability_: float | None = field(default=None, init=False)

    name: str = "otsu_hand"
    version: str = "1.0.0"

    def predict(
        self,
        features: FloatArray,
        *,
        hand: FloatArray | None = None,
        valid: BoolArray | None = None,
    ) -> FloatArray:
        """Return a water probability map in {0, 1} (this baseline is hard-thresholded)."""
        if features.ndim != 3:
            raise ValidationError(f"expected (band, y, x); got shape {features.shape}")

        band = features[self.band_index]
        mask = np.isfinite(band) if valid is None else (valid & np.isfinite(band))
        if not mask.any():
            return np.zeros(band.shape, dtype=np.float64)

        values = band[mask]
        threshold, separability = otsu_threshold_and_separability(values, bins=self.bins)
        self.fitted_threshold_ = threshold
        self.separability_ = separability

        if separability < self.min_separability:
            log.debug(
                "histogram looks unimodal; predicting no water",
                extra={"separability": round(separability, 3), "threshold": round(threshold, 2)},
            )
            return np.zeros(band.shape, dtype=np.float64)

        water = (band <= threshold) & mask

        if hand is not None:
            if hand.shape != band.shape:
                raise ValidationError(f"HAND shape {hand.shape} does not match band {band.shape}")
            water &= np.isfinite(hand) & (hand <= self.hand_threshold_m)

        probability: FloatArray = water.astype(np.float64)
        return probability

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "supervised": False,
            "band_index": self.band_index,
            "hand_threshold_m": self.hand_threshold_m,
            "min_separability": self.min_separability,
            "fitted_threshold_db": self.fitted_threshold_,
            "separability": self.separability_,
            "caveats": [
                "Unsupervised: the threshold is re-derived per chip, so it adapts "
                "to local conditions but cannot learn from labels.",
                f"HAND cut-off of {self.hand_threshold_m} m is a configurable "
                "assumption taken from operational practice, not a measurement.",
                "Known failure mode: urban double-bounce RAISES backscatter over "
                "flooded streets, inverting the signature this method depends on.",
            ],
        }


@dataclass
class PixelEnsembleBaseline:
    """Per-pixel Random Forest or gradient boosting over the band stack.

    The control for the deep model. It sees the same bands but **no spatial
    context**, so the U-Net's margin over it is a direct measurement of what
    spatial context is worth on this task -- isolated from architecture,
    optimiser and augmentation, which are identical inputs either way.
    """

    kind: Literal["random_forest", "gradient_boosting"] = "random_forest"
    n_estimators: int = 200
    max_depth: int | None = 16
    min_samples_leaf: int = 8
    class_weight: str | None = "balanced_subsample"
    max_train_pixels: int = 2_000_000
    seed: int = 42

    model_: Any = field(default=None, init=False, repr=False)
    feature_names_: list[str] = field(default_factory=list, init=False)

    name: str = "pixel_ensemble"
    version: str = "1.0.0"

    def fit(
        self,
        features: FloatArray,
        labels: npt.NDArray[np.int_],
        valid: BoolArray,
        *,
        feature_names: list[str] | None = None,
    ) -> PixelEnsembleBaseline:
        """Fit on ``(n_pixels, n_features)`` drawn from the TRAINING fold only."""
        if features.ndim != 2:
            raise ValidationError(f"expected (pixels, features); got {features.shape}")
        if features.shape[0] != labels.shape[0]:
            raise ValidationError(f"{features.shape[0]} feature rows but {labels.shape[0]} labels")

        x = features[valid]
        y = labels[valid]
        if x.shape[0] == 0:
            raise ValidationError("no valid training pixels")
        if len(np.unique(y)) < 2:
            raise ValidationError(
                f"training data has only class {np.unique(y).tolist()}; "
                "a classifier cannot be fitted on one class"
            )

        if x.shape[0] > self.max_train_pixels:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(x.shape[0], self.max_train_pixels, replace=False)
            x, y = x[idx], y[idx]
            log.info("subsampled training pixels", extra={"n": int(self.max_train_pixels)})

        if self.kind == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            self.model_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight=self.class_weight,
                random_state=self.seed,
                n_jobs=-1,
            )
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.model_ = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.seed,
            )

        self.model_.fit(x, y)
        self.feature_names_ = feature_names or [f"band_{i}" for i in range(x.shape[1])]
        log.info(
            "baseline fitted",
            extra={
                "kind": self.kind,
                "n_pixels": int(x.shape[0]),
                "positive_rate": float(y.mean()),
            },
        )
        return self

    def predict_proba(self, features: FloatArray) -> FloatArray:
        """Water probability for ``(n_pixels, n_features)``."""
        if self.model_ is None:
            raise ValidationError("baseline is not fitted")
        finite = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        proba: FloatArray = self.model_.predict_proba(finite)[:, 1]
        return proba

    def feature_importance(self) -> dict[str, float]:
        """Importances, for the explainability layer and for sanity-checking.

        A model whose top feature is not a SAR band is doing something other
        than detecting water, and that is worth seeing before trusting it.
        """
        if self.model_ is None:
            raise ValidationError("baseline is not fitted")
        importances = getattr(self.model_, "feature_importances_", None)
        if importances is None:
            return {}
        return dict(
            sorted(
                zip(self.feature_names_, (float(v) for v in importances), strict=True),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "supervised": True,
            "kind": self.kind,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "class_weight": self.class_weight,
            "seed": self.seed,
            "caveats": [
                "Per-pixel: no spatial context. The gap to a U-Net measures what "
                "spatial context is worth on this task.",
                "Fitted on the training fold only; the test region never "
                "influences the model or its threshold.",
            ],
        }
