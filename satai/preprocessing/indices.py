"""Band math: SAR ratios and optical indices.

Pure array functions, no I/O, so each one is independently testable — which
matters because a sign error or a unit mistake here produces a plausible-looking
raster that silently poisons every downstream model.

**The unit trap this module exists to prevent.** Sentinel-1 backscatter is
normally distributed in **decibels**, and a decibel is already a logarithm. The
VV/VH *ratio* in linear power is therefore a **difference** in dB::

    10·log₁₀(VV_linear / VH_linear) = VV_dB − VH_dB

Dividing two dB arrays is a common and quiet error: the result has no physical
meaning, varies wildly near zero, and still looks like a reasonable image. Every
function below states its expected units in its signature and its docstring, and
the dB-domain functions refuse to be used on linear data where that is
detectable.

Nodata is propagated as NaN throughout rather than as a sentinel value. A
sentinel such as 0 or -9999 that survives into a normalisation step will drag
the statistics with it; NaN cannot be averaged by accident.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from satai.errors import ValidationError

__all__ = [
    "db_to_linear",
    "dnbr",
    "linear_to_db",
    "log_ratio_db",
    "mndwi",
    "nbr",
    "ndvi",
    "ndwi",
    "normalized_difference",
    "sar_ratio_db",
]

FloatArray = npt.NDArray[np.floating]

#: Physically implausible for calibrated Sentinel-1 γ⁰/σ⁰ in dB, which sits
#: roughly in [-35, +5]. Used only to catch linear data passed as dB.
_DB_PLAUSIBLE_MIN = -60.0
_DB_PLAUSIBLE_MAX = 30.0


def _as_float(array: npt.ArrayLike, name: str) -> FloatArray:
    result = np.asarray(array, dtype=np.float64)
    if result.size == 0:
        raise ValidationError(f"{name} is empty")
    return result


def _check_same_shape(a: FloatArray, b: FloatArray, a_name: str, b_name: str) -> None:
    if a.shape != b.shape:
        raise ValidationError(
            f"{a_name} and {b_name} must have the same shape; got {a.shape} and {b.shape}"
        )


def _warn_if_not_db(array: FloatArray, name: str) -> None:
    """Refuse obviously-linear data where dB is required.

    Linear backscatter is non-negative and typically well under 1. If every
    finite value sits in [0, 1] the caller has almost certainly passed linear
    power to a dB-domain function, which would produce a meaningless result
    without raising anything.
    """
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return
    if float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
        raise ValidationError(
            f"{name} looks like linear power (all values in [0, 1]), not decibels. "
            f"Convert with linear_to_db() first — dividing or differencing linear "
            f"values as if they were dB is a silent error."
        )
    if float(finite.min()) < _DB_PLAUSIBLE_MIN or float(finite.max()) > _DB_PLAUSIBLE_MAX:
        raise ValidationError(
            f"{name} has values outside the plausible dB range "
            f"[{_DB_PLAUSIBLE_MIN}, {_DB_PLAUSIBLE_MAX}]: "
            f"[{float(finite.min()):.1f}, {float(finite.max()):.1f}]"
        )


# --- unit conversion ------------------------------------------------------


def linear_to_db(linear: npt.ArrayLike, *, floor: float = 1e-10) -> FloatArray:
    """Convert linear backscatter power to decibels.

    Values at or below ``floor`` become NaN rather than -inf: a negative
    infinity survives arithmetic and quietly contaminates whatever it touches,
    whereas NaN is designed to be detected.
    """
    values = _as_float(linear, "linear")
    safe = np.where(values > floor, values, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 10.0 * np.log10(safe)


def db_to_linear(db: npt.ArrayLike) -> FloatArray:
    """Convert decibels to linear backscatter power."""
    result: FloatArray = np.power(10.0, _as_float(db, "db") / 10.0)
    return result


# --- SAR ------------------------------------------------------------------


def sar_ratio_db(
    vv_db: npt.ArrayLike, vh_db: npt.ArrayLike, *, check_units: bool = True
) -> FloatArray:
    """Co- to cross-polarisation ratio, computed correctly in the dB domain.

    Returns ``VV_dB − VH_dB``, which equals ``10·log₁₀(VV_linear / VH_linear)``.

    The ratio discriminates surface scattering (open water, bare soil: VV ≫ VH)
    from volume scattering (vegetation canopy: VV ≈ VH), which is why it earns a
    place in the flood feature stack alongside the raw polarisations.

    Parameters
    ----------
    vv_db, vh_db:
        Calibrated backscatter **in decibels**.
    check_units:
        Validate that the inputs plausibly are dB. Disable only when the guard
        is known to be wrong for the data at hand.
    """
    vv = _as_float(vv_db, "vv_db")
    vh = _as_float(vh_db, "vh_db")
    _check_same_shape(vv, vh, "vv_db", "vh_db")
    if check_units:
        _warn_if_not_db(vv, "vv_db")
        _warn_if_not_db(vh, "vh_db")
    return vv - vh


def log_ratio_db(
    pre_db: npt.ArrayLike, post_db: npt.ArrayLike, *, check_units: bool = True
) -> FloatArray:
    """Bi-temporal change: ``post_dB − pre_dB``.

    The primary flood-change feature. Open water reflects radar away from the
    sensor, so new inundation appears as a strong **negative** change — a drop
    of roughly 5–15 dB over previously dry land.

    Sign convention is fixed and deliberate: negative means darkening, which
    means (probably) water. Flipping it would invert every downstream
    interpretation while leaving the magnitudes intact, so the error would
    survive any sanity check based on histograms.

    **Restrict pairs to a single relative orbit.** Scenes from different orbits
    view the ground at different incidence angles, so their difference mixes
    inundation with viewing geometry. ``SceneRef.relative_orbit`` carries the
    information needed to enforce this.
    """
    pre = _as_float(pre_db, "pre_db")
    post = _as_float(post_db, "post_db")
    _check_same_shape(pre, post, "pre_db", "post_db")
    if check_units:
        _warn_if_not_db(pre, "pre_db")
        _warn_if_not_db(post, "post_db")
    return post - pre


# --- optical --------------------------------------------------------------


def normalized_difference(a: npt.ArrayLike, b: npt.ArrayLike, *, eps: float = 1e-10) -> FloatArray:
    """``(a − b) / (a + b)``, the form behind every normalised index.

    Where the denominator is within ``eps`` of zero the result is NaN rather
    than a large arbitrary number. Both bands being near zero means no signal,
    and the honest encoding of no signal is missing data.
    """
    x = _as_float(a, "a")
    y = _as_float(b, "b")
    _check_same_shape(x, y, "a", "b")

    denominator = x + y
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(np.abs(denominator) > eps, (x - y) / denominator, np.nan)
    return np.asarray(result, dtype=np.float64)


def ndwi(green: npt.ArrayLike, nir: npt.ArrayLike) -> FloatArray:
    """Normalised Difference Water Index (McFeeters 1996): ``(G − NIR)/(G + NIR)``.

    Sentinel-2: B3 and B8. Water is strongly positive. Known weakness: built-up
    surfaces also score positive, which is why MNDWI usually does better in
    urban settings — the Mumbai AOI in particular.
    """
    return normalized_difference(green, nir)


def mndwi(green: npt.ArrayLike, swir1: npt.ArrayLike) -> FloatArray:
    """Modified NDWI (Xu 2006): ``(G − SWIR1)/(G + SWIR1)``.

    Sentinel-2: B3 and B11. Suppresses the built-up false positives that affect
    NDWI, at the cost of B11's 20 m native resolution.
    """
    return normalized_difference(green, swir1)


def ndvi(red: npt.ArrayLike, nir: npt.ArrayLike) -> FloatArray:
    """Normalised Difference Vegetation Index: ``(NIR − R)/(NIR + R)``.

    Sentinel-2: B8 and B4. Used as a land-cover covariate for flood work and as
    a fuel-state proxy for fire danger.
    """
    return normalized_difference(nir, red)


def nbr(nir: npt.ArrayLike, swir2: npt.ArrayLike) -> FloatArray:
    """Normalised Burn Ratio: ``(NIR − SWIR2)/(NIR + SWIR2)``.

    Sentinel-2: B8 and B12. Healthy vegetation is high; burnt ground is low.
    """
    return normalized_difference(nir, swir2)


def dnbr(pre_nbr: npt.ArrayLike, post_nbr: npt.ArrayLike) -> FloatArray:
    """Differenced NBR: ``pre − post``. Higher means more severe burning.

    Note the sign convention runs opposite to :func:`log_ratio_db`, and that is
    not an inconsistency — it is what each field's literature uses. dNBR is
    conventionally pre-minus-post so that severity increases upward; SAR log
    ratio is conventionally post-minus-pre so that darkening is negative.
    Both are stated here because assuming either one is how sign errors enter.

    **Thresholds are transferred, not calibrated.** Severity classes come from
    the literature and have not been validated against Indian field data; this
    is recorded in ``docs/limitations.md``.
    """
    pre = _as_float(pre_nbr, "pre_nbr")
    post = _as_float(post_nbr, "post_nbr")
    _check_same_shape(pre, post, "pre_nbr", "post_nbr")
    return pre - post
