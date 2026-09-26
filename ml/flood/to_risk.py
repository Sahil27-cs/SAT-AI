"""Feed the trained flood model's output into the risk engine.

    python -m ml.flood.to_risk \
        --probability data/processed/flood/India_*_probability.tif \
        --exposure <an exposure raster> \
        --out ml/experiments/flood_risk

Until now the two halves of this project did not touch: `ml/flood/predict.py`
produced a calibrated per-pixel water probability, and `satai/risk/engine.py`
computed ``R = H^a * E^b * V^g`` over hazard fields that came from somewhere
else. This module is the join, and it exists as its own file because the join
has two decisions in it that are easy to get wrong silently.

**The probability is not percentile-normalised.** `RiskEngine.compute` offers
``normalise=True``, which rescales a field between robust percentiles. That is
the right treatment for an *index* whose absolute value carries no meaning. It
is the wrong treatment for a model probability: 0.9 means the model puts 90 %
on water, and rescaling it so that the chip's own 98th percentile becomes 1.0
would turn a scene with no flooding into a scene with maximal hazard. So the
hazard field is passed through with ``normalise=False`` and only clipped.

**Unobserved pixels stay unobserved.** Sen1Floods11 leaves about a third of
each chip unlabelled, and the model still emits a number there. Those pixels
are masked to NaN rather than to zero: zero is a claim that the ground is dry.

What this does not do
---------------------
It does not invent exposure. ``E`` is a real quantity -- people or assets per
cell -- and this repository has ingested no exposure raster, so `--exposure`
is required and the command reports BLOCKED without it rather than substituting
a constant. A constant exposure field would reduce the risk map to a rescaling
of the hazard map while *looking* like a multi-factor result, which is the
single most misleading thing this file could produce.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from satai.paths import REPO_ROOT, relative_to_repo
except ImportError:  # pragma: no cover - outside an installed package
    _candidates: list[Path] = []
    if os.environ.get("SATAI_REPO_ROOT"):
        _candidates.append(Path(os.environ["SATAI_REPO_ROOT"]).expanduser().resolve())
    with contextlib.suppress(NameError):
        _candidates.append(Path(__file__).resolve().parents[2])
    _cwd = Path.cwd().resolve()
    _candidates += [_cwd, *_cwd.parents, *(p for p in sorted(_cwd.iterdir()) if p.is_dir())]
    REPO_ROOT = next((c for c in _candidates if (c / "satai" / "provenance.py").is_file()), _cwd)
    sys.path.insert(0, str(REPO_ROOT))

    # The root is on the path now, so the real helper is importable. Importing it
    # here rather than reimplementing it keeps one definition of what a recorded
    # path looks like.
    from satai.paths import relative_to_repo

import numpy as np
import numpy.typing as npt

from satai.risk.engine import RiskEngine, RiskResult

FloatArray = npt.NDArray[np.float64]

DEFAULT_OUT = REPO_ROOT / "ml" / "experiments" / "flood_risk"
BLOCKED_NO_EXPOSURE = (
    "BLOCKED: no exposure raster. R = H^a * E^b * V^g needs E, and this "
    "repository has ingested no population or asset layer. Supply one with "
    "--exposure (co-registered to the probability raster) or the risk field "
    "cannot be computed. Nothing is substituted."
)


def flood_hazard_field(
    probability: FloatArray, valid: npt.NDArray[np.bool_] | None = None
) -> FloatArray:
    """The model probability as a hazard field: clipped, never rescaled.

    `valid` marks the pixels the label set annotates. Everything outside it
    becomes NaN, because the model's output there is an extrapolation onto
    ground nobody checked -- and NaN propagates into the risk field, where a
    zero would have read as "dry".
    """
    hazard = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != hazard.shape:
            raise ValueError(f"valid mask {mask.shape} does not match probability {hazard.shape}")
        hazard = np.where(mask, hazard, np.nan)
    return hazard


def couple(
    engine: RiskEngine,
    probability: FloatArray,
    exposure: FloatArray,
    *,
    vulnerability: FloatArray | None = None,
    valid: npt.NDArray[np.bool_] | None = None,
) -> RiskResult:
    """Run the risk engine with the flood model as the flood hazard.

    ``normalise=False`` is the load-bearing argument here: see the module
    docstring. Exposure and vulnerability arrive already normalised to [0, 1]
    by whoever ingested them, which is where that decision belongs -- their
    units are not this module's business.
    """
    hazard = flood_hazard_field(probability, valid)
    return engine.compute(
        {"flood": hazard},
        exposure=np.clip(np.asarray(exposure, dtype=np.float64), 0.0, 1.0),
        vulnerability=None
        if vulnerability is None
        else np.clip(np.asarray(vulnerability, dtype=np.float64), 0.0, 1.0),
        normalise=False,
    )


def _read_raster(path: Path) -> tuple[FloatArray, dict[str, Any]]:
    import rasterio

    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float64)
        profile = src.profile.copy()
    return array, profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probability", type=Path, required=True, help="a probability GeoTIFF from predict.py"
    )
    parser.add_argument(
        "--exposure", type=Path, default=None, help="exposure raster, co-registered, in [0, 1]"
    )
    parser.add_argument("--vulnerability", type=Path, default=None)
    parser.add_argument("--risk-config", type=Path, default=REPO_ROOT / "configs" / "risk.yaml")
    parser.add_argument("--couplings", type=Path, default=REPO_ROOT / "configs" / "couplings.yaml")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if args.exposure is None:
        print(BLOCKED_NO_EXPOSURE)
        return 2

    probability, profile = _read_raster(args.probability)
    exposure, exposure_profile = _read_raster(args.exposure)
    if exposure.shape != probability.shape:
        raise SystemExit(
            f"exposure {exposure.shape} is not co-registered with the probability "
            f"raster {probability.shape}; resample it first rather than letting "
            f"the engine broadcast two different grids"
        )
    if exposure_profile["crs"] != profile["crs"]:
        raise SystemExit(
            f"exposure CRS {exposure_profile['crs']} != probability CRS {profile['crs']}"
        )

    vulnerability = None
    if args.vulnerability is not None:
        vulnerability, _ = _read_raster(args.vulnerability)

    engine = RiskEngine.from_configs(args.risk_config, args.couplings)
    valid = np.isfinite(probability)
    result = couple(engine, probability, exposure, vulnerability=vulnerability, valid=valid)

    risk = result.risk_vector["flood"]
    summary = {
        "mean": float(np.nanmean(risk)),
        "p90": float(np.nanpercentile(risk, 90)),
        "max": float(np.nanmax(risk)),
    }
    scored = int(np.isfinite(risk).sum())
    unobserved = int((~valid).sum())
    payload: dict[str, Any] = {
        "experiment": "flood_risk_coupling",
        "run_at": datetime.now(UTC).isoformat(),
        "probability_raster": relative_to_repo(args.probability),
        "exposure_raster": relative_to_repo(args.exposure),
        "vulnerability_raster": (
            None if args.vulnerability is None else relative_to_repo(args.vulnerability)
        ),
        "source_kind": "index",
        "hazard_source_kind": "model",
        "config_hash": result.config_hash,
        "formulation": f"R = H^{engine.config.alpha} * E^{engine.config.beta} "
        f"* V^{engine.config.gamma}",
        "hazard_normalised": False,
        "n_pixels_scored": scored,
        "n_pixels_unobserved": unobserved,
        "risk": summary,
        "band_fractions": result.band_fractions("flood"),
        "caveats": [
            *result.caveats,
            "The hazard term is a MODEL probability and is not percentile-"
            "normalised; the risk score is INDEX-DERIVED and is not a forecast.",
            "Risk bands are percentiles of this raster, not absolute thresholds. "
            "A RED cell is in this scene's top decile.",
            "Not an official warning. Warnings for India come from IMD, NDMA and "
            "the State Disaster Management Authorities.",
        ],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"{args.probability.stem}_risk.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"  pixels scored     {scored:,}")
    print(f"  pixels unobserved {unobserved:,}")
    print(f"  mean risk         {summary['mean']:.4f}")
    print(f"  p90 risk          {summary['p90']:.4f}")
    print(f"\nwritten to {relative_to_repo(destination)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
