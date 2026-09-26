"""Track A / Track B distribution comparison gate (ADR-010).

The risk this exists to catch: SAT-AI trains on Sen1Floods11 chips that their
authors calibrated and terrain-corrected (Track A), and runs operationally on a
stack this project builds from raw Sentinel-1 GRD (Track B). If Track B's
radiometry differs materially from Track A's, the model is being asked to
generalise across a domain shift nobody measured. It will produce confident,
plausible, wrong flood masks, and **nothing in the pipeline will raise an
error** -- the output is a valid-looking probability map either way.

So the mismatch is measured, not assumed, and the measurement gates Phase 5.

Two complementary statistics, because each misses something the other catches:

* **Kolmogorov–Smirnov** -- the largest gap between the two empirical CDFs.
  Sensitive to a shift anywhere in the distribution, scale-free, but it says
  nothing about *how far apart* the distributions are in the units that matter.
* **Wasserstein-1 (earth mover's)** -- the mean distance mass has to travel.
  In dB, and therefore directly interpretable: a value of 2.0 for a VV band
  means the two distributions are on average 2 dB apart, which is a quarter of
  the typical land/water separation and very much enough to break a decision
  boundary.

KS p-values are reported but deliberately **not** used as the gate. With
millions of pixels, any real difference is significant, so the p-value only
restates the sample size. The effect size is what matters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from satai.errors import DataQualityError, ValidationError
from satai.logging import get_logger

log = get_logger(__name__)

__all__ = ["BandComparison", "DistributionGateReport", "compare_band", "run_gate"]

#: Wasserstein thresholds in dB for SAR bands. Derived from the physics rather
#: than from convention: land-to-water separation in Sentinel-1 VV is typically
#: 8-12 dB, so a systematic offset of 1 dB is ~10 % of the signal the model
#: keys on (tolerable, worth noting), and 3 dB is ~30 % (not tolerable).
WARN_WASSERSTEIN_DB = 1.0
FAIL_WASSERSTEIN_DB = 3.0

#: KS statistic thresholds. Scale-free, so these apply to any band.
WARN_KS = 0.10
FAIL_KS = 0.25

QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


@dataclass(frozen=True)
class BandComparison:
    """How far apart one band's Track A and Track B distributions are."""

    band: str
    unit: str

    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    quantiles_a: dict[str, float]
    quantiles_b: dict[str, float]

    ks_statistic: float
    ks_pvalue: float
    wasserstein: float
    mean_shift: float
    std_ratio: float

    verdict: str  # pass | warn | fail
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def summary(self) -> str:
        return (
            f"{self.band:<12} {self.verdict.upper():<5} "
            f"KS={self.ks_statistic:.3f} W1={self.wasserstein:.3f}{self.unit} "
            f"mean_shift={self.mean_shift:+.3f} std_ratio={self.std_ratio:.2f}"
        )


@dataclass
class DistributionGateReport:
    """The full Track A vs Track B comparison, and whether Phase 5 may proceed."""

    created_at: str
    track_a_source: str
    track_b_source: str
    bands: list[BandComparison] = field(default_factory=list)
    git_sha: str | None = None

    @property
    def verdict(self) -> str:
        """Worst band decides. A single broken band breaks the model."""
        verdicts = {b.verdict for b in self.bands}
        if "fail" in verdicts:
            return "fail"
        if "warn" in verdicts:
            return "warn"
        return "pass" if verdicts else "empty"

    @property
    def may_proceed(self) -> bool:
        return self.verdict in {"pass", "warn"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "track_a_source": self.track_a_source,
            "track_b_source": self.track_b_source,
            "git_sha": self.git_sha,
            "verdict": self.verdict,
            "may_proceed": self.may_proceed,
            "bands": [asdict(b) for b in self.bands],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def report(self) -> str:
        lines = [
            "=" * 76,
            "  Track A / Track B distribution gate (ADR-010)",
            "=" * 76,
            f"  Track A (training):  {self.track_a_source}",
            f"  Track B (inference): {self.track_b_source}",
            "",
        ]
        lines.extend(f"  {b.summary()}" for b in self.bands)
        lines.append("")
        lines.append(f"  VERDICT: {self.verdict.upper()}")

        if self.verdict == "fail":
            lines += [
                "",
                "  Phase 5 is BLOCKED. The inference stack does not match the",
                "  training distribution, so any flood mask it produces would be",
                "  out of distribution in a way the model cannot signal.",
                "",
                "  Investigate in this order -- these are the usual causes:",
                "    1. Calibration target: is Track B producing gamma0 or sigma0,",
                "       and which does Track A use?",
                "    2. Units: dB vs linear power. A linear/dB mix-up produces a",
                "       spectacular mismatch and is the first thing to rule out.",
                "    3. Speckle filter: kind and window size change the variance",
                "       much more than the mean, so check the sigma-ratio column.",
                "    4. Terrain correction DEM: a different DEM shifts geometry,",
                "       not radiometry, so this shows up as a spatial rather than",
                "       a distributional mismatch.",
                "    5. Nodata: border noise left unmasked drags the low tail down.",
            ]
        elif self.verdict == "warn":
            lines += [
                "",
                "  Phase 5 may proceed, but the shift is large enough to affect",
                "  the transfer result. Report it alongside C3 -- part of any",
                "  rural-to-urban gap may be preprocessing rather than geography,",
                "  and this number is what lets the two be told apart.",
            ]
        lines.append("=" * 76)
        return "\n".join(lines)


def compare_band(
    track_a: npt.ArrayLike,
    track_b: npt.ArrayLike,
    *,
    band: str,
    unit: str = "dB",
    max_samples: int = 200_000,
    seed: int = 0,
) -> BandComparison:
    """Compare one band's distribution between the two tracks.

    Large arrays are subsampled to ``max_samples`` with a fixed seed: the KS
    test's cost is superlinear, and beyond ~10^5 samples the statistic has long
    since converged while the p-value has become meaningless anyway.
    """
    a = np.asarray(track_a, dtype=np.float64).ravel()
    b = np.asarray(track_b, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size == 0 or b.size == 0:
        raise DataQualityError(
            f"band {band!r}: one track has no finite samples "
            f"(track A: {a.size}, track B: {b.size})",
            band=band,
        )
    if a.size < 100 or b.size < 100:
        log.warning(
            "comparing distributions on very few samples",
            extra={"band": band, "n_a": int(a.size), "n_b": int(b.size)},
        )

    rng = np.random.default_rng(seed)
    a_s = rng.choice(a, max_samples, replace=False) if a.size > max_samples else a
    b_s = rng.choice(b, max_samples, replace=False) if b.size > max_samples else b

    ks = stats.ks_2samp(a_s, b_s)
    wasserstein = float(stats.wasserstein_distance(a_s, b_s))

    mean_a, mean_b = float(a.mean()), float(b.mean())
    std_a, std_b = float(a.std()), float(b.std())

    notes: list[str] = []
    verdict = "pass"

    if unit == "dB":
        if wasserstein >= FAIL_WASSERSTEIN_DB:
            verdict = "fail"
            notes.append(
                f"Wasserstein {wasserstein:.2f} dB is {wasserstein / 10:.0%} of a "
                f"typical land/water separation (~10 dB) -- the decision boundary "
                f"learned on Track A does not transfer."
            )
        elif wasserstein >= WARN_WASSERSTEIN_DB:
            verdict = "warn"
            notes.append(f"Wasserstein {wasserstein:.2f} dB is a material offset.")

    if float(ks.statistic) >= FAIL_KS:
        verdict = "fail"
        notes.append(f"KS statistic {float(ks.statistic):.3f} indicates distinct distributions.")
    elif float(ks.statistic) >= WARN_KS and verdict == "pass":
        verdict = "warn"
        notes.append(f"KS statistic {float(ks.statistic):.3f} shows a visible shift.")

    std_ratio = std_b / std_a if std_a > 0 else float("inf")
    if not 0.5 <= std_ratio <= 2.0:
        if verdict != "fail":
            verdict = "warn"
        notes.append(
            f"Variance ratio {std_ratio:.2f} -- usually a speckle-filter mismatch "
            f"(kind or window size) rather than a calibration difference."
        )

    notes.append(
        "KS p-value is reported but not used as the gate: with this many pixels "
        "any real difference is significant, so it only restates the sample size."
    )

    return BandComparison(
        band=band,
        unit=unit,
        n_a=int(a.size),
        n_b=int(b.size),
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        quantiles_a={f"q{int(q * 100):02d}": float(np.quantile(a, q)) for q in QUANTILES},
        quantiles_b={f"q{int(q * 100):02d}": float(np.quantile(b, q)) for q in QUANTILES},
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        wasserstein=wasserstein,
        mean_shift=mean_b - mean_a,
        std_ratio=std_ratio,
        verdict=verdict,
        notes=notes,
    )


def run_gate(
    track_a_bands: dict[str, npt.ArrayLike],
    track_b_bands: dict[str, npt.ArrayLike],
    *,
    track_a_source: str,
    track_b_source: str,
    units: dict[str, str] | None = None,
    git_sha: str | None = None,
) -> DistributionGateReport:
    """Compare every shared band and decide whether Phase 5 may proceed."""
    shared = sorted(set(track_a_bands) & set(track_b_bands))
    if not shared:
        raise ValidationError(
            f"no bands in common; Track A has {sorted(track_a_bands)}, "
            f"Track B has {sorted(track_b_bands)}"
        )

    missing = sorted(set(track_a_bands) ^ set(track_b_bands))
    if missing:
        log.warning("bands present in only one track", extra={"bands": ",".join(missing)})

    units = units or {}
    report = DistributionGateReport(
        created_at=datetime.now(UTC).isoformat(),
        track_a_source=track_a_source,
        track_b_source=track_b_source,
        git_sha=git_sha,
        bands=[
            compare_band(
                track_a_bands[band],
                track_b_bands[band],
                band=band,
                unit=units.get(band, "dB"),
            )
            for band in shared
        ],
    )
    log.info(
        "distribution gate complete",
        extra={"verdict": report.verdict, "n_bands": len(report.bands)},
    )
    return report
