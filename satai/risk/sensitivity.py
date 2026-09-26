"""Risk-engine sensitivity analysis — experiment 6, and the substance of C4.

The exponents in ``R = H^alpha * E^beta * V^gamma`` default to 1.0 because
there is no principled basis for any other value. That is an honest position
only if the consequence of the choice is measured, so this module varies each
parameter one at a time and reports what actually changes.

Three quantities, because they answer different questions:

* **Spearman rank correlation** against the default map. Do the *same places*
  come out as the riskiest, even if the numbers move? This is what matters for
  prioritisation, which is what a risk map is actually used for.
* **Band reassignment fraction**. What share of cells change
  GREEN/YELLOW/ORANGE/RED? This is what a user sees, and it can be large even
  when the rank correlation is near 1.0, because banding is a threshold
  operation on a continuous field.
* **Top-decile stability** (Jaccard). Of the cells in the worst 10 % under the
  default, how many are still there? Prioritisation usually concerns the tail,
  and the tail is where a power transform bites hardest.

A result where rank correlation stays high while band assignment moves a lot is
not a contradiction — it is the finding that the *banding thresholds*, not the
exponents, drive what the map communicates.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from satai.errors import ValidationError
from satai.logging import get_logger
from satai.risk.engine import RiskEngine

log = get_logger(__name__)

__all__ = ["SensitivityPoint", "SensitivityReport", "run_sensitivity"]

DEFAULT_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)


@dataclass(frozen=True)
class SensitivityPoint:
    """One parameter setting, compared against the default configuration."""

    parameter: str
    value: float
    alpha: float
    beta: float
    gamma: float

    spearman_rho: float
    band_reassignment_fraction: float
    top_decile_jaccard: float
    mean_risk: float
    mean_risk_delta: float
    band_fractions: dict[str, float] = field(default_factory=dict)

    @property
    def is_default(self) -> bool:
        return self.value == 1.0

    def summary(self) -> str:
        return (
            f"{self.parameter}={self.value:<5} rho={self.spearman_rho:.4f} "
            f"band_change={self.band_reassignment_fraction:.1%} "
            f"top10_jaccard={self.top_decile_jaccard:.3f} "
            f"mean={self.mean_risk:.4f} ({self.mean_risk_delta:+.4f})"
        )


@dataclass
class SensitivityReport:
    """Full one-at-a-time sensitivity analysis for one hazard."""

    hazard: str
    created_at: str
    n_cells: int
    default_config_hash: str
    points: list[SensitivityPoint] = field(default_factory=list)
    git_sha: str | None = None
    notes: list[str] = field(default_factory=list)

    def for_parameter(self, parameter: str) -> list[SensitivityPoint]:
        return [p for p in self.points if p.parameter == parameter]

    @property
    def min_rank_correlation(self) -> float:
        non_default = [p.spearman_rho for p in self.points if not p.is_default]
        return min(non_default) if non_default else 1.0

    @property
    def max_band_reassignment(self) -> float:
        non_default = [p.band_reassignment_fraction for p in self.points if not p.is_default]
        return max(non_default) if non_default else 0.0

    @property
    def min_top_decile_jaccard(self) -> float:
        non_default = [p.top_decile_jaccard for p in self.points if not p.is_default]
        return min(non_default) if non_default else 1.0

    def interpretation(self) -> str:
        """Plain-language reading of the numbers, stated as what they do and do not show."""
        rho = self.min_rank_correlation
        band = self.max_band_reassignment
        jac = self.min_top_decile_jaccard

        lines: list[str] = []
        if rho >= 0.95:
            lines.append(
                f"Spatial ranking is robust to the exponent choice "
                f"(worst-case Spearman rho = {rho:.4f} across the tested grid). "
                f"The same locations rank as riskiest regardless."
            )
        elif rho >= 0.80:
            lines.append(
                f"Spatial ranking is moderately sensitive (worst-case rho = {rho:.4f}). "
                f"Prioritisation shifts somewhat with the exponents."
            )
        else:
            lines.append(
                f"Spatial ranking is SENSITIVE to the exponent choice "
                f"(worst-case rho = {rho:.4f}). The formulation materially changes "
                f"which places are identified as at risk, so the default exponents "
                f"cannot be treated as an arbitrary convenience."
            )

        lines.append(
            f"Up to {band:.1%} of cells change risk band across the tested grid. "
            + (
                "That is what a user sees, and it is much larger than the rank "
                "correlation suggests -- banding is a threshold operation, so "
                "cells near a cut-point move on small changes."
                if band > 0.10 and rho >= 0.95
                else "Band assignment moves roughly in step with the ranking."
            )
        )
        lines.append(
            f"Top-decile membership (worst-case Jaccard = {jac:.3f}): "
            + (
                "the highest-risk tail is stable, which is the part that drives prioritisation."
                if jac >= 0.80
                else "the highest-risk tail is NOT stable, so any statement about "
                "'the worst-affected areas' depends on the exponents and must be "
                "reported with them."
            )
        )
        return "\n".join(f"  - {line}" for line in lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hazard": self.hazard,
            "created_at": self.created_at,
            "n_cells": self.n_cells,
            "default_config_hash": self.default_config_hash,
            "git_sha": self.git_sha,
            "summary": {
                "min_rank_correlation": self.min_rank_correlation,
                "max_band_reassignment": self.max_band_reassignment,
                "min_top_decile_jaccard": self.min_top_decile_jaccard,
            },
            "points": [asdict(p) for p in self.points],
            "notes": self.notes,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        log.info("sensitivity report written", extra={"path": str(path)})
        return path

    def report(self) -> str:
        lines = [
            "=" * 78,
            f"  Risk-engine sensitivity analysis — hazard: {self.hazard}",
            f"  {self.n_cells:,} cells · default config {self.default_config_hash}",
            "=" * 78,
        ]
        for parameter in ("alpha", "beta", "gamma"):
            points = self.for_parameter(parameter)
            if not points:
                continue
            lines.append(
                f"\n  {parameter} (exponent on "
                f"{ {'alpha': 'hazard', 'beta': 'exposure', 'gamma': 'vulnerability'}[parameter] })"
            )
            lines.append("  " + "-" * 74)
            lines.extend(f"    {p.summary()}" for p in points)

        lines += ["", "  Interpretation", "  " + "-" * 74, self.interpretation(), "=" * 78]
        return "\n".join(lines)


def _top_decile_jaccard(a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]) -> float:
    """Jaccard overlap of the worst-10 % sets under two risk maps."""
    cut_a = float(np.percentile(a, 90))
    cut_b = float(np.percentile(b, 90))
    set_a = a >= cut_a
    set_b = b >= cut_b
    union = int(np.sum(set_a | set_b))
    return float(np.sum(set_a & set_b) / union) if union else 1.0


def run_sensitivity(
    engine: RiskEngine,
    hazards: dict[str, npt.ArrayLike],
    exposure: npt.ArrayLike,
    vulnerability: npt.ArrayLike | None,
    *,
    hazard: str,
    grid: tuple[float, ...] = DEFAULT_GRID,
    git_sha: str | None = None,
) -> SensitivityReport:
    """Vary alpha, beta and gamma one at a time and measure what changes."""
    if hazard not in hazards:
        raise ValidationError(f"hazard {hazard!r} not in {sorted(hazards)}")
    if 1.0 not in grid:
        raise ValidationError("the grid must contain 1.0, the default, as the reference point")

    base_config = engine.config
    baseline = engine.compute(hazards, exposure, vulnerability)
    base_risk = baseline.risk_vector[hazard].ravel()
    base_band = baseline.bands[hazard].ravel()

    points: list[SensitivityPoint] = []

    for parameter in ("alpha", "beta", "gamma"):
        if parameter == "gamma" and vulnerability is None:
            log.info("skipping gamma sweep: no vulnerability layer supplied")
            continue

        for value in grid:
            exponents = {
                "alpha": base_config.alpha,
                "beta": base_config.beta,
                "gamma": base_config.gamma,
            }
            exponents[parameter] = value

            variant = RiskEngine(base_config.with_exponents(**exponents), engine.couplings)
            result = variant.compute(hazards, exposure, vulnerability)
            risk = result.risk_vector[hazard].ravel()
            band = result.bands[hazard].ravel()

            # A constant risk field has no rank order to correlate. SciPy warns
            # and returns NaN; both are expected here and NaN is read below as
            # "nothing was reordered", which is what a constant field means.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", stats.ConstantInputWarning)
                rho = float(stats.spearmanr(base_risk, risk).statistic)
            points.append(
                SensitivityPoint(
                    parameter=parameter,
                    value=value,
                    alpha=exponents["alpha"],
                    beta=exponents["beta"],
                    gamma=exponents["gamma"],
                    spearman_rho=1.0 if np.isnan(rho) else rho,
                    band_reassignment_fraction=float(np.mean(band != base_band)),
                    top_decile_jaccard=_top_decile_jaccard(base_risk, risk),
                    mean_risk=float(np.nanmean(risk)),
                    mean_risk_delta=float(np.nanmean(risk) - np.nanmean(base_risk)),
                    band_fractions=result.band_fractions(hazard),
                )
            )

    notes = [
        "One-at-a-time analysis: each exponent is varied with the others held "
        "at their defaults. It does not explore interactions between them.",
        "Risk bands are percentiles of the risk distribution within this area, "
        "so band fractions are near-constant by construction; the reassignment "
        "fraction measures which CELLS move, not how many are in each band.",
    ]
    if vulnerability is None:
        notes.append("Gamma was not swept: no vulnerability layer was supplied.")

    return SensitivityReport(
        hazard=hazard,
        created_at=datetime.now(UTC).isoformat(),
        n_cells=int(base_risk.size),
        default_config_hash=base_config.config_hash(),
        points=points,
        git_sha=git_sha,
        notes=notes,
    )
