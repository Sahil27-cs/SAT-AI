"""Multi-hazard risk engine (ADR-008).

    R_h = H_h^alpha * E^beta * V^gamma

Multiplicative, because that is the formulation with the correct boundary
condition: zero exposure implies zero risk. An uninhabited floodplain has high
hazard and effectively no risk, and an additive or averaged form gets that
wrong -- it assigns substantial "risk" to an empty river meander. This is the
most common defect in multi-hazard dashboards.

Across hazards the engine deliberately refuses to average. A flood score of 0.7
and a fire score of 0.7 are outputs of different models trained on different
labels with different base rates; their mean denotes nothing. Instead it
returns a **risk vector**, a **dominant hazard**, and a **compound term** that
exists only where ``configs/couplings.yaml`` documents a physical mechanism.

Every value that leaves this module is wrapped in a
:class:`~satai.provenance.ProvenanceEnvelope` carrying its inputs, its
configuration hash and its caveats -- including, always, that these are SAT-AI
prototype research levels and not official warnings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from satai.errors import ValidationError
from satai.logging import get_logger
from satai.provenance import (
    HazardType,
    ProvenanceEnvelope,
    SourceKind,
    SpatialRef,
    envelope,
)

log = get_logger(__name__)

__all__ = [
    "RiskConfig",
    "RiskEngine",
    "RiskResult",
    "normalise_percentile",
]

FloatArray = npt.NDArray[np.floating]


def normalise_percentile(
    values: npt.ArrayLike,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> FloatArray:
    """Scale to [0, 1] using robust percentile bounds.

    Clipped at the 2nd/98th percentile so a single extreme cell cannot compress
    the whole scale -- which, in a multiplicative formulation, would drive every
    other cell's risk toward zero.
    """
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValidationError("cannot normalise an array with no finite values")

    low = float(np.percentile(finite, lower_percentile))
    high = float(np.percentile(finite, upper_percentile))
    if high <= low:
        # A constant field carries no spatial information; a mid-scale constant
        # is the honest encoding, and the caller is warned.
        log.warning("constant field during normalisation", extra={"value": low})
        return np.full_like(array, 0.5, dtype=np.float64)

    return np.clip((array - low) / (high - low), 0.0, 1.0)


@dataclass(frozen=True)
class RiskConfig:
    """Risk-engine parameters, loaded from ``configs/risk.yaml``."""

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    floor: float = 0.01
    method: str = "multiplicative"
    lower_percentile: float = 2.0
    upper_percentile: float = 98.0
    band_percentiles: tuple[float, ...] = (50.0, 75.0, 90.0, 100.0)
    band_names: tuple[str, ...] = ("GREEN", "YELLOW", "ORANGE", "RED")
    forbid_naive_average: bool = True
    disclaimer: str = (
        "SAT-AI prototype risk level. Research output from a student research "
        "prototype. This is NOT an official warning and does not replace IMD, "
        "NDMA or State Disaster Management Authority advisories."
    )
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: Path) -> RiskConfig:
        if not path.is_file():
            raise ValidationError(f"risk config not found: {path}")
        data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

        formulation = data.get("formulation", {})
        exponents = formulation.get("exponents", {})
        normalisation = data.get("normalisation", {})
        levels = data.get("levels", {})
        bands = levels.get("bands", [])

        return cls(
            alpha=float(exponents.get("alpha", 1.0)),
            beta=float(exponents.get("beta", 1.0)),
            gamma=float(exponents.get("gamma", 1.0)),
            floor=float(formulation.get("floor", 0.01)),
            method=str(formulation.get("method", "multiplicative")),
            lower_percentile=float(normalisation.get("lower_percentile", 2.0)),
            upper_percentile=float(normalisation.get("upper_percentile", 98.0)),
            band_percentiles=tuple(float(b["max_percentile"]) for b in bands) or (50, 75, 90, 100),
            band_names=tuple(str(b["name"]) for b in bands) or ("GREEN", "YELLOW", "ORANGE", "RED"),
            forbid_naive_average=bool(
                data.get("multi_hazard", {}).get("forbid_naive_average", True)
            ),
            disclaimer=str(levels.get("disclaimer", cls.disclaimer)).strip(),
            raw=data,
        )

    def with_exponents(self, alpha: float, beta: float, gamma: float) -> RiskConfig:
        """Copy with different exponents, for the sensitivity analysis."""
        return RiskConfig(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            floor=self.floor,
            method=self.method,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
            band_percentiles=self.band_percentiles,
            band_names=self.band_names,
            forbid_naive_average=self.forbid_naive_average,
            disclaimer=self.disclaimer,
            raw=self.raw,
        )

    def config_hash(self) -> str:
        """Stable hash of the parameters that affect the output."""
        payload = {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "floor": self.floor,
            "method": self.method,
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "band_percentiles": list(self.band_percentiles),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class RiskResult:
    """Per-hazard risk, the dominant hazard, and any justified compound term."""

    risk_vector: dict[str, FloatArray]
    dominant_hazard: npt.NDArray[np.str_]
    dominant_margin: FloatArray
    compound: dict[str, FloatArray] = field(default_factory=dict)
    bands: dict[str, npt.NDArray[np.str_]] = field(default_factory=dict)
    config_hash: str = ""
    caveats: tuple[str, ...] = ()

    def band_fractions(self, hazard: str) -> dict[str, float]:
        """Share of cells in each risk band, for the sensitivity analysis."""
        band = self.bands[hazard]
        total = band.size
        unique, counts = np.unique(band, return_counts=True)
        return {str(u): float(c / total) for u, c in zip(unique, counts, strict=True)}


class RiskEngine:
    """Computes multi-hazard risk from hazard, exposure and vulnerability."""

    def __init__(self, config: RiskConfig, couplings: dict[str, Any] | None = None) -> None:
        self.config = config
        self.couplings = couplings or {}

        if config.method != "multiplicative":
            log.warning(
                "non-multiplicative risk formulation in use",
                extra={"method": config.method},
            )

    @classmethod
    def from_configs(cls, risk_path: Path, couplings_path: Path | None = None) -> RiskEngine:
        couplings: dict[str, Any] = {}
        if couplings_path and couplings_path.is_file():
            couplings = yaml.safe_load(couplings_path.read_text(encoding="utf-8")) or {}
        return cls(RiskConfig.from_yaml(risk_path), couplings)

    # -- core ---------------------------------------------------------------

    def compute_hazard_risk(
        self,
        hazard: npt.ArrayLike,
        exposure: npt.ArrayLike,
        vulnerability: npt.ArrayLike | None,
    ) -> FloatArray:
        """``R = H^alpha * E^beta * V^gamma``, on already-normalised inputs.

        A missing vulnerability layer is **excluded**, not imputed. Imputing
        zero would zero the risk everywhere; imputing one would silently assert
        maximum vulnerability. Excluding it and saying so is the only honest
        option, and the exclusion becomes an envelope caveat.
        """
        cfg = self.config
        h = np.clip(np.asarray(hazard, dtype=np.float64), 0.0, 1.0)
        e = np.clip(np.asarray(exposure, dtype=np.float64), 0.0, 1.0)

        if h.shape != e.shape:
            raise ValidationError(f"hazard shape {h.shape} does not match exposure shape {e.shape}")

        risk = np.power(np.maximum(h, cfg.floor), cfg.alpha) * np.power(
            np.maximum(e, cfg.floor), cfg.beta
        )

        if vulnerability is not None:
            v = np.clip(np.asarray(vulnerability, dtype=np.float64), 0.0, 1.0)
            if v.shape != h.shape:
                raise ValidationError(
                    f"vulnerability shape {v.shape} does not match hazard {h.shape}"
                )
            risk = risk * np.power(np.maximum(v, cfg.floor), cfg.gamma)

        clipped: FloatArray = np.clip(risk, 0.0, 1.0)
        return clipped

    def assign_bands(self, risk: npt.ArrayLike) -> npt.NDArray[np.str_]:
        """Assign GREEN/YELLOW/ORANGE/RED by percentile of the risk distribution.

        These are **relative** bands within the area of interest, not absolute
        physical thresholds. A RED cell is in the top decile of this AOI, which
        is a different statement from "this area will flood".
        """
        array = np.asarray(risk, dtype=np.float64)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise ValidationError("cannot band a risk field with no finite values")

        cuts = [float(np.percentile(finite, p)) for p in self.config.band_percentiles[:-1]]

        # Ascending order, first match wins: a cell at the 60th percentile is
        # YELLOW because it failed the GREEN cut, not RED because it passed no
        # cut. Anything above the last cut falls through to the top band.
        out = np.full(array.shape, self.config.band_names[-1], dtype="<U8")
        assigned = np.zeros(array.shape, dtype=bool)
        for name, cut in zip(self.config.band_names[:-1], cuts, strict=True):
            take = (~assigned) & (array <= cut)
            out[take] = name
            assigned |= take
        return out

    def compute(
        self,
        hazards: dict[str, npt.ArrayLike],
        exposure: npt.ArrayLike,
        vulnerability: npt.ArrayLike | None = None,
        *,
        normalise: bool = True,
    ) -> RiskResult:
        """Full multi-hazard computation.

        ``hazards`` maps a hazard name to its probability or index field.
        """
        if not hazards:
            raise ValidationError("no hazards supplied")

        cfg = self.config

        def prep(x: npt.ArrayLike) -> FloatArray:
            return (
                normalise_percentile(
                    x,
                    lower_percentile=cfg.lower_percentile,
                    upper_percentile=cfg.upper_percentile,
                )
                if normalise
                else np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
            )

        e = prep(exposure)
        v = prep(vulnerability) if vulnerability is not None else None

        risk_vector = {
            name: self.compute_hazard_risk(prep(field_values), e, v)
            for name, field_values in hazards.items()
        }

        names = sorted(risk_vector)
        stacked = np.stack([risk_vector[n] for n in names])
        order = np.argsort(stacked, axis=0)
        top = order[-1]
        dominant = np.array(names, dtype="<U16")[top]

        if len(names) > 1:
            second = order[-2]
            margin = (
                np.take_along_axis(stacked, top[None], 0)[0]
                - np.take_along_axis(stacked, second[None], 0)[0]
            )
        else:
            margin = np.ones_like(stacked[0])

        compound = self._compute_compound(risk_vector)

        caveats = [
            cfg.disclaimer,
            "Risk bands are percentiles within this area of interest, not "
            "absolute physical thresholds.",
        ]
        if v is None:
            caveats.append(
                "Vulnerability layer unavailable: risk computed from hazard and "
                "exposure only. The result is therefore an upper bound on what "
                "the configured formulation would produce."
            )
        else:
            caveats.append(
                "Vulnerability is a PROXY index from building density, road "
                "accessibility and elevation above drainage. It omits income, "
                "age structure, disability and tenure, and so systematically "
                "under-represents the vulnerability of marginalised populations."
            )

        return RiskResult(
            risk_vector=risk_vector,
            dominant_hazard=dominant,
            dominant_margin=margin,
            compound=compound,
            bands={n: self.assign_bands(r) for n, r in risk_vector.items()},
            config_hash=cfg.config_hash(),
            caveats=tuple(caveats),
        )

    def _compute_compound(self, risk_vector: dict[str, FloatArray]) -> dict[str, FloatArray]:
        """Apply only the couplings with a documented physical mechanism."""
        out: dict[str, FloatArray] = {}
        for coupling in self.couplings.get("couplings", []):
            if not coupling.get("enabled", False):
                continue
            source, target = coupling["from_hazard"], coupling["to_hazard"]
            if source not in risk_vector or target not in risk_vector:
                continue
            strength = float(coupling.get("strength", 0.0))
            out[coupling["id"]] = np.clip(
                risk_vector[target] + strength * risk_vector[source] * risk_vector[target],
                0.0,
                1.0,
            )
        return out

    # -- provenance ---------------------------------------------------------

    def to_envelope(
        self,
        result: RiskResult,
        hazard: str,
        *,
        spatial: SpatialRef | None = None,
        model_versions: dict[str, str] | None = None,
    ) -> ProvenanceEnvelope[dict[str, float]]:
        """Wrap a hazard's summary statistics for the serving and agent planes.

        ``SourceKind.INDEX`` rather than ``MODEL``: the risk score is a
        documented, configurable composite, not a trained prediction, and the
        interface must not describe it as a forecast.
        """
        risk = result.risk_vector[hazard]
        fractions = result.band_fractions(hazard)

        return envelope(
            {
                "mean_risk": float(np.nanmean(risk)),
                "max_risk": float(np.nanmax(risk)),
                "p90_risk": float(np.nanpercentile(risk, 90)),
                **{f"fraction_{k.lower()}": v for k, v in fractions.items()},
            },
            quantity=f"{hazard}_risk_summary",
            unit="index",
            kind=SourceKind.INDEX,
            source_id="risk_engine",
            version=result.config_hash,
            spatial=spatial,
            hazard=HazardType(hazard) if hazard in {h.value for h in HazardType} else None,
            caveats=list(result.caveats),
            formulation=f"R = H^{self.config.alpha} * E^{self.config.beta} * V^{self.config.gamma}",
            model_versions=model_versions or {},
        )
