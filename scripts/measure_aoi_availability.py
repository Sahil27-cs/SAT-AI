#!/usr/bin/env python3
"""Measure real data availability over each candidate AOI.

This is the script ADR-007 promised. Phase 1 deliberately left every candidate
study area at ``status: candidate``, because selecting a study area by
familiarity and then reporting whatever metrics fall out is how projects end up
with an unexplained IoU and no account of why.

So: measure first, choose second.

Everything here comes from the **CDSE STAC catalogue, which needs no
credentials**, so this runs before any account exists.

What it measures, per AOI:

* Sentinel-1 IW GRD scene count over the window
* **Measured** revisit interval -- the median gap between acquisition dates over
  this footprint. Not the nominal mission figure, which describes an orbit
  rather than a coverage schedule.
* Orbit diversity -- distinct relative orbits and ascending/descending split.
  A pre/post SAR pair from different relative orbits has different incidence
  geometry, so a change-detection signal computed across orbits mixes flooding
  with viewing angle. An AOI with a single dominant orbit is *easier*, not
  worse.
* Sentinel-2 L2A scene count, and the fraction usable below a cloud threshold
* Tile count and approximate area, against the compute budget

Usage
-----
    python scripts/measure_aoi_availability.py
    python scripts/measure_aoi_availability.py --start 2024-06-01 --end 2024-10-31
    python scripts/measure_aoi_availability.py --aoi bihar_ganga --verbose

Writes evidence JSON to ``data/manifests/`` so the numbers in the report can be
traced back to the query that produced them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satai.config import get_settings  # noqa: E402
from satai.errors import ProviderError, SatAIError  # noqa: E402
from satai.geo.aoi import AOI, load_aoi_registry  # noqa: E402
from satai.logging import get_logger  # noqa: E402
from satai.providers.base import SearchQuery  # noqa: E402
from satai.providers.cdse import S1_IW_GRD, S2_L2A, CDSEProvider  # noqa: E402
from satai.providers.manifest import Manifest  # noqa: E402

log = get_logger("measure_aoi")

# Default window: the 2024 south-west monsoon. Chosen because flood signal in
# India is concentrated here, and because it is recent enough to reflect the
# current Sentinel-1 constellation but complete enough to have a full archive.
DEFAULT_START = date(2024, 6, 1)
DEFAULT_END = date(2024, 10, 31)

CLOUD_THRESHOLD = 20.0

# Scoring, from ADR-007's criteria. Printed with the result so a reader can
# disagree with the weights and recompute rather than having to trust them.
#
# THE LABEL TERM IS MULTIPLICATIVE, NOT ONE WEIGHT AMONG FIVE:
#
#     training_suitability = label_score x Σ(data_weight x data_score)
#
# The same argument as the risk engine's multiplicative form (ADR-008). An AOI
# with no labels is not "worth 60% as much" as a labelled one for training --
# it cannot support supervised training at all, so no amount of Sentinel-1
# coverage compensates. An additive weighting gets this wrong: a perfectly
# covered, cloud-free, compute-cheap AOI with zero labels would outrank a
# labelled one, which is exactly backwards.
#
# A zero score means "cannot be the TRAINING region". It does not mean the AOI
# is useless -- an unlabelled AOI is precisely what a transfer-evaluation target
# is, which is Mumbai's role (ADR-007). The report says so explicitly.
DATA_WEIGHTS: dict[str, float] = {
    "s1_coverage": 0.40,
    "revisit": 0.25,
    "s2_usable": 0.15,
    "feasibility": 0.20,
}

MAX_TILES_COMFORTABLE = 800


@dataclass
class AOIMeasurement:
    """Measured availability for one AOI. Plain data, written straight to JSON."""

    aoi_id: str
    name: str
    bbox: tuple[float, float, float, float]
    area_km2: float
    tile_count: int
    utm_epsg: str
    window_days: int = 0

    s1_scenes: int = 0
    s1_dates: int = 0
    s1_median_revisit_days: float | None = None
    s1_max_gap_days: int | None = None
    s1_relative_orbits: int = 0
    s1_dominant_orbit_share: float | None = None
    s1_ascending: int = 0
    s1_descending: int = 0

    s2_scenes: int = 0
    s2_usable_scenes: int = 0
    s2_usable_fraction: float | None = None

    declared_label_sources: list[str] = field(default_factory=list)
    labels_verified: bool = False

    errors: list[str] = field(default_factory=list)
    truncated: bool = False

    # -- component scores, each 0-1 ---------------------------------------

    def score_labels(self) -> float:
        """Declared labels score 0.6; only verification earns full marks.

        Deliberately capped below 1.0 until ``verify_sen1floods11.py`` has
        confirmed the labels actually cover this footprint. A config file
        asserting that labels exist is not evidence that they do.
        """
        if not self.declared_label_sources:
            return 0.0
        return 1.0 if self.labels_verified else 0.6

    def score_s1_coverage(self) -> float:
        """Temporal coverage: distinct acquisition dates against a 6-day ideal.

        Deliberately **not** scenes per km². A Sentinel-1 IW swath is ~250 km
        wide, so a small AOI is fully covered by fewer distinct scenes than a
        large one -- normalising by area would reward an AOI for being small,
        and ``score_feasibility`` already accounts for size. What actually
        matters for flood work is *how many independent looks at this footprint
        exist*, which is the count of distinct acquisition dates.
        """
        if self.s1_dates == 0 or self.window_days <= 0:
            return 0.0
        ideal_dates = self.window_days / 6.0  # best case for the constellation
        return min(1.0, self.s1_dates / ideal_dates)

    def score_revisit(self) -> float:
        """Shorter measured revisit is better; 6 days is the practical ceiling."""
        if self.s1_median_revisit_days is None:
            return 0.0
        return max(0.0, min(1.0, (18.0 - self.s1_median_revisit_days) / 12.0))

    def score_s2_usable(self) -> float:
        if self.s2_usable_fraction is None:
            return 0.0
        return min(1.0, self.s2_usable_fraction / 0.30)

    def score_feasibility(self) -> float:
        """Penalises AOIs whose tile count exceeds a student compute budget."""
        if self.tile_count <= 0:
            return 0.0
        return min(1.0, MAX_TILES_COMFORTABLE / self.tile_count)

    def component_scores(self) -> dict[str, float]:
        return {
            "labels": self.score_labels(),
            "s1_coverage": self.score_s1_coverage(),
            "revisit": self.score_revisit(),
            "s2_usable": self.score_s2_usable(),
            "feasibility": self.score_feasibility(),
        }

    def data_score(self) -> float:
        """Weighted quality of the imagery available, ignoring labels.

        Reported separately because it is what matters for a **transfer-
        evaluation** AOI, where no training labels are needed.
        """
        scores = self.component_scores()
        return sum(scores[k] * w for k, w in DATA_WEIGHTS.items())

    def total_score(self) -> float:
        """Suitability as the **training** region. Zero without labels."""
        return self.score_labels() * self.data_score()

    @property
    def can_train(self) -> bool:
        return self.score_labels() > 0.0


def measure_aoi(
    aoi: AOI,
    provider: CDSEProvider,
    start: date,
    end: date,
    manifest: Manifest,
    *,
    verbose: bool = False,
) -> AOIMeasurement:
    """Query the catalogue for one AOI and summarise what is actually there."""
    measurement = AOIMeasurement(
        aoi_id=aoi.id,
        name=aoi.name,
        bbox=aoi.bbox,
        area_km2=round(aoi.approx_area_km2, 1),
        tile_count=aoi.approx_tile_count(),
        utm_epsg=aoi.utm_epsg,
        declared_label_sources=list(aoi.label_sources),
        window_days=(end - start).days + 1,
    )

    # --- Sentinel-1 ---
    try:
        s1 = provider.search(
            SearchQuery(
                bbox=aoi.bbox,
                start=start,
                end=end,
                collection="SENTINEL-1",
                limit=2000,
                extra={"product_type": S1_IW_GRD},
            )
        )
        manifest.record_search(s1)
        measurement.s1_scenes = len(s1)
        measurement.s1_dates = len(s1.acquisition_dates)
        measurement.truncated |= s1.truncated

        gaps = s1.revisit_gaps_days()
        if gaps:
            measurement.s1_median_revisit_days = round(statistics.median(gaps), 1)
            measurement.s1_max_gap_days = max(gaps)

        orbits = Counter(s.relative_orbit for s in s1.scenes if s.relative_orbit is not None)
        measurement.s1_relative_orbits = len(orbits)
        if orbits:
            measurement.s1_dominant_orbit_share = round(
                orbits.most_common(1)[0][1] / sum(orbits.values()), 2
            )
        directions = Counter(s.orbit_direction for s in s1.scenes if s.orbit_direction)
        measurement.s1_ascending = directions.get("ASCENDING", 0)
        measurement.s1_descending = directions.get("DESCENDING", 0)

        if verbose and s1.scenes:
            print(f"    first S1 scene: {s1.scenes[0].scene_id}")

    except (ProviderError, SatAIError) as exc:
        measurement.errors.append(f"Sentinel-1 query failed: {exc}")
        log.warning("s1 query failed", extra={"aoi": aoi.id, "error": str(exc)})

    # --- Sentinel-2 ---
    try:
        s2 = provider.search(
            SearchQuery(
                bbox=aoi.bbox,
                start=start,
                end=end,
                collection="SENTINEL-2",
                limit=2000,
                extra={"product_type": S2_L2A},
            )
        )
        manifest.record_search(s2)
        measurement.s2_scenes = len(s2)
        measurement.s2_usable_scenes = len(s2.usable(CLOUD_THRESHOLD))
        measurement.truncated |= s2.truncated
        if measurement.s2_scenes:
            measurement.s2_usable_fraction = round(
                measurement.s2_usable_scenes / measurement.s2_scenes, 3
            )

    except (ProviderError, SatAIError) as exc:
        measurement.errors.append(f"Sentinel-2 query failed: {exc}")
        log.warning("s2 query failed", extra={"aoi": aoi.id, "error": str(exc)})

    return measurement


def print_report(measurements: list[AOIMeasurement], start: date, end: date) -> None:
    """Print the comparison table and the interpretation."""
    print()
    print("=" * 96)
    print(f"  SAT-AI — AOI data availability, {start} to {end}")
    print("  Source: CDSE STAC catalogue (public, no credentials required)")
    print("=" * 96)

    header = (
        f"{'AOI':<20} {'S1':>5} {'dates':>6} {'revisit':>8} {'orbits':>7} "
        f"{'S2':>5} {'clear':>7} {'tiles':>7} {'score':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for m in sorted(measurements, key=lambda x: x.total_score(), reverse=True):
        revisit = f"{m.s1_median_revisit_days:.1f}d" if m.s1_median_revisit_days else "—"
        clear = f"{m.s2_usable_fraction:.0%}" if m.s2_usable_fraction is not None else "—"
        print(
            f"{m.aoi_id:<20} {m.s1_scenes:>5} {m.s1_dates:>6} {revisit:>8} "
            f"{m.s1_relative_orbits:>7} {m.s2_scenes:>5} {clear:>7} "
            f"{m.tile_count:>7} {m.total_score():>6.3f}"
        )

    print("\nColumns: S1/S2 = scene counts · dates = distinct S1 acquisition dates")
    print("         revisit = MEASURED median gap between acquisitions (not nominal)")
    print(
        f"         orbits = distinct relative orbits · "
        f"clear = S2 scenes ≤{CLOUD_THRESHOLD:.0f}% cloud"
    )
    print(f"         tiles = 512px @10m tiles to cover the AOI (budget ≈{MAX_TILES_COMFORTABLE})")
    print("         score = TRAINING suitability; 0 means no labels, not 'useless'")

    print("\n" + "-" * 96)
    print("  Component scores")
    print("-" * 96)
    weight_row = "  ".join(f"{k}={v:.2f}" for k, v in DATA_WEIGHTS.items())
    print(f"  data weights (ADR-007): {weight_row}")
    print("  TRAINING score = labels x data score  (multiplicative: no labels, no")
    print("  supervised training, whatever the imagery looks like -- see ADR-008)\n")
    comp_header = (
        f"{'AOI':<20} {'labels':>8} {'s1_cov':>8} {'revisit':>8} "
        f"{'s2_use':>8} {'feasib':>8} {'DATA':>7} {'TRAIN':>7}"
    )
    print(comp_header)
    print("-" * len(comp_header))
    for m in sorted(measurements, key=lambda x: x.total_score(), reverse=True):
        c = m.component_scores()
        print(
            f"{m.aoi_id:<20} {c['labels']:>8.2f} {c['s1_coverage']:>8.2f} "
            f"{c['revisit']:>8.2f} {c['s2_usable']:>8.2f} "
            f"{c['feasibility']:>8.2f} {m.data_score():>7.3f} {m.total_score():>7.3f}"
        )

    # --- interpretation -------------------------------------------------
    print("\n" + "=" * 96)
    print("  Notes")
    print("=" * 96)

    for m in measurements:
        notes: list[str] = []
        if m.s1_median_revisit_days and m.s1_median_revisit_days > 12:
            notes.append(
                f"measured revisit {m.s1_median_revisit_days:.1f}d is longer than the "
                "nominal 6-12d — many flood peaks will be missed entirely"
            )
        if m.s1_dominant_orbit_share and m.s1_dominant_orbit_share < 0.5:
            notes.append(
                f"no dominant relative orbit ({m.s1_dominant_orbit_share:.0%}) — "
                "restrict change detection to within-orbit pairs"
            )
        if m.s2_usable_fraction is not None and m.s2_usable_fraction < 0.15:
            notes.append(
                f"only {m.s2_usable_fraction:.0%} of optical scenes are usable — "
                "this AOI is effectively SAR-only in this window"
            )
        if m.tile_count > MAX_TILES_COMFORTABLE:
            notes.append(
                f"{m.tile_count} tiles exceeds the compute budget — narrow the bbox "
                "or justify the cost in an ADR"
            )
        if not m.can_train:
            notes.append(
                "no declared labels -> cannot be the TRAINING region. This is the "
                "profile of a transfer-EVALUATION target (its data score is "
                f"{m.data_score():.2f}), which is a valid and useful role."
            )
        if m.declared_label_sources and not m.labels_verified:
            notes.append(
                "label sources are DECLARED but UNVERIFIED — "
                "run scripts/verify_sen1floods11.py before relying on them"
            )
        if m.errors:
            notes.extend(m.errors)

        if notes:
            print(f"\n  {m.aoi_id}:")
            for note in notes:
                print(f"    • {note}")

    if any(m.truncated for m in measurements):
        print(
            "\n  ⚠ At least one query hit the result limit. Scene counts are a "
            "LOWER BOUND, not an exact count."
        )

    print("\n" + "=" * 96)
    print("  This ranking is EVIDENCE, not a decision.")
    print("  Read the component scores, weigh them against the research question,")
    print("  then promote one AOI to status: selected in configs/aoi.yaml and")
    print("  record the table plus your reasoning in docs/adr/ADR-007.")
    print("=" * 96 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    parser.add_argument("--aoi", action="append", help="Limit to these AOI ids.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.start > args.end:
        print(f"error: --start {args.start} is after --end {args.end}", file=sys.stderr)
        return 2

    settings = get_settings()
    settings.ensure_directories()

    registry = load_aoi_registry()
    aois = [a for a in registry.aois if not args.aoi or a.id in args.aoi]
    if not aois:
        print(f"error: no AOI matched {args.aoi}", file=sys.stderr)
        return 2

    manifest = Manifest(
        manifest_id=f"aoi_availability_{datetime.now(UTC):%Y%m%d}",
        purpose=(
            f"Phase 2 AOI selection evidence: CDSE catalogue availability over "
            f"{len(aois)} candidate AOIs, {args.start} to {args.end}"
        ),
    )

    provider = CDSEProvider()
    print(f"\nQuerying CDSE STAC for {len(aois)} AOI(s). This takes a minute or two.\n")

    measurements: list[AOIMeasurement] = []
    for aoi in aois:
        print(f"  → {aoi.id} ({aoi.approx_area_km2:,.0f} km²)")
        measurements.append(
            measure_aoi(aoi, provider, args.start, args.end, manifest, verbose=args.verbose)
        )

    print_report(measurements, args.start, args.end)

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "window": {"start": args.start.isoformat(), "end": args.end.isoformat()},
        "cloud_threshold": CLOUD_THRESHOLD,
        "data_weights": DATA_WEIGHTS,
        "scoring": "training_score = label_score * data_score (multiplicative)",
        "source": "CDSE STAC catalogue",
        "measurements": [
            {
                **asdict(m),
                "component_scores": m.component_scores(),
                "total_score": round(m.total_score(), 4),
            }
            for m in measurements
        ],
    }
    out = settings.manifests_dir / f"aoi_availability_{datetime.now(UTC):%Y%m%d}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    manifest.stats = {"aois_measured": len(measurements)}
    manifest.write(settings.manifests_dir)

    print(f"Evidence written to {out.relative_to(REPO_ROOT)}")
    print(f"Manifest written to {settings.manifests_dir.relative_to(REPO_ROOT)}\n")

    return 1 if all(m.s1_scenes == 0 for m in measurements) else 0


if __name__ == "__main__":
    raise SystemExit(main())
