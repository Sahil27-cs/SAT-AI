"""Post-event change detection from a real pre/post Sentinel-1 pair.

    python -m ml.hazards.run_damage --aoi nepal_koshi_terai \
        --pre 2024-09-15 --post 2024-09-27

`satai.hazards.damage` has always been able to compute a log-ratio change field
and classify its magnitude. What it never had was a pair of real acquisitions.
This runner supplies one, from the Planetary Computer's terrain-corrected
Sentinel-1 archive.

**This is post-event damage assessment. It is not earthquake prediction**, and
it is not a building damage grade. What it measures is the change in radar
backscatter between two dates. That change can be structural damage; it can
equally be standing water, a harvested field, or a different incidence angle.
The artifact says so, and every caveat travels with the number.

The orbit constraint, which is the whole reason this is defensible
-----------------------------------------------------------------
Both scenes must come from the **same relative orbit**. Sentinel-1 views a given
point from a different angle on a different track, and backscatter depends
strongly on that angle — so a pre/post pair from two orbits produces a change
field that mixes real change with viewing geometry, and the geometry usually
wins. This runner refuses a cross-orbit pair rather than producing a
plausible-looking map of nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
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

    from satai.paths import relative_to_repo

import numpy as np

from ml.flood.infer_scene import read_window, to_db
from satai.geo.aoi import load_aoi_registry
from satai.hazards.damage import assess_change, log_ratio
from satai.providers import PlanetaryComputerProvider, SearchQuery

DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "damage"


def _nearest(scenes: list[Any], target: date) -> Any | None:
    dated = [s for s in scenes if s.acquired_at]
    if not dated:
        return None
    return min(dated, key=lambda s: abs((s.acquired_at.date() - target).days))


def write_overlay(metadata_path: Path) -> dict[str, Any]:
    """Draw the change field as a map overlay and record it in the metadata.

    Separate from ``main`` so an existing result can be drawn without reading
    both scenes again. Only change at or above the analysis threshold is drawn,
    with opacity following its magnitude up to 10 dB.
    """
    import rasterio

    from ml.hazards.overlay import continuous_overlay

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    change_path = REPO_ROOT / meta["artifacts"]["change_db"]
    with rasterio.open(change_path) as src:
        magnitude = np.abs(src.read(1).astype("float32"))
        profile = {"crs": src.crs, "transform": src.transform}
    threshold = float(meta["method"]["threshold_db"])
    overlay = continuous_overlay(
        magnitude,
        profile,
        REPO_ROOT / "frontend" / "public" / "layers" / f"{change_path.stem}.png",
        vmin=threshold,
        vmax=10.0,
        colour=(147, 51, 234),
        floor=threshold,
    )
    described: dict[str, Any] = {
        "url": f"/layers/{overlay.path.name}",
        **overlay.describe(),
        "shows": f"absolute backscatter change of at least {threshold:g} dB",
    }
    meta["overlay"] = described
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return described


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay-from",
        type=Path,
        default=None,
        help="draw the overlay for an existing metadata file and stop",
    )
    parser.add_argument("--aoi")
    parser.add_argument("--pre", type=date.fromisoformat)
    parser.add_argument("--post", type=date.fromisoformat)
    parser.add_argument("--band", default="vv", choices=("vv", "vh"))
    parser.add_argument("--threshold-db", type=float, default=3.0)
    parser.add_argument("--window-days", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.overlay_from is not None:
        spec = write_overlay(args.overlay_from)
        print(f"overlay {spec['url']}  {spec['width_px']}x{spec['height_px']} px")
        return 0
    if not (args.aoi and args.pre and args.post):
        parser.error("--aoi, --pre and --post are required unless --overlay-from is given")

    registry = load_aoi_registry(REPO_ROOT / "configs" / "aoi.yaml")
    aoi = next((a for a in registry.aois if a.id == args.aoi), None)
    if aoi is None:
        raise SystemExit(f"{args.aoi!r} is not a configured study area")

    provider = PlanetaryComputerProvider()

    def find(target: date) -> Any | None:
        result = provider.search(
            SearchQuery(
                bbox=aoi.bbox,
                start=target - timedelta(days=args.window_days),
                end=target + timedelta(days=args.window_days),
                collection="sentinel-1-rtc",
                limit=50,
            )
        )
        return _nearest(result.scenes, target)

    pre_scene, post_scene = find(args.pre), find(args.post)
    if pre_scene is None or post_scene is None:
        print(
            f"BLOCKED: no Sentinel-1 RTC scene within {args.window_days} days of "
            f"{'pre' if pre_scene is None else 'post'} date over {aoi.id}."
        )
        return 2

    # The constraint that makes the difference interpretable.
    if pre_scene.relative_orbit != post_scene.relative_orbit:
        print(
            f"BLOCKED: the nearest scenes are on different relative orbits "
            f"({pre_scene.relative_orbit} and {post_scene.relative_orbit}). "
            f"Backscatter depends strongly on incidence angle, so their "
            f"difference would mix real change with viewing geometry. Choose "
            f"dates one repeat cycle apart on the same track."
        )
        return 3

    print(f"pre   {pre_scene.scene_id}")
    print(f"      {pre_scene.acquired_at}  orbit {pre_scene.relative_orbit}")
    print(f"post  {post_scene.scene_id}")
    print(f"      {post_scene.acquired_at}  orbit {post_scene.relative_orbit}")
    separation = (post_scene.acquired_at - pre_scene.acquired_at).days

    pre_linear, _t, profile = read_window(provider.asset_href(pre_scene, args.band), aoi.bbox)
    post_linear, _t2, _p2 = read_window(provider.asset_href(post_scene, args.band), aoi.bbox)
    pre_db, post_db = to_db(pre_linear), to_db(post_linear)

    valid = np.isfinite(pre_db) & np.isfinite(post_db)
    change = log_ratio(np.where(valid, pre_db, 0.0), np.where(valid, post_db, 0.0), already_db=True)
    assessment = assess_change(change, threshold_db=args.threshold_db, valid=valid)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{aoi.id}_{pre_scene.acquired_at:%Y%m%d}_{post_scene.acquired_at:%Y%m%d}_{args.band}"

    import rasterio

    raster_profile = dict(profile)
    raster_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    raster_profile.pop("nodata", None)
    change_path = args.out / f"{stem}_change_db.tif"
    with rasterio.open(change_path, "w", **raster_profile) as dst:
        dst.write(np.nan_to_num(change, nan=0.0).astype("float32"), 1)

    payload: dict[str, Any] = {
        "analysis": "post_event_change_detection",
        "not_a_prediction": (
            "Post-event damage assessment from observed imagery. This is NOT "
            "earthquake prediction, and SAT-AI makes no earthquake forecast of "
            "any kind."
        ),
        "hazard": "damage",
        "source_kind": "derived",
        "aoi": aoi.id,
        "aoi_name": aoi.name,
        "country": aoi.country,
        "bbox": list(aoi.bbox),
        "run_at": datetime.now(UTC).isoformat(),
        "pair": {
            "pre_scene_id": pre_scene.scene_id,
            "pre_acquired_at": pre_scene.acquired_at.isoformat(),
            "post_scene_id": post_scene.scene_id,
            "post_acquired_at": post_scene.acquired_at.isoformat(),
            "relative_orbit": pre_scene.relative_orbit,
            "orbit_direction": pre_scene.orbit_direction,
            "separation_days": separation,
            "band": args.band,
            "radiometry": pre_scene.properties["radiometry"],
            "collection": "sentinel-1-rtc",
            "provider": "planetary_computer",
            "same_relative_orbit": True,
        },
        "method": {
            "operation": "log-ratio of gamma0 in dB, post minus pre",
            "threshold_db": args.threshold_db,
            "classes": ["none", "low", "moderate", "high"],
        },
        "result": {
            "changed_fraction": round(assessment.changed_fraction, 6),
            "fractions": {k: round(v, 6) for k, v in assessment.fractions.items()},
            "coverage": round(assessment.coverage, 6),
            "n_valid_pixels": assessment.n_valid,
            "n_total_pixels": assessment.n_total,
            "mean_abs_change_db": round(float(np.nanmean(np.abs(change[valid]))), 4),
            "p95_abs_change_db": round(float(np.nanpercentile(np.abs(change[valid]), 95)), 4),
        },
        "artifacts": {"change_db": relative_to_repo(change_path)},
        "caveats": [
            *assessment.caveats(),
            f"Both scenes are from relative orbit {pre_scene.relative_orbit}, "
            f"{separation} days apart. A cross-orbit pair would mix change with "
            f"incidence-angle difference, and this runner refuses one.",
            "Over a flood, most of this change IS the flood water. Separating "
            "structural damage from inundation needs the flood extent as a mask, "
            "and the flood extent for this scene did not pass its own gate.",
            "Not a building damage grade and not a monetary loss estimate: "
            "SAT-AI has no building footprints and no asset values.",
        ],
    }
    (args.out / f"{stem}_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_overlay(args.out / f"{stem}_metadata.json")

    result = payload["result"]
    print(f"\n      separation {separation} days, same orbit")
    print(f"      coverage        {result['coverage']:.1%}")
    print(f"      changed         {result['changed_fraction']:.2%} of valid pixels")
    for name, share in assessment.fractions.items():
        print(f"        {name:<10} {share:>8.2%}")
    print(f"      mean |change|   {result['mean_abs_change_db']:.2f} dB")
    print(f"\nwritten to {relative_to_repo(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
