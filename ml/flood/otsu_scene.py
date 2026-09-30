"""Flood extent over a real Sentinel-1 scene with the evaluated Otsu baseline.

    python -m ml.flood.otsu_scene --aoi nepal_koshi_terai \
        --start 2024-09-25 --end 2024-09-29

Why the baseline, not the U-Net
-------------------------------
The U-Net was run over this scene and its own distribution gate refused the
transfer: it was trained on sigma0 and the scene is gamma0, about 2.5 dB apart
on both polarisations. A ratio-only U-Net would pass the gate, because the
offset cancels in VV - VH, but on held-out India it scores IoU 0.3686, below the
baseline. The signal that separates water from land lives in the absolute
backscatter, which is exactly what does not survive a change of product.

Otsu does not have that problem by construction. It picks its threshold from
each tile's own histogram, so a constant radiometric offset moves the threshold
with it. It is also the one flood method in this project that was scored
without ever seeing labels from the region it was scored on: IoU 0.3754 on
held-out India, with its separability guard (0.66) chosen leave-one-region-out
across all eleven regions.

It is applied here exactly as it was evaluated: per 512 px tile, with the guard,
on VV in dB. A tile whose histogram is unimodal is reported as having no water
rather than bisected.

Flood, not water
----------------
Rivers are water in every image. The Koshi alone would dominate any "water
extent" over this area. Pixels the JRC Global Surface Water record marks as
water in all twelve months of the year are subtracted, so what remains is water
that was not there in a normal year. This is JRC's own definition of permanent
water, used unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from datetime import UTC, date, datetime
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

from ml.flood.infer_scene import GDAL_ENV, read_window, to_db
from ml.flood.predict import _pixel_area_km2, _write_geotiff
from ml.hazards.overlay import categorical_overlay
from satai.geo.aoi import load_aoi_registry
from satai.ml.baseline import OtsuHandBaseline
from satai.providers import PlanetaryComputerProvider, SearchQuery
from satai.providers.planetary import _post_json, sign_href

DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "flood_scenes"
OVERLAY_DIR = REPO_ROOT / "frontend" / "public" / "layers"

#: The tile size the baseline was evaluated at (one Sen1Floods11 chip).
TILE = 512

#: The baseline's measured skill, quoted with the protocol that produced it.
BASELINE_IOU_HELD_OUT_INDIA = 0.3754

#: JRC seasonality: months per year with water. 12 means permanent.
PERMANENT_MONTHS = 12

#: Overlay colours. Flood only; permanent water is left transparent so the map
#: does not claim the river as flooding.
PALETTE: dict[int, tuple[int, int, int, int]] = {1: (37, 99, 235, 215)}


def otsu_by_tile(
    vv_db: np.ndarray, baseline: OtsuHandBaseline
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the baseline tile by tile, as it was evaluated."""
    height, width = vv_db.shape
    water = np.zeros(vv_db.shape, dtype=bool)
    thresholds, separabilities = [], []
    tiles = unimodal = empty = 0

    for top in range(0, height, TILE):
        for left in range(0, width, TILE):
            block = vv_db[top : top + TILE, left : left + TILE]
            tiles += 1
            valid = np.isfinite(block)
            if valid.sum() < 1000:
                empty += 1
                continue
            predicted = baseline.predict(block[None], valid=valid)
            if baseline.separability_ is not None:
                separabilities.append(baseline.separability_)
            if (
                baseline.separability_ is not None
                and baseline.separability_ < baseline.min_separability
            ):
                unimodal += 1
                continue
            if baseline.fitted_threshold_ is not None:
                thresholds.append(baseline.fitted_threshold_)
            water[top : top + TILE, left : left + TILE] = predicted > 0.5

    return water, {
        "tiles": tiles,
        "tiles_without_data": empty,
        "tiles_called_unimodal": unimodal,
        "tiles_thresholded": tiles - empty - unimodal,
        "median_threshold_db": round(float(np.median(thresholds)), 3) if thresholds else None,
        "threshold_range_db": [round(float(min(thresholds)), 3), round(float(max(thresholds)), 3)]
        if thresholds
        else None,
        "median_separability": round(float(np.median(separabilities)), 4)
        if separabilities
        else None,
    }


def permanent_water(profile: dict[str, Any], bbox: tuple[float, float, float, float]) -> np.ndarray:
    """JRC Global Surface Water seasonality == 12, on the scene grid."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    document = _post_json(
        "https://planetarycomputer.microsoft.com/api/stac/v1/search",
        {"collections": ["jrc-gsw"], "bbox": list(bbox), "limit": 4},
    )
    months = np.zeros((profile["height"], profile["width"]), dtype=np.uint8)
    with rasterio.Env(**GDAL_ENV):
        for feature in document.get("features", []):
            href = sign_href(feature["assets"]["seasonality"]["href"], "jrc-gsw")
            with rasterio.open(href) as src:
                west, south, east, north = bbox
                window = rasterio.windows.from_bounds(
                    west - 0.01, south - 0.01, east + 0.01, north + 0.01, transform=src.transform
                ).intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                window = window.round_offsets().round_lengths()
                block = src.read(1, window=window)
                part = np.zeros_like(months)
                reproject(
                    source=block,
                    destination=part,
                    src_transform=src.window_transform(window),
                    src_crs=src.crs,
                    dst_transform=profile["transform"],
                    dst_crs=profile["crs"],
                    resampling=Resampling.nearest,
                )
                months = np.maximum(months, part)
    return months >= PERMANENT_MONTHS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    registry = load_aoi_registry(REPO_ROOT / "configs" / "aoi.yaml")
    aoi = next((a for a in registry.aois if a.id == args.aoi), None)
    if aoi is None:
        raise SystemExit(f"{args.aoi!r} is not a configured study area")

    provider = PlanetaryComputerProvider()
    result = provider.search(
        SearchQuery(
            bbox=aoi.bbox, start=args.start, end=args.end, collection="sentinel-1-rtc", limit=50
        )
    )
    scenes = [s for s in result.scenes if args.scene_id in (None, s.scene_id)]
    if not scenes:
        print(f"BLOCKED: no sentinel-1-rtc scene over {aoi.id} in the window.")
        return 2
    scene = scenes[0]
    print(f"scene  {scene.scene_id}  {scene.acquired_at}")

    linear, _transform, profile = read_window(provider.asset_href(scene, "vv"), aoi.bbox)
    vv_db = to_db(linear)
    print(f"       VV {vv_db.shape[0]}x{vv_db.shape[1]} px read")

    baseline = OtsuHandBaseline()
    water, tile_stats = otsu_by_tile(vv_db, baseline)
    permanent = permanent_water(profile, aoi.bbox)
    observed = np.isfinite(vv_db)
    flood = water & ~permanent & observed

    pixel_km2, area_method = _pixel_area_km2(profile)
    water_km2 = float(water.sum() * pixel_km2)
    permanent_in_water_km2 = float((water & permanent).sum() * pixel_km2)
    flood_km2 = float(flood.sum() * pixel_km2)
    observed_km2 = float(observed.sum() * pixel_km2)

    # Agreement with the U-Net run whose gate failed, on the same grid. Not a
    # validation of either -- neither is ground truth here -- but a large
    # disagreement would be worth knowing about.
    agreement: dict[str, Any] | None = None
    stem = f"{aoi.id}_{scene.acquired_at:%Y%m%dT%H%M%S}"
    unet_mask_path = args.out / f"{stem}_flood_mask.tif"
    if unet_mask_path.is_file():
        import rasterio

        with rasterio.open(unet_mask_path) as src:
            unet = src.read(1).astype(bool)
        if unet.shape == water.shape:
            both = (unet & water & observed).sum()
            either = ((unet | water) & observed).sum()
            agreement = {
                "compared_with": relative_to_repo(unet_mask_path),
                "iou_between_methods": round(float(both / either), 4) if either else None,
                "note": (
                    "Agreement between two predictions, not accuracy. The U-Net "
                    "run failed its distribution gate and neither map has ground "
                    "truth on this scene."
                ),
            }

    args.out.mkdir(parents=True, exist_ok=True)
    otsu_stem = f"{stem}_otsu"
    out_profile = dict(profile)
    _write_geotiff(
        args.out / f"{otsu_stem}_flood_mask.tif", flood.astype(np.uint8), out_profile, dtype="uint8"
    )
    _write_geotiff(
        args.out / f"{otsu_stem}_permanent_water.tif",
        permanent.astype(np.uint8),
        out_profile,
        dtype="uint8",
    )
    # Where the radar actually saw the ground. The flood mask is 0/1 with no
    # nodata, so without this a pixel outside the swath is indistinguishable
    # from a dry one -- and anything downstream would count it as safe.
    _write_geotiff(
        args.out / f"{otsu_stem}_observed.tif",
        observed.astype(np.uint8),
        out_profile,
        dtype="uint8",
    )

    overlay = categorical_overlay(
        flood.astype(np.uint8), profile, PALETTE, OVERLAY_DIR / f"{otsu_stem}_flood.png"
    )

    payload: dict[str, Any] = {
        "analysis": "flood_extent_otsu_scene",
        "hazard": "flood",
        "source_kind": "model",
        "aoi": aoi.id,
        "aoi_name": aoi.name,
        "country": aoi.country,
        "bbox": list(aoi.bbox),
        "processed_at": datetime.now(UTC).isoformat(),
        "observation": {
            "scene_id": scene.scene_id,
            "acquired_at": scene.acquired_at.isoformat() if scene.acquired_at else None,
            "platform": scene.platform,
            "orbit_direction": scene.orbit_direction,
            "relative_orbit": scene.relative_orbit,
            "radiometry": scene.properties["radiometry"],
            "collection": "sentinel-1-rtc",
            "provider": "planetary_computer",
            "resolution_m": 10.0,
        },
        "method": {
            "name": "otsu_baseline",
            "version": baseline.version,
            "band": "VV in dB",
            "tile_px": TILE,
            "min_separability": baseline.min_separability,
            "guard_selected_by": "leave-one-region-out over 11 Sen1Floods11 regions",
            "held_out_india_iou": BASELINE_IOU_HELD_OUT_INDIA,
            "why_not_the_unet": (
                "The U-Net's distribution gate failed on this scene (sigma0 "
                "training data, gamma0 scene). A ratio-only U-Net would pass the "
                "gate but scores IoU 0.3686 on held-out India, below this "
                "baseline. Otsu sets its threshold per tile from the scene "
                "itself, so a radiometric offset does not bias it."
            ),
            "permanent_water": "JRC Global Surface Water v1.3 seasonality = 12 months, subtracted",
            "hand_mask": "not applied (no HAND raster for this area)",
        },
        "tiles": tile_stats,
        "result": {
            "observed_area_km2": round(observed_km2, 2),
            "water_area_km2": round(water_km2, 2),
            "permanent_water_removed_km2": round(permanent_in_water_km2, 2),
            "flood_area_km2": round(flood_km2, 2),
            "flood_fraction_of_observed": round(flood_km2 / observed_km2, 6)
            if observed_km2
            else None,
            "pixel_area_method": area_method,
        },
        "agreement_with_unet": agreement,
        "overlay": {"url": f"/layers/{overlay.path.name}", **overlay.describe()},
        "artifacts": {
            "flood_mask": relative_to_repo(args.out / f"{otsu_stem}_flood_mask.tif"),
            "permanent_water": relative_to_repo(args.out / f"{otsu_stem}_permanent_water.tif"),
            "observed": relative_to_repo(args.out / f"{otsu_stem}_observed.tif"),
            "overlay_png": relative_to_repo(overlay.path),
        },
        "caveats": [
            f"The method's measured skill is IoU {BASELINE_IOU_HELD_OUT_INDIA} on "
            "held-out Indian chips. It is not an accuracy figure for this scene: "
            "no ground truth exists here.",
            "No HAND mask is applied, so low-backscatter surfaces other than water "
            "(smooth bare fields, radar shadow on steep slopes) can be called water.",
            "Permanent water uses JRC's 1984-2020 record. A river channel that "
            "moved after 2020 is not recognised as permanent and appears as flood.",
            "MODEL-DERIVED from one acquisition at 00:12 UTC on 27 September 2024. "
            "Not an official flood extent; UNOSAT published its own for this event.",
        ],
    }
    (args.out / f"{otsu_stem}_metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print(
        f"       tiles: {tile_stats['tiles_thresholded']} thresholded, "
        f"{tile_stats['tiles_called_unimodal']} unimodal, {tile_stats['tiles_without_data']} empty"
    )
    print(
        f"       water {water_km2:,.1f} km2, of which permanent {permanent_in_water_km2:,.1f} km2"
    )
    print(f"       flood {flood_km2:,.1f} km2 of {observed_km2:,.1f} km2 observed")
    if agreement:
        print(f"       agreement with gate-failed U-Net: IoU {agreement['iou_between_methods']}")
    print(f"\nwritten to {relative_to_repo(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
