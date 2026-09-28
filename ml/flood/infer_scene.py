"""Run the trained flood model over a real Sentinel-1 scene, for a real AOI.

    python -m ml.flood.infer_scene --aoi nepal_koshi_terai \
        --start 2024-09-25 --end 2024-09-29 \
        --checkpoint models/flood/flood_unet_loro_india_sar_ratio_best.pt

Until now the model had only ever seen Sen1Floods11 chips. This is the path
from a catalogue query to a map layer: search the Planetary Computer for a real
acquisition over a configured study area, read the two polarisations as a
windowed COG read, run the U-Net, and write artifacts that carry where every
number came from.

The domain shift, handled rather than hidden
--------------------------------------------
The model was trained on Earth Engine `COPERNICUS/S1_GRD` — **sigma0**,
ellipsoid-corrected. The Planetary Computer's analysis-ready collection is
**gamma0 RTC**, terrain-flattened. They are different radiometric quantities. A
model applied across that gap can produce a perfectly plausible-looking map that
means nothing.

So this command does not simply predict. It first runs the project's own Track
A/B distribution gate (ADR-010, `satai.ml.distribution_gate`) comparing the
scene's band distributions against the training chips the model actually saw,
and it writes the verdict into the artifact beside the prediction. When the gate
fails, the artifacts are still written — a refused prediction teaches nothing —
but they are labelled with the failure, and `--require-gate` makes the failure
fatal for callers that need it to be.

What is deliberately not done here
----------------------------------
No HAND mask, no permanent-water exclusion, no speckle filter. Each would change
the numbers, and the model was trained without them; adding one at inference
only would mean evaluating a different system from the one that was measured.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
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
import torch

from ml.flood.dataset import bands_for, build_reader, stack_with_ratio
from ml.flood.evaluate import load_checkpoint
from ml.flood.predict import _pixel_area_km2, _polygonise, _write_geotiff
from satai.geo.aoi import load_aoi_registry
from satai.logging import get_logger
from satai.ml.distribution_gate import run_gate
from satai.providers import PlanetaryComputerProvider, SearchQuery

log = get_logger(__name__)

DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "flood_scenes"

#: Tile size the model was trained at. Inference uses the same size because a
#: U-Net with BatchNorm is not scale-invariant: running a 4000 px scene through
#: in one piece changes the statistics every normalisation layer sees.
TILE = 512

#: Overlap between tiles, blended on the way out. Without it the seams show as
#: straight lines of disagreement that a reader would mistake for a river.
OVERLAP = 64

#: Below this, `10*log10` produces -inf. RTC stores zero for pixels outside the
#: swath or shadowed by terrain; they become nodata, not very-dry ground.
FLOOR_LINEAR = 1e-7


def to_db(linear: np.ndarray) -> np.ndarray:
    """Linear gamma0 to decibels, with the invalid pixels marked as such."""
    out = np.full(linear.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(linear) & (linear > FLOOR_LINEAR)
    out[valid] = 10.0 * np.log10(linear[valid])
    return out


#: GDAL settings for reading a ~2 GB COG across the public internet. Set inside
#: a `rasterio.Env` rather than exported, so the behaviour travels with the code
#: instead of depending on how the process happened to be launched.
GDAL_ENV: dict[str, Any] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    # No CPL_VSIL_CURL_ALLOWED_EXTENSIONS: the signed asset ends ".rtc.tiff"
    # and carries a query string, so an extension allow-list rejects the very
    # file this module exists to read.
    "GDAL_HTTP_MAX_RETRY": "10",
    "GDAL_HTTP_RETRY_DELAY": "3",
    "GDAL_HTTP_TIMEOUT": "120",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "134217728",
    "GDAL_CACHEMAX": 512,
}

#: Rows per read. An AOI this size is ~10^8 pixels, and asking for all of it in
#: one call means a single dropped tile 90 % of the way through discards the
#: whole transfer. Striping makes a failure cost one stripe.
STRIPE_ROWS = 1024


def read_window(
    href: str, bbox: tuple[float, float, float, float], *, attempts: int = 4
) -> tuple[np.ndarray, Any, Any]:
    """Read one band over `bbox` (EPSG:4326) straight from the remote COG.

    Striped, with per-stripe retry. The remote store drops connections often
    enough at this size that a single `src.read` of the whole window fails more
    often than it succeeds, and the failure arrives after several minutes of
    transfer.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        height, width = int(window.height), int(window.width)
        transform = src.window_transform(window)
        profile = src.profile.copy()

        data = np.zeros((height, width), dtype=np.float32)
        for top in range(0, height, STRIPE_ROWS):
            rows = min(STRIPE_ROWS, height - top)
            stripe = Window(window.col_off, window.row_off + top, width, rows)
            for attempt in range(attempts):
                try:
                    data[top : top + rows] = src.read(
                        1, window=stripe, boundless=True, fill_value=0
                    ).astype(np.float32)
                    break
                except rasterio.errors.RasterioIOError as exc:
                    if attempt == attempts - 1:
                        raise
                    log.warning(
                        "stripe read failed, retrying",
                        extra={"row_off": top, "attempt": attempt + 1, "error": str(exc)[:100]},
                    )
                    time.sleep(2.0 * (attempt + 1))
            done = min(top + rows, height)
            print(f"          {done:>6,}/{height:,} rows", end="\r", flush=True)

    profile.update(height=height, width=width, transform=transform, count=1, dtype="float32")
    return data, transform, profile


@torch.no_grad()
def predict_tiled(model: torch.nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    """Probability over a whole scene, tile by tile, overlaps averaged.

    The accumulator is a weighted mean rather than a last-writer-wins paste: a
    pixel at the edge of one tile is at the centre of its neighbour, and the
    centre prediction is the better-conditioned of the two.
    """
    _, height, width = features.shape
    total = np.zeros((height, width), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)
    step = TILE - OVERLAP

    for top in range(0, max(height - OVERLAP, 1), step):
        for left in range(0, max(width - OVERLAP, 1), step):
            bottom, right = min(top + TILE, height), min(left + TILE, width)
            patch = features[:, top:bottom, left:right]

            # Pad short edge tiles; the padding is dropped again below, so it
            # never contributes to the output.
            pad_y, pad_x = TILE - patch.shape[1], TILE - patch.shape[2]
            if pad_y or pad_x:
                patch = np.pad(patch, ((0, 0), (0, pad_y), (0, pad_x)), mode="reflect")

            tensor = torch.from_numpy(patch[None].astype(np.float32)).to(device)
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

            total[top:bottom, left:right] += probability[: bottom - top, : right - left]
            weight[top:bottom, left:right] += 1.0

    return (total / np.maximum(weight, 1.0)).astype(np.float32)


def training_reference(
    chips_root: Path, bands: tuple[str, ...], limit: int
) -> dict[str, np.ndarray]:
    """Band samples from the chips the model was actually trained on.

    The gate needs something to compare against, and the only defensible
    reference is the training distribution itself -- not a published figure for
    what Sentinel-1 backscatter "should" look like.
    """
    reader = build_reader(chips_root)
    chips = reader.available(reader.discover())[:limit]
    if not chips:
        raise SystemExit(f"no Sen1Floods11 chips under {chips_root} to compare against")

    stacks = []
    for chip in chips:
        features, _labels, valid = reader.load(chip)
        stack = stack_with_ratio(features, len(bands) == 3)
        stacks.append(np.where(valid[None], stack, np.nan))
    combined = np.concatenate(stacks, axis=2)
    return {band: combined[i] for i, band in enumerate(bands)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", required=True, help="study-area id from configs/aoi.yaml")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene-id", default=None, help="pin one scene instead of the first hit")
    parser.add_argument("--collection", default="sentinel-1-rtc")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chips", type=Path, default=REPO_ROOT / "data/raw/sen1floods11")
    parser.add_argument("--gate-chips", type=int, default=24)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--require-gate",
        action="store_true",
        help="exit non-zero when the distribution gate refuses the transfer",
    )
    args = parser.parse_args(argv)

    registry = load_aoi_registry(REPO_ROOT / "configs" / "aoi.yaml")
    aoi = next((a for a in registry.aois if a.id == args.aoi), None)
    if aoi is None:
        raise SystemExit(
            f"{args.aoi!r} is not a configured study area. Configured: "
            f"{', '.join(a.id for a in registry.aois)}"
        )

    # --- 1. find a real acquisition -----------------------------------------
    provider = PlanetaryComputerProvider()
    result = provider.search(
        SearchQuery(
            bbox=aoi.bbox, start=args.start, end=args.end, collection=args.collection, limit=50
        )
    )
    scenes = [s for s in result.scenes if args.scene_id in (None, s.scene_id)]
    if not scenes:
        print(
            f"BLOCKED: no {args.collection} scene over {aoi.id} between "
            f"{args.start} and {args.end}. Nothing was predicted."
        )
        return 2
    scene = scenes[0]
    print(f"scene   {scene.scene_id}")
    print(
        f"        acquired {scene.acquired_at}  {scene.orbit_direction}  "
        f"rel orbit {scene.relative_orbit}"
    )
    print(f"        radiometry {scene.properties['radiometry']}")

    # --- 2. read the two polarisations over the AOI -------------------------
    stack_db = []
    profile = None
    for band in ("vv", "vh"):
        href = provider.asset_href(scene, band)
        linear, _transform, profile = read_window(href, aoi.bbox)
        stack_db.append(to_db(linear))
        print(f"        {band}: {linear.shape[0]}x{linear.shape[1]} px read from the COG")
    if profile is None:  # pragma: no cover - the loop above always runs twice
        raise SystemExit("no band was read; nothing to predict from")

    features_db = stack_with_ratio(np.stack(stack_db), with_ratio=True)
    observed = np.isfinite(features_db).all(axis=0)
    if not observed.any():
        print("BLOCKED: every pixel in the AOI window is outside the swath or shadowed.")
        return 2

    # --- 3. the gate, before anything is predicted --------------------------
    model, normalizer, payload = load_checkpoint(args.checkpoint, torch.device(args.device))
    bands = bands_for("vv_vh_ratio" in payload["bands"])

    reference = training_reference(args.chips, bands, args.gate_chips)
    gate = run_gate(
        {b: reference[b] for b in bands},
        {b: np.where(observed, features_db[i], np.nan) for i, b in enumerate(bands)},
        track_a_source=(
            f"Sen1Floods11 HandLabeled ({args.gate_chips} chips), sigma0 dB, GEE S1_GRD"
        ),
        track_b_source=f"{scene.scene_id}, {scene.properties['radiometry']} -> dB",
    )
    print(f"\ngate    {gate.verdict}")
    for comparison in gate.bands:
        print(f"        {comparison.summary()}")

    if args.require_gate and not gate.may_proceed:
        print(
            "\nBLOCKED by --require-gate: the scene's distribution differs from "
            "the training distribution by more than the gate allows, so a "
            "prediction here would not be the model that was measured."
        )
        return 3

    # --- 4. inference --------------------------------------------------------
    normalised = normalizer.transform_stack(features_db, list(bands))
    normalised = np.nan_to_num(normalised, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    probability = predict_tiled(model, normalised, torch.device(args.device))
    probability = np.where(observed, probability, np.nan).astype(np.float32)

    mask = np.nan_to_num(probability, nan=0.0) >= args.threshold
    pixel_km2, area_method = _pixel_area_km2(profile)
    flooded_km2 = float(mask.sum() * pixel_km2)
    observed_km2 = float(observed.sum() * pixel_km2)

    # --- 5. artifacts --------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{aoi.id}_{scene.acquired_at:%Y%m%dT%H%M%S}"
    probability_path = args.out / f"{stem}_flood_probability.tif"
    mask_path = args.out / f"{stem}_flood_mask.tif"
    geojson_path = args.out / f"{stem}_flood_extent.geojson"

    _write_geotiff(probability_path, probability, profile, dtype="float32")
    _write_geotiff(mask_path, mask.astype(np.uint8), profile, dtype="uint8")
    features = _polygonise(mask, profile, np.nan_to_num(probability, nan=0.0))
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )

    metadata: dict[str, Any] = {
        "aoi": aoi.id,
        "aoi_name": aoi.name,
        "country": aoi.country,
        "bbox": list(aoi.bbox),
        "source_kind": "model",
        "hazard": "flood",
        # -- observation
        "scene_id": scene.scene_id,
        "collection": args.collection,
        "provider": "planetary_computer",
        "platform": scene.platform,
        "acquired_at": scene.acquired_at.isoformat() if scene.acquired_at else None,
        "orbit_direction": scene.orbit_direction,
        "relative_orbit": scene.relative_orbit,
        "radiometry": scene.properties["radiometry"],
        "polarizations": scene.properties["polarizations"],
        "instrument_mode": scene.properties["instrument_mode"],
        "resolution_m": 10.0,
        "crs": str(profile["crs"]),
        # -- model
        "model": "flood_unet",
        "model_version": f"1.0.0+{payload['fold']}",
        "checkpoint": relative_to_repo(args.checkpoint),
        "bands": list(bands),
        "threshold": args.threshold,
        "normalizer_fold": payload["fold"],
        "test_iou_on_held_out_india": 0.5230,
        # -- result
        "processed_at": datetime.now(UTC).isoformat(),
        "grid": {"height": int(profile["height"]), "width": int(profile["width"])},
        "pixel_area_km2": pixel_km2,
        "pixel_area_method": area_method,
        "observed_area_km2": round(observed_km2, 4),
        "flooded_area_km2": round(flooded_km2, 4),
        "flooded_fraction_of_observed": round(flooded_km2 / observed_km2, 6)
        if observed_km2
        else None,
        "n_polygons": len(features),
        "artifacts": {
            "probability": relative_to_repo(probability_path),
            "mask": relative_to_repo(mask_path),
            "extent_geojson": relative_to_repo(geojson_path),
        },
        # -- the honesty layer
        "distribution_gate": gate.to_dict(),
        "caveats": [
            f"The model was trained on Sen1Floods11 sigma0 (Earth Engine "
            f"COPERNICUS/S1_GRD). This scene is {scene.properties['radiometry']}. "
            f"They are different radiometric quantities; the distribution gate "
            f"above is the measurement of that gap, not a reassurance about it.",
            "IoU 0.5230 is this model's score on held-out India chips from "
            "Sen1Floods11. It is NOT an accuracy claim for this scene: no "
            "ground truth exists here, so nothing has been scored.",
            "No HAND mask, no permanent-water exclusion and no speckle filter "
            "are applied, because none was applied in training. Permanent water "
            "bodies are therefore included in the flood extent.",
            "MODEL-DERIVED. Not an observation of flooding, and not an official "
            "warning. Warnings come from the national and state disaster "
            "management authorities.",
        ],
    }
    metadata_path = args.out / f"{stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\n        observed {observed_km2:,.1f} km2 of the AOI window")
    print(
        f"        flooded  {flooded_km2:,.1f} km2 "
        f"({100 * flooded_km2 / observed_km2:.1f}% of observed)"
        if observed_km2
        else ""
    )
    print(f"        {len(features)} polygon(s)")
    print(f"\nwritten to {relative_to_repo(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
