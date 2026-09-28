"""Run the trained flood model and write the artifacts the serving plane reads.

    python -m ml.flood.predict --checkpoint models/flood/flood_unet_loro_india_sar_ratio_best.pt \
        --chips data/raw/sen1floods11 --region India --limit 8

Batch inference, deliberately. The serving plane never runs this: a 512x512
segmentation pass over a real AOI is seconds of GPU work and hundreds of
megabytes of raster, which is neither a Vercel function's job nor its memory
budget (ADR-001). This writes files; FastAPI reads them.

Outputs per chip, plus one summary:

    <chip>_flood_probability.tif   float32 GeoTIFF, tiled, LZW, overviews
    <chip>_flood_extent.geojson    polygons above the threshold, EPSG:4326
    <chip>_metadata.json           model, version, threshold, scene, areas
    prediction_manifest.json       every chip, and what was skipped and why

**Affected area is computed on the projected grid, not on degrees.** A square
degree is not a square kilometre and its size changes with latitude; measuring
inundation in degrees would be wrong by tens of percent across the study areas
this project covers. The UTM zone comes from the AOI registry.

Every artifact carries provenance: which model at which version, which
checkpoint, which threshold, the fold the model was trained on and the region
it was tested on. A raster whose origin is unknown cannot be defended.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
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
import torch

from ml.flood.dataset import BandSelection, build_reader, selection_for_bands, stack_with_ratio
from ml.flood.evaluate import load_checkpoint

DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "flood"


def _write_geotiff(path: Path, array: np.ndarray, profile: dict[str, Any], *, dtype: str) -> None:
    """Write a tiled, compressed GeoTIFF with overviews.

    Tiled with internal overviews rather than a plain strip TIFF: this is what
    makes a raster cloud-optimised, so a web client can fetch the one tile it is
    displaying instead of the whole file. A 512x512 chip does not need it; an
    AOI mosaic does, and writing them the same way means the serving path does
    not change when the inputs grow.
    """
    import rasterio
    from rasterio.enums import Resampling

    out = dict(profile)
    out.update(
        driver="GTiff",
        dtype=dtype,
        count=1,
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        predictor=3 if dtype.startswith("float") else 2,
    )
    # The source profile carries the SAR scene's own nodata, which is NaN --
    # valid for float32 and rejected outright for uint8. Carried over blindly it
    # aborts the write. The mask has no nodata concept anyway: 0 means "not
    # flooded", not "not observed", and conflating the two would turn every dry
    # pixel into a gap.
    if dtype.startswith("float"):
        out["nodata"] = float("nan")
    else:
        out.pop("nodata", None)
    with rasterio.open(path, "w", **out) as dst:
        dst.write(array.astype(dtype), 1)
        dst.build_overviews([2, 4, 8], Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")


def _polygonise(
    mask: np.ndarray, profile: dict[str, Any], probability: np.ndarray
) -> list[dict[str, Any]]:
    """Vectorise the binary mask, carrying mean probability per polygon.

    Only shapes where the mask is True are emitted. The mean probability is
    attached because a polygon at 0.51 and one at 0.99 are different claims,
    and a bare outline loses that entirely.
    """
    import rasterio.features
    from rasterio.warp import transform_geom

    features: list[dict[str, Any]] = []
    for geometry, value in rasterio.features.shapes(
        mask.astype(np.uint8), mask=mask, transform=profile["transform"]
    ):
        if not value:
            continue
        # Stored in EPSG:4326 regardless of the source CRS: GeoJSON's own
        # specification says coordinates are WGS84 lon/lat, and a viewer that
        # trusts that would place a UTM-coordinate file in the Gulf of Guinea.
        if profile["crs"] and profile["crs"].to_epsg() != 4326:
            geometry = transform_geom(profile["crs"], "EPSG:4326", geometry)
        features.append({"type": "Feature", "geometry": geometry, "properties": {}})

    if features:
        flooded = probability[mask]
        mean_probability = float(flooded.mean()) if flooded.size else None
        for feature in features:
            feature["properties"] = {
                "class": "flood_extent",
                "mean_probability_scene": round(mean_probability, 4)
                if mean_probability is not None
                else None,
            }
    return features


def _pixel_area_km2(profile: dict[str, Any]) -> tuple[float, str]:
    """Area of one pixel in km2, and how it was obtained.

    Sen1Floods11 chips are stored in EPSG:4326, so pixel size is in degrees. A
    degree of longitude shrinks with the cosine of latitude -- about 111 km at
    the equator and 96 km at 30 N -- so treating the grid as square would
    overstate area in northern India by roughly 13 %. The correction is applied
    at the chip's own centre latitude.
    """
    transform = profile["transform"]
    crs = profile["crs"]

    if crs and crs.to_epsg() != 4326:
        # Already projected: pixel size is in metres and needs no correction.
        return abs(transform.a * transform.e) / 1_000_000.0, "projected_crs"

    height = profile["height"]
    centre_lat = transform.f + transform.e * (height / 2.0)
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * math.cos(math.radians(centre_lat))
    return abs(transform.a * km_per_deg_lon * transform.e * km_per_deg_lat), "cosine_latitude"


@torch.no_grad()
def predict_chip(
    model: torch.nn.Module,
    normalizer: Any,
    reader: Any,
    chip: Any,
    bands: tuple[str, ...],
    device: torch.device,
    with_ratio: bool | BandSelection,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Probability, validity mask and the source raster profile for one chip."""
    import rasterio

    features, _labels, valid = reader.load(chip)
    stacked = stack_with_ratio(features, with_ratio)
    normalised = normalizer.transform_stack(stacked, list(bands))
    normalised = np.nan_to_num(normalised, nan=0.0, posinf=0.0, neginf=0.0)

    tensor = torch.from_numpy(normalised.astype(np.float32))[None].to(device)
    probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    with rasterio.open(reader.s1_path(chip)) as src:
        profile = src.profile.copy()
    return probability, valid, profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chips", type=Path, default=REPO_ROOT / "data/raw/sen1floods11")
    parser.add_argument("--region", default=None, help="Only chips from this region")
    parser.add_argument("--limit", type=int, default=0, help="0 = every matching chip")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-geojson", action="store_true", help="Skip vectorisation")
    args = parser.parse_args(argv)

    if not 0.0 < args.threshold < 1.0:
        raise SystemExit(f"threshold must be strictly between 0 and 1, got {args.threshold}")

    device = torch.device(args.device)
    model, normalizer, payload = load_checkpoint(args.checkpoint, device)
    selection = selection_for_bands(payload["bands"])
    with_ratio = selection
    bands = selection.bands

    reader = build_reader(args.chips)
    chips = reader.available(reader.discover())
    if args.region:
        chips = [c for c in chips if c.region == args.region]
    if args.limit:
        chips = chips[: args.limit]
    if not chips:
        raise SystemExit(f"no chips matched region={args.region!r} under {args.chips}")

    args.out.mkdir(parents=True, exist_ok=True)
    produced: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    print(f"checkpoint {args.checkpoint.name}  fold {payload['fold']}  threshold {args.threshold}")
    print(f"{len(chips)} chip(s) -> {args.out}")

    for index, chip in enumerate(chips, start=1):
        try:
            probability, valid, profile = predict_chip(
                model, normalizer, reader, chip, bands, device, with_ratio
            )
        except Exception as exc:  # noqa: BLE001 - one bad chip must not end the run
            skipped.append({"chip": chip.key, "reason": f"{type(exc).__name__}: {exc}"})
            continue

        mask = (probability >= args.threshold) & valid
        pixel_km2, area_method = _pixel_area_km2(profile)
        flooded_km2 = float(mask.sum()) * pixel_km2
        observed_km2 = float(valid.sum()) * pixel_km2

        stem = f"{chip.region}_{chip.chip_id}"
        _write_geotiff(
            args.out / f"{stem}_flood_probability.tif",
            np.where(valid, probability, np.nan),
            profile,
            dtype="float32",
        )
        _write_geotiff(
            args.out / f"{stem}_flood_mask.tif", mask.astype(np.uint8), profile, dtype="uint8"
        )

        n_polygons = 0
        if not args.no_geojson:
            features = _polygonise(mask, profile, probability)
            n_polygons = len(features)
            (args.out / f"{stem}_flood_extent.geojson").write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "crs": {
                            "type": "name",
                            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
                        },
                        "features": features,
                    }
                ),
                encoding="utf-8",
            )

        metadata = {
            "chip": chip.key,
            "region": chip.region,
            "source_kind": "model",
            "model": "flood_unet",
            "model_version": f"1.0.0+{payload['fold']}",
            "checkpoint": relative_to_repo(args.checkpoint),
            "trained_fold": payload["fold"],
            "bands": payload["bands"],
            "threshold": args.threshold,
            "computed_at": datetime.now(UTC).isoformat(),
            "flooded_area_km2": round(flooded_km2, 4),
            "observed_area_km2": round(observed_km2, 4),
            "flooded_fraction_of_observed": (
                round(flooded_km2 / observed_km2, 4) if observed_km2 else None
            ),
            "valid_pixel_fraction": round(float(valid.mean()), 4),
            "n_polygons": n_polygons,
            "pixel_area_method": area_method,
            "caveats": [
                "Model output, not an observation. The satellite measured "
                "backscatter; the flood extent is this model's interpretation of it.",
                "SAT-AI prototype output. NOT an official warning. Official "
                "warnings for India come from IMD, NDMA and State Disaster "
                "Management Authorities.",
                "Known failure mode: urban double-bounce RAISES backscatter over "
                "flooded streets, inverting the signature this model relies on.",
                f"Areas use a {area_method.replace('_', '-')} pixel-size correction; "
                "the grid is geographic, so a square degree is not a square kilometre.",
                *(
                    [
                        f"This chip is from {chip.region}, which was in the TRAINING "
                        f"partition of fold {payload['fold']}. Its output is not a "
                        f"generalisation estimate."
                    ]
                    if chip.region not in (payload.get("test_regions") or [chip.region])
                    else []
                ),
            ],
        }
        (args.out / f"{stem}_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        produced.append(metadata)

        if index % 5 == 0 or index == len(chips):
            print(f"  {index}/{len(chips)}", end="\r", flush=True)

    total_flooded = sum(m["flooded_area_km2"] for m in produced)
    total_observed = sum(m["observed_area_km2"] for m in produced)
    manifest = {
        "run_at": datetime.now(UTC).isoformat(),
        "checkpoint": relative_to_repo(args.checkpoint),
        "trained_fold": payload["fold"],
        "region_filter": args.region,
        "threshold": args.threshold,
        "n_chips_predicted": len(produced),
        "n_chips_skipped": len(skipped),
        "skipped": skipped,
        "total_flooded_area_km2": round(total_flooded, 4),
        "total_observed_area_km2": round(total_observed, 4),
        "chips": produced,
    }
    (args.out / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(
        f"\n{len(produced)} chip(s) predicted, {len(skipped)} skipped\n"
        f"  flooded {total_flooded:,.1f} km2 of {total_observed:,.1f} km2 observed "
        f"({total_flooded / total_observed:.1%})"
        if total_observed
        else f"\n{len(produced)} chip(s) predicted"
    )
    print(f"  artifacts in {relative_to_repo(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
