"""Burn severity from a real Sentinel-2 pre/post pair (dNBR).

    python -m ml.hazards.run_burn_severity --aoi uttarakhand_kumaon \
        --pre 2024-03-31 --post 2024-05-15 --tile 44RKT

`satai.hazards.wildfire` could always compute NBR, dNBR and the USGS severity
classes. It never had imagery. The Planetary Computer serves Sentinel-2 L2A
without an account, which makes this the first burned-area product in the
project -- and the first wildfire result that is more than active-fire points.

The event
---------
The 2024 Uttarakhand fire season. ISRO's Indian Institute of Remote Sensing
reports two peaks, in the first and third weeks of April 2024, with Nainital
at 1,715 fire incidents against 316 in 2023 and Almora at 965 against 443. Both
districts are inside the Kumaon study area. The pre-fire image is the last
cloud-free acquisition before the first peak; the post-fire image is the first
fully clear one after the second peak and before the monsoon.

What this measures, and what it cannot
--------------------------------------
dNBR is the drop in the Normalised Burn Ratio between the two dates. It is the
standard burn-severity index, and it is also sensitive to things that are not
fire: six weeks of pre-monsoon drying lowers NBR on unburned slopes, and the
sun is higher in May than in March, which changes terrain shadow in the
Himalayan foothills. The classes are therefore labelled with the USGS names
they correspond to, and every artifact says they have not been validated
against field plots. This is post-fire burn assessment. It is not ignition
prediction, and it is not a fire-danger forecast.
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

from ml.flood.infer_scene import GDAL_ENV
from ml.hazards.overlay import categorical_overlay
from satai.geo.aoi import load_aoi_registry
from satai.hazards.wildfire import SEVERITY_CLASSES, classify_severity, dnbr, nbr
from satai.providers import PlanetaryComputerProvider, SearchQuery

DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "wildfire"
OVERLAY_DIR = REPO_ROOT / "frontend" / "public" / "layers"

#: Scene classification codes that are kept. Everything else -- no data,
#: saturation, cloud shadow, water, cloud, cirrus, snow -- is masked, because
#: each changes NBR for reasons unrelated to fire. Code 2 ("dark area pixels")
#: is kept deliberately: fresh burn scars are dark, and masking them would
#: remove the signal this module exists to measure.
KEEP_SCL = (2, 4, 5, 7)

#: Colours for the overlay, one per USGS class, index-aligned with
#: SEVERITY_CLASSES. Unburned and regrowth are left transparent so the map shows
#: only what burned.
SEVERITY_PALETTE: dict[int, tuple[int, int, int, int]] = {
    # "low" (code 2) is deliberately not drawn: on this pair it is as large as
    # its mirror image, enhanced regrowth, which marks it as seasonal noise.
    3: (247, 146, 38, 205),  # moderate-low
    4: (221, 72, 32, 215),  # moderate-high
    5: (140, 20, 30, 225),  # high
}

#: Area of one 20 m Sentinel-2 pixel, in km2.
PIXEL_KM2 = 0.02 * 0.02


def reflectance(dn: np.ndarray, baseline: str | None) -> np.ndarray:
    """L2A digital numbers to surface reflectance, offset applied.

    From processing baseline 04.00 (January 2022) Sentinel-2 L2A carries
    BOA_ADD_OFFSET = -1000. NBR is a ratio, so an unapplied offset does not
    cancel: it pulls every value towards zero and shrinks every dNBR.
    """
    offset = -1000.0 if baseline and float(baseline) >= 4.0 else 0.0
    values = dn.astype(np.float64)
    out = (values + offset) / 10_000.0
    out[values == 0] = np.nan
    return out


def read_band(
    href: str, bbox: tuple[float, float, float, float], shape: tuple[int, int] | None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one band over the AOI part of the tile, optionally onto a given grid."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        bounds = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        window = from_bounds(*bounds, transform=src.transform).intersection(
            rasterio.windows.Window(0, 0, src.width, src.height)
        )
        window = window.round_offsets().round_lengths()
        out_shape = shape or (int(window.height), int(window.width))
        data = src.read(
            1,
            window=window,
            out_shape=out_shape,
            resampling=Resampling.average if shape else Resampling.nearest,
        )
        # The window's own transform, rescaled when the band was resampled onto
        # a coarser grid (10 m NIR onto the 20 m SWIR grid).
        base = src.window_transform(window)
        transform = rasterio.Affine(
            base.a * window.width / out_shape[1],
            base.b,
            base.c,
            base.d,
            base.e * window.height / out_shape[0],
            base.f,
        )
        profile = {
            "crs": src.crs,
            "transform": transform,
            "height": out_shape[0],
            "width": out_shape[1],
        }
    return data, profile


#: ESA WorldCover class for tree cover. Burn severity is computed over forest
#: only: the April wheat harvest in the foothills drops NBR just as a fire does,
#: and over cropland dNBR measures the harvest calendar, not burning.
WORLDCOVER_TREE = 10
WORLDCOVER_ITEM = "ESA_WorldCover_10m_2021_v200_N27E078"


def worldcover_on_grid(profile: dict[str, Any]) -> np.ndarray:
    """ESA WorldCover 2021 resampled onto the analysis grid (mode resampling)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    from satai.providers.planetary import _post_json, sign_href

    document = _post_json(
        "https://planetarycomputer.microsoft.com/api/stac/v1/search",
        {"collections": ["esa-worldcover"], "ids": [WORLDCOVER_ITEM], "limit": 1},
    )
    href = sign_href(document["features"][0]["assets"]["map"]["href"], "esa-worldcover")
    destination = np.zeros((profile["height"], profile["width"]), dtype=np.uint8)
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            resampling=Resampling.mode,
        )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", required=True)
    parser.add_argument("--pre", type=date.fromisoformat, required=True)
    parser.add_argument("--post", type=date.fromisoformat, required=True)
    parser.add_argument("--tile", required=True, help="MGRS tile, e.g. 44RKT")
    parser.add_argument("--max-cloud", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    registry = load_aoi_registry(REPO_ROOT / "configs" / "aoi.yaml")
    aoi = next((a for a in registry.aois if a.id == args.aoi), None)
    if aoi is None:
        raise SystemExit(f"{args.aoi!r} is not a configured study area")

    provider = PlanetaryComputerProvider()

    def find(day: date) -> Any | None:
        result = provider.search(
            SearchQuery(
                bbox=aoi.bbox,
                start=day - timedelta(days=1),
                end=day + timedelta(days=1),
                collection="sentinel-2-l2a",
                max_cloud_cover=args.max_cloud,
                limit=50,
            )
        )
        hits = [s for s in result.scenes if f"_T{args.tile}_" in s.scene_id]
        return min(hits, key=lambda s: s.cloud_cover or 100.0) if hits else None

    pre, post = find(args.pre), find(args.post)
    if pre is None or post is None:
        print(f"BLOCKED: no clear T{args.tile} scene on {'pre' if pre is None else 'post'} date.")
        return 2

    print(f"pre   {pre.scene_id}  cloud {pre.cloud_cover:.1f}%")
    print(f"post  {post.scene_id}  cloud {post.cloud_cover:.1f}%")

    # SWIR2 and the classification layer are 20 m; NIR (B08) is 10 m and is
    # averaged onto the 20 m grid so the two bands describe the same ground.
    def load(scene: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        swir_dn, profile = read_band(provider.asset_href(scene, "B12"), aoi.bbox, None)
        shape = (profile["height"], profile["width"])
        nir_dn, _ = read_band(provider.asset_href(scene, "B08"), aoi.bbox, shape)
        scl, _ = read_band(provider.asset_href(scene, "SCL"), aoi.bbox, shape)
        baseline = scene.properties.get("processing_baseline")
        value = nbr(reflectance(nir_dn, baseline), reflectance(swir_dn, baseline))
        clear = np.isin(scl, KEEP_SCL)
        return np.where(clear, value, np.nan), clear, profile

    pre_nbr, _, profile = load(pre)
    post_nbr, _, _ = load(post)
    if pre_nbr.shape != post_nbr.shape:
        raise SystemExit(f"grid mismatch {pre_nbr.shape} vs {post_nbr.shape}")

    raw = dnbr(pre_nbr, post_nbr)
    raw_valid = np.isfinite(raw)
    raw_burned_km2 = float(((raw >= 99.0) & raw_valid).sum() * PIXEL_KM2)

    # Forest only, then remove the seasonal shift. Eight weeks of pre-monsoon
    # drying lowers NBR across the whole landscape; the median dNBR over forest
    # measures that shift, because most of the forest did not burn. This is the
    # dNBR offset of Key and Benson (2006), estimated from the scene rather than
    # from a hand-picked unburned plot.
    landcover = worldcover_on_grid(profile)
    forest = (landcover == WORLDCOVER_TREE) & raw_valid
    if forest.sum() < 1000:
        print("BLOCKED: fewer than 1,000 forest pixels in the window; no offset can be estimated.")
        return 3
    offset = float(np.median(raw[forest]))
    change = np.where(forest, raw - offset, np.nan)

    severity = classify_severity(change)
    codes = np.digitize(change, (-100.0, 99.0, 269.0, 439.0, 659.0)).astype(np.int16)
    codes[~np.isfinite(change)] = -1

    # Burned area counts moderate-low and above only. The low class is tested
    # rather than assumed away: fire lowers NBR and never raises it, so any
    # spread that appears equally on both sides of the offset is not fire. The
    # mirror tail (corrected dNBR at or below -269) estimates how much of the
    # "burned" tail that noise alone would produce.
    finite = np.isfinite(change)
    burned = (change >= 269.0) & finite
    mirror = (change <= -269.0) & finite
    burned_km2 = float(burned.sum() * PIXEL_KM2)
    mirror_km2 = float(mirror.sum() * PIXEL_KM2)
    net_burned_km2 = max(burned_km2 - mirror_km2, 0.0)
    low_km2 = float(((change >= 99.0) & (change < 269.0) & finite).sum() * PIXEL_KM2)
    regrowth_km2 = float(((change < -100.0) & finite).sum() * PIXEL_KM2)
    valid_km2 = float(np.isfinite(change).sum() * PIXEL_KM2)
    forest_km2 = float(forest.sum() * PIXEL_KM2)
    area_by_class = {
        name: round(float(((codes == index) & np.isfinite(change)).sum() * PIXEL_KM2), 3)
        for index, name in enumerate(SEVERITY_CLASSES)
    }

    import rasterio

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{aoi.id}_{args.tile}_{pre.acquired_at:%Y%m%d}_{post.acquired_at:%Y%m%d}"
    raster_profile = {
        "driver": "GTiff",
        "height": profile["height"],
        "width": profile["width"],
        "count": 1,
        "crs": profile["crs"],
        "transform": profile["transform"],
        "compress": "LZW",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(
        args.out / f"{stem}_dnbr.tif", "w", dtype="float32", **raster_profile
    ) as dst:
        dst.write(change.astype("float32"), 1)
    with rasterio.open(
        args.out / f"{stem}_severity.tif", "w", dtype="int16", nodata=-1, **raster_profile
    ) as dst:
        dst.write(codes, 1)

    overlay = categorical_overlay(
        codes, profile, SEVERITY_PALETTE, OVERLAY_DIR / f"{stem}_severity.png"
    )

    payload: dict[str, Any] = {
        "analysis": "burn_severity_dnbr",
        "hazard": "wildfire",
        "source_kind": "derived",
        "not_a_prediction": (
            "Post-fire burn assessment from observed imagery. Not ignition "
            "prediction and not a fire-danger forecast."
        ),
        "aoi": aoi.id,
        "aoi_name": aoi.name,
        "country": aoi.country,
        "run_at": datetime.now(UTC).isoformat(),
        "event": {
            "name": "Uttarakhand forest fires, 2024 season",
            "peaks": "first and third weeks of April 2024",
            "reported_by": "ISRO Indian Institute of Remote Sensing",
            "reference": "https://science.iirs.gov.in/satellite-based-observations-of-forest-fires-2024-in-uttarakhand/",
            "reported_incidents": {"Nainital": 1715, "Almora": 965, "Champawat": 1135},
        },
        "pair": {
            "pre_scene_id": pre.scene_id,
            "pre_acquired_at": pre.acquired_at.isoformat(),
            "pre_cloud_cover": pre.cloud_cover,
            "post_scene_id": post.scene_id,
            "post_acquired_at": post.acquired_at.isoformat(),
            "post_cloud_cover": post.cloud_cover,
            "mgrs_tile": args.tile,
            "separation_days": (post.acquired_at - pre.acquired_at).days,
            "collection": "sentinel-2-l2a",
            "provider": "planetary_computer",
            "processing_baseline": [
                pre.properties.get("processing_baseline"),
                post.properties.get("processing_baseline"),
            ],
        },
        "method": {
            "index": "NBR = (B08 - B12) / (B08 + B12), dNBR = (pre - post) * 1000",
            "resolution_m": 20,
            "classes": "USGS dNBR breaks: -100, 99, 269, 439, 659",
            "masked_scene_classes": "no data, saturated, cloud shadow, water, cloud, cirrus, snow",
            "domain": "ESA WorldCover 2021 tree cover only",
            "offset_correction": (
                "median dNBR over forest subtracted before classification "
                "(dNBR offset, Key and Benson 2006)"
            ),
            "offset_dnbr": round(offset, 2),
        },
        "uncorrected": {
            "burned_area_km2_all_land_cover": round(raw_burned_km2, 2),
            "why_not_used": (
                "Raw dNBR over every land cover flags cropland harvest and "
                "seasonal drying as burned. It is recorded to show the size of "
                "that error, not as a result."
            ),
        },
        "result": {
            "forest_area_km2": round(forest_km2, 2),
            "valid_area_km2": round(valid_km2, 2),
            "burned_area_km2": round(burned_km2, 2),
            "burned_area_definition": "corrected dNBR >= 269 (moderate-low and above)",
            "noise_mirror_km2": round(mirror_km2, 2),
            "net_burned_area_km2": round(net_burned_km2, 2),
            "low_class_km2": round(low_km2, 2),
            "enhanced_regrowth_km2": round(regrowth_km2, 2),
            "low_class_separable_from_noise": bool(low_km2 > 1.5 * regrowth_km2),
            "burned_fraction_of_valid": round(burned_km2 / valid_km2, 6) if valid_km2 else None,
            "area_km2_by_class": area_by_class,
            "fractions": {k: round(v, 6) for k, v in severity.fractions.items()},
            "coverage": round(severity.n_valid / severity.n_total, 6),
        },
        "overlay": {
            "url": f"/layers/{overlay.path.name}",
            **overlay.describe(),
        },
        "artifacts": {
            "dnbr": relative_to_repo(args.out / f"{stem}_dnbr.tif"),
            "severity": relative_to_repo(args.out / f"{stem}_severity.tif"),
            "overlay_png": relative_to_repo(overlay.path),
        },
        "caveats": [
            f"The low-severity class ({low_km2:,.0f} km2) is excluded from burned "
            f"area: it is about as large as its mirror image, enhanced regrowth "
            f"({regrowth_km2:,.0f} km2). Fire only lowers NBR, so variation that "
            "appears equally in both directions is seasonal, not fire.",
            f"The net figure subtracts the mirror tail ({mirror_km2:,.1f} km2 at "
            "corrected dNBR <= -269), which assumes non-fire variation is "
            "symmetric about the offset.",
            f"The seasonal offset ({offset:+.1f} dNBR) is the median over all "
            "forest in the window, which assumes most of that forest did not "
            "burn. Where fire was widespread the offset is overestimated and "
            "burned area underestimated.",
            "Not validated against field plots. The classes are the USGS dNBR "
            "breaks, calibrated elsewhere, applied here unchanged.",
            f"The images are {(post.acquired_at - pre.acquired_at).days} days "
            "apart across the pre-monsoon dry season. Unburned vegetation dries "
            "and its NBR falls, so the low-severity class will include some "
            "drying that is not fire.",
            "Sun elevation is higher in May than in March, so terrain shadow "
            "differs between the two dates on steep Himalayan slopes.",
            "Only the part of the study area inside one Sentinel-2 tile is "
            f"analysed (T{args.tile}); the rest is not covered by this pair.",
            "Burned area here is surface change consistent with fire. It is "
            "not the official burnt-area figure; the state forest department "
            "reports its own.",
        ],
    }
    (args.out / f"{stem}_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"\n      valid area   {valid_km2:,.1f} km2 ({payload['result']['coverage']:.1%} of window)"
    )
    print(f"      burned       {burned_km2:,.1f} km2 (moderate-low and above)")
    print(f"      noise mirror {mirror_km2:,.1f} km2  ->  net {net_burned_km2:,.1f} km2")
    print(
        f"      low {low_km2:,.0f} km2 vs regrowth {regrowth_km2:,.0f} km2 "
        f"(separable: {low_km2 > 1.5 * regrowth_km2})"
    )
    for name in SEVERITY_CLASSES:
        print(f"        {name:<18} {area_by_class[name]:>9,.2f} km2")
    print(
        f"      overlay      {overlay.width}x{overlay.height} px "
        f"at ~{overlay.approx_resolution_m:.0f} m"
    )
    print(f"\nwritten to {relative_to_repo(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
