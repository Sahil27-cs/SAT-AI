"""Co-register ancillary rasters to every Sen1Floods11 chip, for C2's blocked arms.

    python scripts/build_ancillary.py --dem
    python scripts/build_ancillary.py --rain

C2's rainfall and terrain arms were recorded as blocked on "no DEM or rainfall
co-registered to Sen1Floods11 chips". Both products exist on the Planetary
Computer without an account, so what was missing was this step: resampling
each onto each chip's own 512 x 512 grid so the model sees them pixel-aligned
with the backscatter.

``--dem``
    Copernicus DEM GLO-30, bilinear onto the chip grid, plus slope in degrees
    computed on that grid with the pixel spacing in metres (a degree of
    longitude is shorter than a degree of latitude, so the two axes need
    different spacings). HAND is *not* derived: it needs flow routing over the
    whole upstream catchment, not a 5 km chip, and is recorded as such.

``--rain``
    GPM IMERG V06 half-hourly precipitation, accumulated over the 72 hours
    ending at the close of the chip's acquisition day. IMERG is 0.1 degrees,
    about 11 km, so over a 5 km chip the field is effectively one value: this
    arm tests whether chip-level rainfall context helps, not whether rainfall
    texture does.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from ml.flood.dataset import build_reader  # noqa: E402
from ml.flood.infer_scene import GDAL_ENV  # noqa: E402

from satai.paths import relative_to_repo  # noqa: E402
from satai.providers.planetary import _post_json, sign_href  # noqa: E402

CHIPS = REPO_ROOT / "data" / "raw" / "sen1floods11"
ANCILLARY = REPO_ROOT / "data" / "processed" / "ancillary"
CATALOGUE = (
    "https://storage.googleapis.com/sen1floods11/v1.1/catalog/sen1floods11_hand_labeled_source"
)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"


def chip_profile(key: str) -> dict[str, Any]:
    import rasterio

    with rasterio.open(CHIPS / "S1Hand" / f"{key}_S1Hand.tif") as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "height": src.height,
            "width": src.width,
            "bounds": tuple(src.bounds),
        }


def build_dem(keys: list[str]) -> dict[str, Any]:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    out_dir = ANCILLARY / "dem"
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles: dict[str, str] = {}  # item id -> signed href, reused across chips
    written, skipped = 0, 0

    for index, key in enumerate(keys, start=1):
        destination_path = out_dir / f"{key}.tif"
        if destination_path.exists():
            skipped += 1
            continue
        profile = chip_profile(key)
        west, south, east, north = profile["bounds"]
        document = _post_json(
            STAC,
            {"collections": ["cop-dem-glo-30"], "bbox": [west, south, east, north], "limit": 4},
        )
        features = document.get("features", [])
        if not features:
            print(f"\n  {key}: no DEM tile covers this chip")
            continue

        elevation = np.full((profile["height"], profile["width"]), np.nan, dtype=np.float32)
        with rasterio.Env(**GDAL_ENV):
            for feature in features:
                if feature["id"] not in tiles:
                    tiles[feature["id"]] = sign_href(
                        feature["assets"]["data"]["href"], "cop-dem-glo-30"
                    )
                with rasterio.open(tiles[feature["id"]]) as src:
                    # Read only the chip's corner of the 1-degree tile, with a
                    # few pixels of margin so bilinear sampling at the chip edge
                    # has neighbours. Reprojecting from the whole band instead
                    # pulls ~50 MB per tile across the network for a 5 km chip.
                    pad = 3 * abs(src.transform.a)
                    window = rasterio.windows.from_bounds(
                        west - pad, south - pad, east + pad, north + pad, transform=src.transform
                    ).intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                    window = window.round_offsets().round_lengths()
                    if window.width < 1 or window.height < 1:
                        continue
                    block = src.read(1, window=window).astype(np.float32)
                    part = np.full_like(elevation, np.nan)
                    reproject(
                        source=block,
                        destination=part,
                        src_transform=src.window_transform(window),
                        src_crs=src.crs,
                        src_nodata=src.nodata,
                        dst_transform=profile["transform"],
                        dst_crs=profile["crs"],
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                    )
                    # A chip straddling two tiles takes each pixel from
                    # whichever tile covers it; neither overwrites the other.
                    elevation = np.where(np.isnan(elevation), part, elevation)

        # Slope on the chip grid. Spacing in metres per axis: longitude
        # shrinks with the cosine of latitude, latitude does not.
        centre_lat = (south + north) / 2.0
        dx = abs(profile["transform"].a) * 111_320.0 * np.cos(np.radians(centre_lat))
        dy = abs(profile["transform"].e) * 110_574.0
        grad_y, grad_x = np.gradient(elevation.astype(np.float64), dy, dx)
        slope = np.degrees(np.arctan(np.hypot(grad_x, grad_y))).astype(np.float32)

        with rasterio.open(
            destination_path,
            "w",
            driver="GTiff",
            height=profile["height"],
            width=profile["width"],
            count=2,
            dtype="float32",
            crs=profile["crs"],
            transform=profile["transform"],
            compress="LZW",
            nodata=np.nan,
        ) as dst:
            dst.write(elevation, 1)
            dst.write(slope, 2)
            dst.set_band_description(1, "elevation_m")
            dst.set_band_description(2, "slope_deg")
        written += 1
        if index % 25 == 0 or index == len(keys):
            print(f"  dem {index:>3}/{len(keys)}", flush=True)

    return {
        "product": "Copernicus DEM GLO-30",
        "collection": "cop-dem-glo-30",
        "bands": ["elevation_m", "slope_deg"],
        "resampling": "bilinear onto each chip's 10 m grid",
        "written": written,
        "already_present": skipped,
        "not_derived": "HAND (needs flow routing over the upstream catchment)",
    }


def chip_dates(keys: list[str]) -> dict[str, str]:
    """Acquisition day per chip, from Sen1Floods11's own STAC catalogue."""
    cache = CHIPS / "chip_dates.json"
    dates: dict[str, str] = json.loads(cache.read_text()) if cache.exists() else {}
    for key in keys:
        if key in dates:
            continue
        request = urllib.request.Request(  # noqa: S310
            f"{CATALOGUE}/{key}/{key}.json", headers={"User-Agent": "SAT-AI/0.5"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            dates[key] = json.loads(response.read())["properties"]["datetime"][:10]
    cache.write_text(json.dumps(dates, indent=1, sort_keys=True))
    return dates


def build_rain(keys: list[str]) -> dict[str, Any]:
    """72-hour IMERG accumulation per chip.

    IMERG's store is chunked 12 half-hours by the whole globe, so a 72-hour
    window costs twelve global chunks whatever area is asked for. Chips are
    therefore grouped by date and every chip on a date is sampled from one read.
    """
    import rasterio
    import xarray as xr

    out_dir = ANCILLARY / "rain"
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = chip_dates(keys)
    by_date: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        if not (out_dir / f"{key}.tif").exists():
            by_date[dates[key]].append(key)

    collection = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(
                "https://planetarycomputer.microsoft.com/api/stac/v1/collections/gpm-imerg-hhr",
                headers={"User-Agent": "SAT-AI/0.5"},
            ),
            timeout=90,
        ).read()
    )
    asset = collection["assets"]["zarr-abfs"]
    token = json.loads(
        urllib.request.urlopen(
            "https://planetarycomputer.microsoft.com/api/sas/v1/token/gpm-imerg-hhr", timeout=60
        ).read()
    )["token"]
    options = dict(asset.get("xarray:storage_options") or {})
    options["credential"] = token
    # decode_times=False on purpose: the store labels its axis "julian", which
    # would shift every date by about 13 days if decoded. The axis is minutes
    # since 2000-06-01 at a 30-minute step, verified against the first and last
    # values, so indices are computed directly.
    store = xr.open_zarr(
        asset["href"], storage_options=options, decode_times=False, consolidated=True
    )
    precipitation = store["precipitationCal"]
    origin = np.datetime64("2000-06-01T00:00")

    print(f"  rain: {len(by_date)} acquisition dates, {sum(map(len, by_date.values()))} chips")
    for n, (day, day_keys) in enumerate(sorted(by_date.items()), start=1):
        end = np.datetime64(day) + np.timedelta64(1, "D")
        stop = int((end - origin) / np.timedelta64(30, "m"))
        start = stop - 144
        started = time.time()
        profiles = {key: chip_profile(key) for key in day_keys}
        lons = [(p["bounds"][0] + p["bounds"][2]) / 2 for p in profiles.values()]
        lats = [(p["bounds"][1] + p["bounds"][3]) / 2 for p in profiles.values()]
        window = precipitation.isel(time=slice(start, stop)).sel(
            lon=slice(min(lons) - 0.2, max(lons) + 0.2), lat=slice(min(lats) - 0.2, max(lats) + 0.2)
        )
        # mm/hr at 30-minute steps: sum and halve for mm.
        total = (window.sum("time") * 0.5).load()
        for key, profile in profiles.items():
            lon = (profile["bounds"][0] + profile["bounds"][2]) / 2
            lat = (profile["bounds"][1] + profile["bounds"][3]) / 2
            value = float(total.interp(lon=lon, lat=lat).values)
            with rasterio.open(
                out_dir / f"{key}.tif",
                "w",
                driver="GTiff",
                height=profile["height"],
                width=profile["width"],
                count=1,
                dtype="float32",
                crs=profile["crs"],
                transform=profile["transform"],
                compress="LZW",
            ) as dst:
                dst.write(
                    np.full((profile["height"], profile["width"]), value, dtype=np.float32), 1
                )
                dst.set_band_description(1, "rain_72h_mm")
        print(
            f"  rain {n:>2}/{len(by_date)}  {day}  {len(day_keys):>3} chips  "
            f"{time.time() - started:,.0f} s",
            flush=True,
        )

    return {
        "product": "GPM IMERG V06 final, half-hourly (precipitationCal)",
        "collection": "gpm-imerg-hhr",
        "bands": ["rain_72h_mm"],
        "window": "72 hours ending at 00:00 UTC the day after acquisition",
        "note": "0.1 degree product sampled at the chip centre; constant across each chip",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", action="store_true")
    parser.add_argument("--rain", action="store_true")
    args = parser.parse_args(argv)
    if not (args.dem or args.rain):
        parser.error("choose --dem, --rain or both")

    reader = build_reader(CHIPS)
    keys = sorted(chip.key for chip in reader.available(reader.discover()))
    manifest_path = ANCILLARY / "manifest.json"
    ANCILLARY.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )

    if args.dem:
        manifest["dem"] = build_dem(keys)
        print(
            f"\n  dem: {manifest['dem']['written']} written, "
            f"{manifest['dem']['already_present']} present"
        )
    if args.rain:
        manifest["rain"] = build_rain(keys)

    manifest["chips"] = len(keys)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  manifest: {relative_to_repo(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
