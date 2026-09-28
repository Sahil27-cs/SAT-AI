"""Historical cyclone wind exposure and risk, from observed best tracks.

    python -m ml.hazards.run_cyclone --aoi odisha_mahanadi --storm FANI --season 2019

`satai.hazards.extreme_weather` has always been able to compute a parametric
wind field; what it could not do was obtain a track, and its first line refuses
to invent one. This runner supplies observed tracks from IBTrACS, builds the
wind field over a study area, and combines it with WorldPop population through
the project's risk engine.

**This is a hindcast, and the word matters.** A best track is a post-season
reanalysis. Nothing here forecasts a cyclone, and nothing here should be
described as a cyclone track prediction — the artifact says "historical
analysis" in its own payload so that the label travels with the numbers.

The exposure term is real population, not a placeholder. That is the whole
reason this can produce a risk field at all: `R = H^a * E^b * V^g` with a
constant E is a rescaled hazard map wearing a risk map's clothes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
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

    from satai.paths import relative_to_repo

import numpy as np

from satai.geo.aoi import load_aoi_registry
from satai.hazards.extreme_weather import track_exposure, wind_risk_index
from satai.providers.ibtracs import fetch_basin, load_tracks
from satai.providers.worldpop import fetch_population, product_for
from satai.risk.engine import RiskEngine

DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "cyclone"

#: Grid spacing for the wind field, in degrees. ~1 km at this latitude. The
#: parametric vortex has no structure finer than its own radius of maximum
#: wind, so a 10 m grid would be a hundred times the storage for no additional
#: information.
GRID_DEG = 0.01


def _grid(bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, Any]:
    from rasterio.transform import from_origin

    min_lon, min_lat, max_lon, max_lat = bbox
    lons = np.arange(min_lon, max_lon, GRID_DEG)
    lats = np.arange(max_lat, min_lat, -GRID_DEG)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    transform = from_origin(min_lon, max_lat, GRID_DEG, GRID_DEG)
    return lon_grid, lat_grid, transform


def _resample_population(
    population_path: Path, bbox: tuple[float, float, float, float], shape: tuple[int, int]
) -> tuple[np.ndarray, float]:
    """Population on the wind grid, plus the total it represents.

    Resampled with **sum**, not nearest or bilinear: population is a count per
    cell, and any interpolating resampler silently changes how many people the
    raster claims exist.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    with rasterio.open(population_path) as src:
        window = rasterio.windows.from_bounds(*bbox, transform=src.transform)
        source = src.read(1, window=window, boundless=True, fill_value=0.0).astype(np.float64)
        source_transform = src.window_transform(window)
        source_crs = src.crs
        nodata = src.nodata

    if nodata is not None:
        source = np.where(source == nodata, 0.0, source)
    source = np.where(np.isfinite(source) & (source >= 0), source, 0.0)

    destination = np.zeros(shape, dtype=np.float64)
    reproject(
        source=source,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=from_origin(bbox[0], bbox[3], GRID_DEG, GRID_DEG),
        dst_crs="EPSG:4326",
        resampling=Resampling.sum,
    )
    return destination, float(source.sum())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", required=True)
    parser.add_argument("--storm", required=True, help="IBTrACS storm NAME, e.g. FANI")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--basin", default="NI", help="NI = North Indian")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    registry = load_aoi_registry(REPO_ROOT / "configs" / "aoi.yaml")
    aoi = next((a for a in registry.aois if a.id == args.aoi), None)
    if aoi is None:
        raise SystemExit(f"{args.aoi!r} is not a configured study area")

    # --- observed track ------------------------------------------------------
    archive = fetch_basin(args.basin, REPO_ROOT / "data/raw/ibtracs" / f"ibtracs.{args.basin}.csv")
    tracks = load_tracks(archive, name=args.storm, season=args.season)
    if not tracks:
        print(
            f"BLOCKED: IBTrACS {args.basin} has no storm named {args.storm!r} in "
            f"{args.season}. Nothing was computed."
        )
        return 2
    track = max(tracks, key=lambda t: len(t.points))
    print(f"storm   {track.name} {track.season} ({track.sid})")
    print(f"        {len(track.points)} observed fixes, peak wind {track.peak_wind_ms:.1f} m/s")

    # --- wind field over the study area --------------------------------------
    lon_grid, lat_grid, transform = _grid(aoi.bbox)
    peak_wind = track_exposure(track.points, lon_grid, lat_grid)
    hazard = wind_risk_index(peak_wind)
    print(f"        grid {lon_grid.shape[0]}x{lon_grid.shape[1]} at {GRID_DEG} deg")
    print(f"        peak wind over the AOI {peak_wind.max():.1f} m/s")

    # --- real exposure -------------------------------------------------------
    # 1 km, matching GRID_DEG: the wind field has no structure finer than the
    # storm's radius of maximum wind, so the 100 m product would be half a
    # gigabyte of resolution thrown away on the next line.
    product = product_for(aoi.country, resolution="1km")
    population_path = fetch_population(product, REPO_ROOT / "data/raw/worldpop" / product.filename)
    population, total_in_window = _resample_population(population_path, aoi.bbox, lon_grid.shape)
    # Normalised for the engine, which wants [0, 1]; the count is kept for the
    # report, because "0.43 exposure" is not a number anyone can act on.
    denominator = float(np.percentile(population[population > 0], 99)) if population.any() else 1.0
    exposure = np.clip(population / max(denominator, 1e-9), 0.0, 1.0)
    print(f"        population in AOI {total_in_window:,.0f} (WorldPop {product.year})")

    # --- risk ----------------------------------------------------------------
    engine = RiskEngine.from_configs(
        REPO_ROOT / "configs" / "risk.yaml", REPO_ROOT / "configs" / "couplings.yaml"
    )
    result = engine.compute({"cyclone": hazard}, exposure=exposure, normalise=False)
    risk = result.risk_vector["cyclone"]

    people_above_damage = float(population[peak_wind >= 25.0].sum())

    # --- artifacts -----------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{aoi.id}_{track.name.lower()}_{track.season}"

    import rasterio

    profile = {
        "driver": "GTiff",
        "height": lon_grid.shape[0],
        "width": lon_grid.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "LZW",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    layers = {"peak_wind_ms": peak_wind, "wind_hazard_index": hazard, "cyclone_risk": risk}
    for layer, array in layers.items():
        with rasterio.open(args.out / f"{stem}_{layer}.tif", "w", **profile) as dst:
            dst.write(array.astype("float32"), 1)

    payload: dict[str, Any] = {
        "analysis": "historical_cyclone_wind_exposure",
        "not_a_prediction": (
            "Hindcast from an observed best track. SAT-AI does not forecast "
            "cyclone tracks and this is not a cyclone track prediction."
        ),
        "aoi": aoi.id,
        "aoi_name": aoi.name,
        "country": aoi.country,
        "bbox": list(aoi.bbox),
        "hazard": "cyclone",
        "run_at": datetime.now(UTC).isoformat(),
        "grid": {"shape": list(lon_grid.shape), "resolution_deg": GRID_DEG, "crs": "EPSG:4326"},
        "track": track.provenance(),
        # Small enough to serve: 62 fixes is a few kilobytes, and a track is the
        # one part of a cyclone analysis a reader can check against any public
        # record of the storm.
        "track_geojson": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[p.lon, p.lat] for p in track.points],
                    },
                    "properties": {
                        "storm": track.name,
                        "season": track.season,
                        "sid": track.sid,
                        "source_kind": "observation",
                        "dataset": "IBTrACS v04r01",
                    },
                },
                *[
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [point.lon, point.lat]},
                        "properties": {
                            "observed_at": when.isoformat(),
                            "max_wind_ms": round(point.max_wind_ms, 1),
                            "radius_max_wind_km": round(point.radius_max_wind_km, 1),
                            "source_kind": "observation",
                        },
                    }
                    for point, when in zip(track.points, track.times, strict=True)
                ],
            ],
        },
        "exposure": product.provenance(population_path),
        "wind": {
            "source_kind": "derived",
            "method": "modified Rankine vortex, peak over all track fixes",
            "peak_wind_ms_over_aoi": round(float(peak_wind.max()), 2),
            "damage_threshold_ms": 25.0,
            "saturation_ms": 70.0,
        },
        "risk": {
            "source_kind": "index",
            "formulation": f"R = H^{engine.config.alpha} * E^{engine.config.beta}",
            "config_hash": result.config_hash,
            "mean": round(float(np.nanmean(risk)), 6),
            "p90": round(float(np.nanpercentile(risk, 90)), 6),
            "max": round(float(np.nanmax(risk)), 6),
            "band_fractions": result.band_fractions("cyclone"),
        },
        "population": {
            "total_in_aoi": round(total_in_window),
            "in_cells_above_damage_threshold": round(people_above_damage),
            "normalisation": "population / 99th percentile of populated cells, clipped to 1",
        },
        "artifacts": {
            layer: relative_to_repo(args.out / f"{stem}_{layer}.tif") for layer in layers
        },
        "caveats": [
            *result.caveats,
            *track.provenance()["caveats"],
            "The wind field is parametric. It has no storm-motion asymmetry, no "
            "terrain roughness and no boundary-layer correction, so it "
            "overstates wind over rough terrain and understates the dangerous "
            "semicircle.",
            "Population exposed is counted where the modelled wind exceeds the "
            "damage threshold. It is not a casualty or damage estimate.",
            "Not an official warning. Cyclone warnings for India come from the "
            "India Meteorological Department.",
        ],
    }
    (args.out / f"{stem}_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"        risk mean {payload['risk']['mean']:.4f}  p90 {payload['risk']['p90']:.4f}")
    print(f"        population in cells above 25 m/s: {people_above_damage:,.0f}")
    print(f"\nwritten to {relative_to_repo(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
