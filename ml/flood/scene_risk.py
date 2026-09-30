"""Flood risk over a real scene: observed flood extent x real population.

    python -m ml.flood.scene_risk --metadata \
        data/processed/flood_scenes/nepal_koshi_terai_20240927T001212_otsu_metadata.json

The risk map was the last thing blocked on "no exposure raster". WorldPop now
supplies exposure, and ``ml/flood/otsu_scene.py`` supplies a flood extent from a
method that is valid on this scene. This joins them through the project's risk
engine, ``R = H^a * E^b * V^g``, on WorldPop's own 100 m grid.

Why the population grid and not the radar grid
-----------------------------------------------
Population is a count per cell. Pushing it onto a 10 m grid would invent where
inside each 100 m cell people live. Averaging the 10 m flood mask up to 100 m
instead gives the flooded *fraction* of each population cell, which is a
quantity the data actually supports. The hazard term is that fraction.

What each term is
-----------------
* **H** -- fraction of the 100 m cell observed as flood water (0..1). Unobserved
  cells are NaN and are banded as NODATA, never as safe.
* **E** -- WorldPop 2020 population, divided by its 99th percentile over
  populated cells and clipped to 1.
* **V** -- not available. No vulnerability layer exists for this area, so it is
  excluded rather than imputed; the engine records that caveat itself.

The headline number is not the index. "Risk 0.43" is not something anyone can
act on; the population living in flooded cells is. Both are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from ml.hazards.overlay import continuous_overlay  # noqa: E402
from satai.paths import relative_to_repo  # noqa: E402
from satai.providers.worldpop import fetch_population, product_for  # noqa: E402
from satai.risk.engine import RiskEngine  # noqa: E402

OVERLAY_DIR = REPO_ROOT / "frontend" / "public" / "layers"


def main(argv: list[str] | None = None) -> int:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.windows import from_bounds

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="an otsu_scene metadata file")
    args = parser.parse_args(argv)

    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    flood_path = REPO_ROOT / meta["artifacts"]["flood_mask"]
    bbox = tuple(meta["bbox"])

    # --- exposure on its own grid --------------------------------------------
    product = product_for(meta["country"])
    population_path = fetch_population(product, REPO_ROOT / "data/raw/worldpop" / product.filename)
    with rasterio.open(population_path) as src:
        window = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
        population = src.read(1, window=window).astype(np.float64)
        grid = {
            "crs": src.crs,
            "transform": src.window_transform(window),
            "height": population.shape[0],
            "width": population.shape[1],
        }
        nodata = src.nodata
    if nodata is not None:
        population = np.where(population == nodata, 0.0, population)
    population = np.where(np.isfinite(population) & (population > 0), population, 0.0)

    # --- hazard: flooded and observed fractions per population cell ----------
    with rasterio.open(flood_path) as src:
        flood_10m = src.read(1).astype(np.float32)
        flood_profile = {"crs": src.crs, "transform": src.transform}
    # Coverage is its own raster: the flood mask is 0/1 with no nodata, so an
    # unobserved pixel and a dry one look identical in it.
    observed_path = REPO_ROOT / meta["artifacts"]["observed"]
    with rasterio.open(observed_path) as src:
        observed_10m = src.read(1).astype(np.float32)

    def to_population_grid(values: np.ndarray) -> np.ndarray:
        out = np.full((grid["height"], grid["width"]), np.nan, dtype=np.float32)
        reproject(
            source=values,
            destination=out,
            src_transform=flood_profile["transform"],
            src_crs=flood_profile["crs"],
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        return out

    flooded_fraction = to_population_grid(flood_10m)
    coverage = to_population_grid(observed_10m)
    hazard = np.where(np.isfinite(coverage) & (coverage > 0.5), flooded_fraction, np.nan)

    # --- risk -----------------------------------------------------------------
    populated = population[population > 0]
    denominator = float(np.percentile(populated, 99)) if populated.size else 1.0
    exposure = np.clip(population / max(denominator, 1e-9), 0.0, 1.0)
    engine = RiskEngine.from_configs(
        REPO_ROOT / "configs" / "risk.yaml", REPO_ROOT / "configs" / "couplings.yaml"
    )
    result = engine.compute({"flood": hazard}, exposure=exposure, normalise=False)
    risk = result.risk_vector["flood"]

    observed = np.isfinite(hazard)
    people_observed = float(population[observed].sum())
    people_in_flood = float((population * np.nan_to_num(hazard, nan=0.0))[observed].sum())
    cells_any_flood = observed & (np.nan_to_num(hazard) > 0)
    people_in_touched_cells = float(population[cells_any_flood].sum())

    # --- artifacts -------------------------------------------------------------
    out_dir = args.metadata.parent
    stem = args.metadata.name.replace("_metadata.json", "")
    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": 1,
        "dtype": "float32",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "compress": "LZW",
        "nodata": np.nan,
    }
    for name, array in (("flood_fraction", hazard), ("exposure", exposure), ("risk", risk)):
        with rasterio.open(out_dir / f"{stem}_{name}.tif", "w", **profile) as dst:
            dst.write(np.asarray(array, dtype=np.float32), 1)

    overlays = {
        "hazard": continuous_overlay(
            hazard,
            grid,
            OVERLAY_DIR / f"{stem}_hazard.png",
            vmin=0.0,
            vmax=1.0,
            colour=(37, 99, 235),
            floor=0.0,
        ),
        "exposure": continuous_overlay(
            exposure,
            grid,
            OVERLAY_DIR / f"{stem}_exposure.png",
            vmin=0.0,
            vmax=1.0,
            colour=(124, 58, 237),
            floor=0.0,
        ),
        "risk": continuous_overlay(
            np.asarray(risk, dtype=np.float32),
            grid,
            OVERLAY_DIR / f"{stem}_risk.png",
            vmin=0.0,
            vmax=float(np.nanpercentile(risk, 99.5)) or 1.0,
            colour=(220, 38, 38),
            floor=engine.config.floor,
        ),
    }

    payload: dict[str, Any] = {
        "analysis": "flood_risk_scene",
        "hazard": "flood",
        "source_kind": "index",
        "aoi": meta["aoi"],
        "aoi_name": meta["aoi_name"],
        "country": meta["country"],
        "run_at": datetime.now(UTC).isoformat(),
        "formulation": f"R = H^{engine.config.alpha} * E^{engine.config.beta} (V excluded)",
        "config_hash": result.config_hash,
        "grid": {"crs": str(grid["crs"]), "resolution": "WorldPop 100 m (3 arc-seconds)"},
        "layers": {
            "hazard": {
                "what": "fraction of each 100 m cell observed as flood water",
                "source_kind": "model",
                "from": relative_to_repo(args.metadata),
                "method": meta["method"]["name"],
                "scene_id": meta["observation"]["scene_id"],
                "acquired_at": meta["observation"]["acquired_at"],
                "overlay": {
                    "url": f"/layers/{overlays['hazard'].path.name}",
                    **overlays["hazard"].describe(),
                },
            },
            "exposure": {
                "what": "WorldPop 2020 population, normalised by its 99th percentile",
                "source_kind": "model",
                **product.provenance(population_path),
                "overlay": {
                    "url": f"/layers/{overlays['exposure'].path.name}",
                    **overlays["exposure"].describe(),
                },
            },
            "vulnerability": {
                "available": False,
                "reason": "No vulnerability layer exists for this area; excluded, not imputed.",
            },
            "risk": {
                "what": "flood risk index, relative within this study area",
                "source_kind": "index",
                "overlay": {
                    "url": f"/layers/{overlays['risk'].path.name}",
                    **overlays["risk"].describe(),
                },
            },
        },
        "result": {
            "people_in_observed_cells": round(people_observed),
            "people_in_flood_water_estimate": round(people_in_flood),
            "people_in_cells_touched_by_flood": round(people_in_touched_cells),
            "risk_mean": round(float(np.nanmean(risk)), 6),
            "risk_p99": round(float(np.nanpercentile(risk, 99)), 6),
            "band_fractions": result.band_fractions("flood"),
        },
        "caveats": [
            *result.caveats,
            "People in flood water = population x flooded fraction per 100 m "
            "cell. It assumes people are spread evenly within a cell, which "
            "they are not; it is an exposure estimate, not a casualty or "
            "displacement figure.",
            "The hazard is the Otsu baseline's extent (IoU 0.3754 on held-out "
            "India), one acquisition, no HAND mask. Its errors carry into every "
            "number here.",
            "Risk bands are percentiles within this area, not absolute levels.",
            "Not an official warning or impact assessment.",
        ],
    }
    (out_dir / f"{stem}_risk_metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print(f"  grid {grid['width']}x{grid['height']} at WorldPop 100 m")
    print(f"  people in observed cells       {people_observed:>12,.0f}")
    print(f"  people in flood water (est.)   {people_in_flood:>12,.0f}")
    print(f"  people in cells touched        {people_in_touched_cells:>12,.0f}")
    print(
        f"  risk mean {payload['result']['risk_mean']:.5f}  p99 {payload['result']['risk_p99']:.5f}"
    )
    print(f"\nwritten to {relative_to_repo(out_dir / f'{stem}_risk_metadata.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
