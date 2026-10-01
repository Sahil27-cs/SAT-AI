"""Build and export exposure statistics for Bihar flood inundation.

Calculates:
- Cropland exposure (km2)
- Building count exposure
- Road network exposure (km)
- Historical hazard zonation overlap (%)
Outputs to data/processed/exposure/.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from satai.geo.bihar_impact import (
    BIHAR_DISTRICTS,
    BIHAR_EVENT_CATALOGUE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satai.impact.exposure")

OUTPUT_DIR = Path("data/processed/exposure")


def build_exposure_stats():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exposure_summary = {}

    for event_date, event_data in BIHAR_EVENT_CATALOGUE.items():
        total_cropland_exp = sum(d["cropland_affected_km2"] for d in event_data["districts"].values())
        total_bld_exp = sum(d["buildings_exposed"] for d in event_data["districts"].values())
        total_road_exp = sum(d["roads_exposed_km"] for d in event_data["districts"].values())

        # Calculate hazard overlap percent: fraction of inundation falling in High or Very High hazard zones
        inundation_in_high_hazard = 0.0
        for d_name, d_impact in event_data["districts"].items():
            profile = BIHAR_DISTRICTS.get(d_name, {})
            if profile.get("historical_hazard") in ("High", "Very High"):
                inundation_in_high_hazard += d_impact["flooded_area_km2"]

        hazard_overlap_pct = (
            round((inundation_in_high_hazard / event_data["total_flooded_area_km2"]) * 100.0, 2)
            if event_data["total_flooded_area_km2"] > 0
            else 0.0
        )

        exposure_summary[event_date] = {
            "event_date": event_date,
            "title": event_data["title"],
            "sensor": event_data["sensor"],
            "flooded_area_km2": event_data["total_flooded_area_km2"],
            "affected_cropland_km2": round(total_cropland_exp, 2),
            "affected_building_count": total_bld_exp,
            "affected_road_km": round(total_road_exp, 2),
            "hazard_overlap_percent": hazard_overlap_pct,
            "high_hazard_inundation_km2": round(inundation_in_high_hazard, 2),
            "number_of_affected_districts": len(event_data["districts"]),
            "calculation_method": "geodesic_projected_utm_zone_45n",
            "uncertainty_and_limitations": [
                "Building counts are settlement density proxies derived from Census and WorldPop grids.",
                "Road network exposure represents national and state highway alignments intersecting 10m flood footprints.",
                "Historical hazard overlap reflects 1998-2019 NRSC atlas zonation, not post-2019 flood defense works.",
            ],
        }

        output_file = OUTPUT_DIR / f"bihar_exposure_{event_date}.json"
        output_file.write_text(json.dumps(exposure_summary[event_date], indent=2), encoding="utf-8")
        logger.info("Saved exposure stats to %s", output_file)

    master_file = OUTPUT_DIR / "bihar_exposure_latest.json"
    master_file.write_text(json.dumps(exposure_summary["2022-10-15"], indent=2), encoding="utf-8")
    logger.info("Saved latest exposure master to %s", master_file)
    return exposure_summary


def main():
    build_exposure_stats()


if __name__ == "__main__":
    main()
