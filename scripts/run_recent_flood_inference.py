"""Run U-Net flood inference on a recent Sentinel-1 acquisition with distribution gating.

Usage:
    python scripts/run_recent_flood_inference.py [--scene-id SCENE_ID]
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from satai.geo.recent_scene_engine import (
    get_recent_flood_inference,
    get_recent_district_impact_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run U-Net flood inference on recent Sentinel-1 scene.")
    parser.add_argument("--scene-id", type=str, default=None, help="Sentinel-1 scene ID")
    args = parser.parse_args()

    inf = get_recent_flood_inference(args.scene_id)
    if not inf.get("available"):
        print("=== RECENT FLOOD INFERENCE WITHHELD OR UNAVAILABLE ===")
        print(f"Message: {inf.get('message') or inf.get('reason')}")
        if "distribution_gate" in inf:
            gate = inf["distribution_gate"]
            print(f"Gate Status: {gate.get('status')}")
            for d in gate.get("details", []):
                print(f"  - {d}")
        return

    print("=== MODEL-INFERRED FLOOD EXTENT ===")
    print(f"Scene ID: {inf['scene_id']}")
    print(f"Event Date: {inf['event_date']}")
    print(f"Satellite / Sensor: {inf['satellite']} / {inf['sensor']}")
    print(f"Model: {inf.get('distribution_gate', {}).get('status', 'PASSED')} distribution gate")
    print(f"Inundated Area: {inf['flooded_area_km2']} km² ({inf['flooded_percentage']}% of Bihar)")
    print(f"Observed Area: {inf['observed_area_km2']} km²")
    print(f"Cropland Exposed: {inf['cropland_exposed_km2']} km²")
    print(f"CRS Used: {inf['crs_used']}")
    print(f"Label: {inf['inference_label']}")

    dist = get_recent_district_impact_stats(inf["scene_id"])
    print(f"\n=== DISTRICT BREAKDOWN (Top 5 of {dist.get('affected_districts_count')} affected) ===")
    for d in dist.get("district_ranking", [])[:5]:
        print(f"• {d['district_name']}: {d['flooded_area_km2']} km² ({d['flood_percentage']}% of district), cropland: {d['cropland_affected_km2']} km²")


if __name__ == "__main__":
    main()
