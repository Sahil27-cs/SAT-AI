"""Discover recent Sentinel-1 satellite scenes for Bihar.

Usage:
    python scripts/discover_recent_scenes.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--max-results N]
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from satai.geo.sentinel_discovery import discover_recent_sentinel1_scenes, BIHAR_BBOX


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover recent Sentinel-1 scenes from CDSE / ESA archive.")
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--max-results", type=int, default=10, help="Max results to return")
    args = parser.parse_args()

    print(f"Searching Copernicus Sentinel-1 catalog for Bihar AOI: {BIHAR_BBOX}")
    if args.start_date:
        print(f"Start Date Filter: {args.start_date}")
    if args.end_date:
        print(f"End Date Filter: {args.end_date}")

    scenes = discover_recent_sentinel1_scenes(
        start_date=args.start_date,
        end_date=args.end_date,
        max_results=args.max_results,
    )

    print(f"\nDiscovered {len(scenes)} matching scenes:")
    for i, sc in enumerate(scenes, 1):
        recent_flag = " [RECENT]" if sc.get("is_recent") else " [HISTORICAL BASELINE]"
        print(f"{i}. {sc['scene_id']}{recent_flag}")
        print(f"   Acquired: {sc.get('acquisition_datetime')}")
        print(f"   Mode: {sc.get('mode')}, Product: {sc.get('product_type')}, Orbit: {sc.get('orbit')} (Track {sc.get('relative_orbit')})")
        print(f"   Source: {sc.get('source')}")
        print(f"   Download URL: {sc.get('download_url')}")
        print()


if __name__ == "__main__":
    main()
