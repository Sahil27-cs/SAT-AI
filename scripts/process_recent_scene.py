"""Process and verify radiometric characteristics of a recent Sentinel-1 scene.

Usage:
    python scripts/process_recent_scene.py [--scene-id SCENE_ID]
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from satai.geo.sentinel_discovery import get_latest_satellite_scene, discover_recent_sentinel1_scenes
from satai.geo.recent_scene_engine import run_recent_distribution_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Process and verify recent Sentinel-1 scene characteristics.")
    parser.add_argument("--scene-id", type=str, default=None, help="Sentinel-1 scene ID to process")
    args = parser.parse_args()

    if args.scene_id:
        scene_id = args.scene_id
    else:
        latest = get_latest_satellite_scene(require_recent_only=True)
        if not latest:
            print("No recent satellite scene found in catalog.")
            return
        scene_id = latest["scene_id"]

    print(f"=== PROCESSING SENTINEL-1 SCENE: {scene_id} ===")
    print("1. Calibration: Radiometric sigma-nought (sigma0) backscatter cross section")
    print("2. Thermal Noise: Applied ESA instrument look-up tables")
    print("3. Terrain Correction: Range-Doppler Terrain Correction with SRTM 30m DEM")
    print("4. Target CRS: UTM Zone 45N (EPSG:32645), a projected CRS used for metric area calculations")
    print("5. Resolution: 10m grid (100 m² per pixel)")
    print("6. Dual-polarization extraction: VV, VH, and cross-ratio (VV_dB - VH_dB)")

    gate = run_recent_distribution_gate(scene_id)
    print("\n=== DISTRIBUTION GATE VERDICT ===")
    print(f"Status: {gate.get('status')}")
    print(f"VV Wasserstein Shift: {gate.get('vv_wasserstein_db')} dB (Fail threshold: 3.5 dB)")
    print(f"VH Wasserstein Shift: {gate.get('vh_wasserstein_db')} dB (Fail threshold: 3.5 dB)")
    print(f"Valid Pixel Fraction: {gate.get('valid_pixel_fraction')}")
    print("Details:")
    for d in gate.get("details", []):
        print(f"  • {d}")


if __name__ == "__main__":
    main()
