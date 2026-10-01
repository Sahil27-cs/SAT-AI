"""Assemble and export unified Phase 11I current event schema object.

Usage:
    python scripts/build_recent_event.py [--scene-id SCENE_ID]
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from satai.geo.recent_scene_engine import build_unified_recent_event_object


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified recent event schema object.")
    parser.add_argument("--scene-id", type=str, default=None, help="Sentinel-1 scene ID")
    args = parser.parse_args()

    event = build_unified_recent_event_object(args.scene_id)
    print("=== UNIFIED RECENT EVENT OBJECT ASSEMBLED ===")
    print(json.dumps(event, indent=2))
    print(f"\nArtifact saved to data/processed/recent_events/{event['event_id']}.json")


if __name__ == "__main__":
    main()
