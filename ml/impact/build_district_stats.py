"""Build and export district-level flood impact statistics for Bihar.

Computes district-level inundation, agricultural land exposure, historical hazard,
and the SAT-AI Impact Index for validated satellite flood events in Bihar.
Outputs results to data/processed/district_impact/.
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
    BIHAR_EVENT_CATALOGUE,
    get_district_impact_stats,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satai.impact.district")

OUTPUT_DIR = Path("data/processed/district_impact")


def build_all_district_stats():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_events = {}

    for event_date in BIHAR_EVENT_CATALOGUE:
        logger.info("Computing district impact for event date %s...", event_date)
        stats = get_district_impact_stats(event_date=event_date, sort_by="flooded_area_km2")
        all_events[event_date] = stats

        event_file = OUTPUT_DIR / f"bihar_district_impact_{event_date}.json"
        event_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        logger.info("Saved %s", event_file)

    master_file = OUTPUT_DIR / "bihar_district_impact_latest.json"
    master_file.write_text(json.dumps(all_events["2022-10-15"], indent=2), encoding="utf-8")
    logger.info("Exported master district impact to %s", master_file)
    return all_events


def main():
    build_all_district_stats()


if __name__ == "__main__":
    main()
