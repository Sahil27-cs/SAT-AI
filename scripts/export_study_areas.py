"""Export `configs/aoi.yaml` to JSON the frontend imports at build time.

Why a generated file rather than a fetch
----------------------------------------
Study areas are *configuration*, not results. They change when someone edits a
YAML file and opens a pull request, not on a satellite revisit, so fetching them
at runtime would add a network round trip and a failure mode to data that is
fixed at build time. The hazard results the dashboard shows still come from the
serving plane; only the region catalogue is baked in.

Why generated rather than hand-written
--------------------------------------
Because a second hand-maintained copy drifts. The frontend previously carried
its own idea of which regions existed, and the moment `aoi.yaml` gained three
AOIs the two disagreed with nothing to catch it.
`tests/test_study_area_export.py` fails the build if this file is stale, so the
YAML stays the single source of truth and the drift is a red test rather than a
region that silently never appears.

Run after editing configs/aoi.yaml:

    python scripts/export_study_areas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satai.geo.aoi import AOI, load_aoi_registry  # noqa: E402

OUTPUT = REPO_ROOT / "frontend" / "lib" / "study-areas.generated.json"


def _event_payload(aoi: AOI) -> list[dict[str, Any]]:
    """Events, with the verified flag preserved rather than flattened away.

    An unverified event still travels to the interface -- it is a real research
    lead and hiding it would misrepresent the register -- but it carries
    `verified: false`, and the UI renders it as CANDIDATE instead of offering
    historical analysis for it.
    """
    return [
        {
            "id": event.id,
            "name": event.name,
            "hazard": event.hazard,
            "occurredOn": event.occurred_on.isoformat(),
            "sensor": event.sensor,
            "acquisitionUtc": (
                event.acquisition_utc.isoformat() if event.acquisition_utc else None
            ),
            "referenceProducts": list(event.reference_products),
            "verified": event.verified,
            "verificationNote": " ".join(event.verification_note.split()),
            "displayStatus": event.display_status,
        }
        for event in aoi.events
    ]


def build_payload() -> dict[str, Any]:
    registry = load_aoi_registry()
    return {
        "_comment": (
            "GENERATED FILE -- do not edit. Source: configs/aoi.yaml. "
            "Regenerate with: python scripts/export_study_areas.py"
        ),
        "version": registry.version,
        "studyAreas": [
            {
                "id": aoi.id,
                "name": aoi.name,
                "country": aoi.country,
                "bbox": list(aoi.bbox),
                "centroid": list(aoi.centroid),
                "utmEpsg": aoi.utm_epsg,
                "areaKm2": round(aoi.approx_area_km2, 1),
                "tileCount": aoi.approx_tile_count(),
                "primaryHazards": list(aoi.primary_hazards),
                "studyRole": aoi.study_role.value,
                "status": aoi.status,
                "hasLabels": aoi.has_labels,
                "labelSources": list(aoi.label_sources),
                # Collapsed to a single line: these are multi-line YAML folded
                # scalars and the ragged internal whitespace renders badly.
                "selectionRationale": " ".join(aoi.selection_rationale.split()),
                "notes": " ".join(aoi.notes.split()) if aoi.notes else None,
                "hazardCoverage": aoi.hazard_coverage(),
                "events": _event_payload(aoi),
            }
            for aoi in registry.aois
        ],
    }


def main() -> int:
    payload = build_payload()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    areas = payload["studyAreas"]
    verified = sum(len([e for e in a["events"] if e["verified"]]) for a in areas)
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  {len(areas)} study areas, {verified} verified event(s)")
    for area in areas:
        events = ", ".join(
            f"{e['id']}{'' if e['verified'] else ' (unverified)'}" for e in area["events"]
        )
        print(f"  {area['id']:22} {area['studyRole']:20} {events or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
