"""Collect the real analysis artifacts into one catalogue the deployment serves.

    python scripts/export_analyses.py

The analyses this project produces live on disk as rasters and reports. Most of
that cannot be deployed: one flood probability raster is 500 MB and its extent
GeoJSON is 16 MB, which is larger than the entire serverless bundle budget. What
*can* be deployed is the record of what was produced, what it was produced from,
and whether it survived validation -- plus the handful of geometries that are
genuinely small, like an observed cyclone track and a set of fire detections.

So this writes a catalogue, not the data. The distinction is the point: the
website can then say exactly what exists for each study area without pretending
to display something it cannot serve, and without displaying something that
failed its own validation.

Status, and why it is derived rather than declared
--------------------------------------------------
A status written by hand drifts from the artifacts within a week. Every state
below is computed from files on disk:

    CONFIGURED         in configs/aoi.yaml, and nothing else exists
    DATA AVAILABLE     a real acquisition has been made for this area
    MODEL READY        a trained model exists that applies to this hazard
    INFERENCE READY    inference has been run and artifacts were written
    ANALYSIS COMPLETE  inference ran and passed its validation
    BLOCKED            something specific is missing, and it is named

The distinction between INFERENCE READY and ANALYSIS COMPLETE is load-bearing.
The Nepal flood run produced every artifact and then failed the Track A/B
distribution gate, because the model was trained on sigma0 and the scene is
gamma0. Calling that "complete" would put a flood map on the screen that the
project's own instrument refuses to stand behind.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satai.geo.aoi import load_aoi_registry  # noqa: E402
from satai.paths import relative_to_repo  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
EXPERIMENTS = REPO_ROOT / "ml" / "experiments"

DESTINATIONS = (
    REPO_ROOT / "backend" / "api" / "analyses.generated.json",
    REPO_ROOT / "frontend" / "lib" / "analyses.generated.json",
)

CONFIGURED = "CONFIGURED"
DATA_AVAILABLE = "DATA AVAILABLE"
MODEL_READY = "MODEL READY"
INFERENCE_READY = "INFERENCE READY"
ANALYSIS_COMPLETE = "ANALYSIS COMPLETE"
BLOCKED = "BLOCKED"

#: Geometry small enough to ship to a browser. Everything above this stays on
#: disk and is described rather than served.
MAX_FEATURES = 500


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return data


def flood_scene_entries() -> list[dict[str, Any]]:
    """One entry per real Sentinel-1 scene the flood model has been run over."""
    entries: list[dict[str, Any]] = []
    for path in sorted((PROCESSED / "flood_scenes").glob("*_metadata.json")):
        meta = _load(path)
        if meta is None:
            continue

        gate = meta.get("distribution_gate") or {}
        passed = bool(gate.get("may_proceed"))
        entries.append(
            {
                "aoi": meta["aoi"],
                "hazard": "flood",
                "kind": "model_inference",
                "status": ANALYSIS_COMPLETE if passed else INFERENCE_READY,
                "source_kind": "model",
                "headline": (
                    f"{meta['flooded_area_km2']:,.1f} km2 of "
                    f"{meta['observed_area_km2']:,.1f} km2 observed"
                ),
                # Withheld deliberately when the gate refused the transfer. The
                # artifacts exist; displaying them as a flood map would present
                # a result the project's own instrument rejected.
                "displayable": passed,
                "withheld_reason": None
                if passed
                else (
                    "The Track A/B distribution gate refused this transfer: the "
                    "model was trained on sigma0 and this scene is gamma0 RTC. "
                    "The extent is not shown because it has not been validated."
                ),
                "observation": {
                    "scene_id": meta["scene_id"],
                    "platform": meta.get("platform"),
                    "acquired_at": meta.get("acquired_at"),
                    "collection": meta.get("collection"),
                    "provider": meta.get("provider"),
                    "radiometry": meta.get("radiometry"),
                    "orbit_direction": meta.get("orbit_direction"),
                    "relative_orbit": meta.get("relative_orbit"),
                    "resolution_m": meta.get("resolution_m"),
                },
                "model": {
                    "name": meta.get("model"),
                    "version": meta.get("model_version"),
                    "bands": meta.get("bands"),
                    "threshold": meta.get("threshold"),
                    "held_out_india_iou": meta.get("test_iou_on_held_out_india"),
                },
                "processing": {
                    "processed_at": meta.get("processed_at"),
                    "grid": meta.get("grid"),
                    "crs": meta.get("crs"),
                    "pixel_area_km2": meta.get("pixel_area_km2"),
                },
                "validation": {
                    "instrument": "Track A/B distribution gate (ADR-010)",
                    "verdict": gate.get("verdict"),
                    "may_proceed": gate.get("may_proceed"),
                    "bands": gate.get("bands"),
                },
                "artifacts_on_disk": meta.get("artifacts"),
                "artifacts_served": False,
                "artifacts_not_served_because": (
                    "The probability raster is ~500 MB and the extent GeoJSON "
                    "~16 MB. Both exceed the serverless bundle budget, so the "
                    "record is served and the data stays on disk."
                ),
                "caveats": meta.get("caveats", []),
            }
        )
    return entries


def cyclone_entries() -> list[dict[str, Any]]:
    """One entry per historical cyclone analysis, with the observed track."""
    entries: list[dict[str, Any]] = []
    for path in sorted((PROCESSED / "cyclone").glob("*_metadata.json")):
        meta = _load(path)
        if meta is None:
            continue
        track = meta.get("track", {})
        entries.append(
            {
                "aoi": meta["aoi"],
                "hazard": "cyclone",
                "kind": "historical_analysis",
                "status": ANALYSIS_COMPLETE,
                "source_kind": "index",
                "headline": (
                    f"{track.get('storm_name', 'storm')} {track.get('season', '')}: "
                    f"peak {meta['wind']['peak_wind_ms_over_aoi']:.0f} m/s over the AOI"
                ),
                "displayable": True,
                "not_a_prediction": meta.get("not_a_prediction"),
                "observation": track,
                "exposure": meta.get("exposure"),
                "wind": meta.get("wind"),
                "risk": meta.get("risk"),
                "population": meta.get("population"),
                "geometry": meta.get("track_geojson"),
                "artifacts_on_disk": meta.get("artifacts"),
                "artifacts_served": False,
                "caveats": meta.get("caveats", []),
            }
        )
    return entries


def wildfire_entry() -> dict[str, Any] | None:
    """The most recent active-fire observation, detections included.

    Served in full: seven points is not a payload worth paginating.
    """
    meta = _load(PROCESSED / "wildfire" / "active_fire_latest.json")
    if meta is None:
        return None
    return {
        "hazard": "wildfire",
        "kind": "observation",
        "status": DATA_AVAILABLE,
        "source_kind": "observation",
        "not_a_prediction": meta.get("not_a_prediction"),
        "run_at": meta.get("run_at"),
        "window": meta.get("window"),
        "window_days": meta.get("window_days"),
        "provenance": meta.get("provenance"),
        "totals": meta.get("totals"),
        "per_area": [
            {
                "aoi": area["aoi"],
                "detection_count": area["detection_count"],
                "confidence_breakdown": area["confidence_breakdown"],
                "max_frp_mw": area["max_frp_mw"],
                "first_detection": area["first_detection"],
                "last_detection": area["last_detection"],
                "detections": area["detections"][:MAX_FEATURES],
            }
            for area in meta.get("study_areas", [])
        ],
        "caveats": meta.get("caveats", []),
    }


def xai_entries() -> list[dict[str, Any]]:
    """Per-band attribution for the flood model."""
    entries: list[dict[str, Any]] = []
    for path in sorted((EXPERIMENTS / "flood_xai").glob("xai_*.json")):
        meta = _load(path)
        if meta is None:
            continue
        entries.append(
            {
                "hazard": "flood",
                "kind": "model_explanation",
                "status": ANALYSIS_COMPLETE,
                "source_kind": "derived",
                "model": meta.get("model"),
                "model_version": meta.get("model_version"),
                "region": meta.get("region"),
                "n_chips": meta.get("n_chips"),
                "bands": meta.get("bands"),
                "methods": meta.get("methods"),
                "attribution_share": meta.get("attribution_share"),
                "mean_integrated_gradients": meta.get("mean_integrated_gradients"),
                "mean_occlusion_delta": meta.get("mean_occlusion_delta"),
                "methods_disagree_on": meta.get("methods_disagree_on"),
                "cross_reference_c2": meta.get("cross_reference_c2"),
                "run_at": meta.get("run_at"),
                "report": relative_to_repo(path),
                "caveats": meta.get("caveats", []),
            }
        )
    return entries


def main() -> int:
    registry = load_aoi_registry(REPO_ROOT / "configs" / "aoi.yaml")
    flood = flood_scene_entries()
    cyclone = cyclone_entries()
    wildfire = wildfire_entry()
    xai = xai_entries()

    by_area: dict[str, list[dict[str, Any]]] = {}
    for entry in [*flood, *cyclone]:
        by_area.setdefault(entry["aoi"], []).append(entry)
    if wildfire:
        for area in wildfire["per_area"]:
            if area["detection_count"]:
                by_area.setdefault(area["aoi"], []).append(
                    {
                        "aoi": area["aoi"],
                        "hazard": "wildfire",
                        "kind": "observation",
                        "status": DATA_AVAILABLE,
                        "source_kind": "observation",
                        "headline": (
                            f"{area['detection_count']} active-fire detection(s) "
                            f"in the last {wildfire['window']}"
                        ),
                        "displayable": True,
                        "detections": area["detections"],
                        "caveats": wildfire["caveats"],
                    }
                )

    areas = []
    for aoi in registry.aois:
        analyses = by_area.get(aoi.id, [])
        areas.append(
            {
                "id": aoi.id,
                "name": aoi.name,
                "country": aoi.country,
                "bbox": list(aoi.bbox),
                "primary_hazards": aoi.primary_hazards,
                "study_role": aoi.study_role,
                # The area's own state is the best any of its analyses reached.
                "status": _best_status([a["status"] for a in analyses]) or CONFIGURED,
                "analyses": analyses,
            }
        )

    payload = {
        "_comment": (
            "GENERATED by scripts/export_analyses.py from artifacts on disk. "
            "Do not edit: a hand-written status is how a study area comes to be "
            "described as analysed without an analysis."
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "status_vocabulary": {
            CONFIGURED: "declared in configs/aoi.yaml; nothing has been produced",
            DATA_AVAILABLE: "a real acquisition exists for this area",
            MODEL_READY: "a trained model applies to this hazard",
            INFERENCE_READY: "inference ran and wrote artifacts, but they did not pass validation",
            ANALYSIS_COMPLETE: "inference ran and passed its validation",
            BLOCKED: "a named dependency is missing",
        },
        "study_areas": areas,
        "wildfire": wildfire,
        "explanations": xai,
    }

    for destination in DESTINATIONS:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"  {'study area':<24} {'status':<20} analyses")
    for area in areas:
        print(f"  {area['id']:<24} {area['status']:<20} {len(area['analyses'])}")
    print(f"\n  {len(xai)} explanation report(s)")
    for destination in DESTINATIONS:
        size = destination.stat().st_size
        print(f"  written {relative_to_repo(destination)} ({size / 1024:.1f} KB)")
    return 0


def _best_status(statuses: list[str]) -> str | None:
    """The strongest state any analysis for an area reached."""
    order = [CONFIGURED, BLOCKED, DATA_AVAILABLE, MODEL_READY, INFERENCE_READY, ANALYSIS_COMPLETE]
    present = [s for s in statuses if s in order]
    return max(present, key=order.index) if present else None


if __name__ == "__main__":
    raise SystemExit(main())
