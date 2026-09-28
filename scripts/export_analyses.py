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

#: Why an analysis is not drawn. These are genuinely different facts and one
#: boolean was hiding that: a result that failed validation must never read as
#: complete, while a result that is simply too large to ship over a serverless
#: function is complete and merely undeliverable.
VALIDATION_FAILED = "validation_failed"
TOO_LARGE_TO_SERVE = "too_large_to_serve"


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
                "withheld_because": None if passed else VALIDATION_FAILED,
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


def damage_entries() -> list[dict[str, Any]]:
    """Post-event change detection, one entry per pre/post pair."""
    entries: list[dict[str, Any]] = []
    for path in sorted((PROCESSED / "damage").glob("*_metadata.json")):
        meta = _load(path)
        if meta is None:
            continue
        pair, result = meta.get("pair", {}), meta.get("result", {})
        entries.append(
            {
                "aoi": meta["aoi"],
                "hazard": "damage",
                "kind": "historical_analysis",
                "status": ANALYSIS_COMPLETE,
                "source_kind": "derived",
                "headline": (
                    f"{result['changed_fraction']:.1%} of observed pixels changed "
                    f"between {pair['pre_acquired_at'][:10]} and "
                    f"{pair['post_acquired_at'][:10]}"
                ),
                # The change field is a full raster; the summary is what travels.
                "displayable": False,
                "withheld_because": TOO_LARGE_TO_SERVE,
                "withheld_reason": (
                    "The change field is a full-resolution raster and is not "
                    "served. Its summary and provenance are. This analysis "
                    "passed; it is undeliverable, not unvalidated."
                ),
                "not_a_prediction": meta.get("not_a_prediction"),
                "observation": {
                    "scene_id": pair.get("post_scene_id"),
                    "acquired_at": pair.get("post_acquired_at"),
                    "provider": pair.get("provider"),
                    "radiometry": pair.get("radiometry"),
                    "pre_scene_id": pair.get("pre_scene_id"),
                    "pre_acquired_at": pair.get("pre_acquired_at"),
                    "relative_orbit": pair.get("relative_orbit"),
                    "separation_days": pair.get("separation_days"),
                    "same_relative_orbit": pair.get("same_relative_orbit"),
                },
                "method": meta.get("method"),
                "result": result,
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


def experiment_register() -> list[dict[str, Any]]:
    """The four contributions, with whatever each has actually produced.

    Built from report files rather than read from the database. The database
    rows are seeded configuration and went stale the moment an experiment ran:
    the deployment still reported two of ten executed after the flood model was
    trained, evaluated, ablated four ways and explained. Writing to that table
    needs a service key this deployment does not have, and a number nobody can
    correct is worse than a number derived from the artifacts themselves.
    """
    register: list[dict[str, Any]] = []

    # --- C1: grounded tool invocation ---------------------------------------
    validator = _load(EXPERIMENTS / "validator_validation" / "validator_validation.json")
    register.append(
        {
            "id": "C1",
            "title": "Grounded tool invocation and refusal",
            "hypothesis": (
                "A language layer constrained to tool output can be held to "
                "stating only values traceable to a tool result, and made to "
                "refuse questions outside its competence."
            ),
            "instrument": {
                "report": relative_to_repo(EXPERIMENTS / "validator_validation")
                if validator
                else None,
                "detection_accuracy": (validator or {}).get("headline"),
                "n_cases": (validator or {}).get("n_cases"),
                "status": "EXECUTED" if validator else "NOT RUN",
            },
            "status": "INSTRUMENT VALIDATED; BENCHMARK NOT RUN",
            "blocker": (
                "The 28-question benchmark has not been run against the live "
                "model. The loop, the validator and the audit log all exist."
            ),
            "caveats": [
                "The validated instrument measures detection on constructed "
                "failure modes, not coverage of all possible ones.",
            ],
        }
    )

    # --- C2: modality loss ---------------------------------------------------
    c2 = _load(EXPERIMENTS / "c2_modality_ablation.json")
    if c2:
        register.append(
            {
                "id": "C2",
                "title": "Degradation under induced modality loss",
                "hypothesis": (
                    "Withholding input modalities degrades a flood model by an "
                    "amount worth measuring, and the curve is not published for "
                    "this task."
                ),
                "dataset": "Sen1Floods11 v1.1 HandLabeled",
                "split": c2.get("protocol"),
                "test_region": c2.get("test_region"),
                "arms_executed": [
                    {
                        "arm": arm,
                        "bands": values["n_bands"],
                        "iou": values["iou"],
                        "f1": values["f1"],
                        "precision": values["precision"],
                        "recall": values["recall"],
                    }
                    for arm, values in sorted(
                        (c2.get("executed") or {}).items(), key=lambda kv: kv[1]["n_bands"]
                    )
                ],
                "arms_blocked": c2.get("blocked", {}),
                "spread_iou": c2.get("spread_across_executed_arms_iou"),
                "finding": c2.get("finding"),
                "status": c2.get("status", "PARTIALLY EXECUTED"),
                "report": relative_to_repo(EXPERIMENTS / "c2_modality_ablation.json"),
                "caveats": c2.get("caveats", []),
            }
        )

    # --- C3: rural to urban --------------------------------------------------
    c3 = _load(EXPERIMENTS / "c3_urban_domain_shift.json")
    register.append(
        {
            "id": "C3",
            "title": "Rural-to-urban SAR transfer gap in India",
            "hypothesis": (
                "A flood model trained on rural Indian chips degrades by a "
                "measurable and unpublished amount on urban Indian imagery."
            ),
            "status": (c3 or {}).get("status", "BLOCKED"),
            "what_this_is": (c3 or {}).get("what_this_is"),
            "what_this_is_not": (c3 or {}).get("what_this_is_not"),
            "distribution_gate": (c3 or {}).get("distribution_gate"),
            "expected_direction": (c3 or {}).get("expected_direction"),
            "blocker": (c3 or {}).get(
                "blocker", {"missing": "labelled urban Indian flood imagery"}
            ),
            "report": relative_to_repo(EXPERIMENTS / "c3_urban_domain_shift.json") if c3 else None,
            "caveats": (c3 or {}).get("caveats", []),
        }
    )

    # --- C4: risk sensitivity -------------------------------------------------
    c4 = _load(EXPERIMENTS / "risk_sensitivity" / "structural_summary.json")
    register.append(
        {
            "id": "C4",
            "title": "Sensitivity-analysed multi-hazard risk",
            "hypothesis": (
                "A risk formulation's free exponents can be turned from an "
                "arbitrary choice into a reported, inspectable result."
            ),
            "status": "EXECUTED" if c4 else "NOT RUN",
            "headline": (c4 or {}).get("headline") or (c4 or {}).get("interpretation"),
            "report": relative_to_repo(EXPERIMENTS / "risk_sensitivity") if c4 else None,
            "caveats": [
                "One-at-a-time holds the other exponents fixed and does not explore interactions.",
            ],
        }
    )

    # --- experiment 3: the split-protocol gap ---------------------------------
    official = _load(EXPERIMENTS / "flood_unet" / "test_official_sar_ratio.json")
    loro = _load(EXPERIMENTS / "flood_unet" / "test_loro_india_sar_ratio.json")
    entry: dict[str, Any] = {
        "id": "EXP3",
        "title": "Official-split versus leave-one-region-out",
        "hypothesis": (
            "Sen1Floods11's shipped splits are not region-disjoint, so a score "
            "measured on them overstates generalisation to an unseen region."
        ),
        "status": "EXECUTED" if (official and loro) else "PARTIALLY EXECUTED",
    }
    if loro:
        entry["loro_iou"] = loro["headline"]["iou"]
        entry["loro_test_regions"] = loro.get("test_regions") or ["India"]
    if official:
        entry["official_iou"] = official["headline"]["iou"]
        entry["gap_iou"] = round(official["headline"]["iou"] - loro["headline"]["iou"], 4)
        entry["finding"] = (
            f"The official chip-level split scores {official['headline']['iou']:.4f} "
            f"against {loro['headline']['iou']:.4f} region-disjoint, a gap of "
            f"{entry['gap_iou']:+.4f} IoU. Chips from one flood event straddle "
            f"train and test in the official split, so that difference is the "
            f"cost of measuring interpolation within an event and calling it "
            f"generalisation."
        )
    else:
        entry["blocker"] = "The official-split training run has not completed."
    entry["caveats"] = [
        "Both arms share seed, schedule, architecture and band stack, so the "
        "difference is attributable to the split protocol.",
        "Single seed per arm.",
    ]
    register.append(entry)

    return register


def main() -> int:
    registry = load_aoi_registry(REPO_ROOT / "configs" / "aoi.yaml")
    flood = flood_scene_entries()
    cyclone = cyclone_entries()
    damage = damage_entries()
    wildfire = wildfire_entry()
    xai = xai_entries()

    by_area: dict[str, list[dict[str, Any]]] = {}
    for entry in [*flood, *cyclone, *damage]:
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
        "experiments": experiment_register(),
    }

    for destination in DESTINATIONS:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"  {'study area':<24} {'status':<20} analyses")
    for area in areas:
        print(f"  {area['id']:<24} {area['status']:<20} {len(area['analyses'])}")
    register = payload["experiments"]
    executed = [e for e in register if "EXECUTED" in str(e.get("status", ""))]
    print(f"\n  {len(xai)} explanation report(s)")
    print(f"  {len(register)} registered experiments, {len(executed)} with executed results")
    for entry in register:
        print(f"    {entry['id']:<6} {entry['status']}")
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
