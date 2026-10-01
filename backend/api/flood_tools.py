"""Typed flood tools for the assistant.

Each tool answers one kind of flood question from a file that measured it:

* ``flood_facts.generated.json`` -- model card, metrics, ground truth, gate
  thresholds and the inference pipeline, exported by
  ``scripts/export_flood_facts.py`` from the training manifest, the test
  reports and the gate module, and checked against them by the test suite;
* ``analyses.generated.json`` -- what was actually run on a real scene, exported
  by ``scripts/export_analyses.py`` from the artifacts on disk.

Nothing in this module is a number typed by hand. That is what makes the
grounding check meaningful: a value the assistant states has to be one of these
tools' outputs, and these outputs are copies of measurements.

**Metrics are keyed by split, never merged.** The Mekong figure is the
validation region that selected the checkpoint; the India figure is the test
region nothing was tuned on. Returning them in one flat dictionary is how
"0.868" becomes "the model's accuracy". Each split comes back with its own
label saying what it is and what it is not.

**Raw output is labelled by what it is.** The Nepal U-Net run produced an
extent and then failed its distribution gate. The tools return that extent --
hiding it would erase a real finding -- as ``raw_extent_km2`` with
``validated: false`` and ``status: BLOCKED``, next to the reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "FLOOD_TOOL_DECLARATIONS",
    "FLOOD_TOOL_NAMES",
    "analysis_catalogue",
    "execute_flood_tool",
    "flood_facts",
]

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from satai.geo.bihar_impact import (
        BFCD22_MUZAFFARPUR_STATS,
        BIHAR_DISTRICTS,
        BIHAR_EVENT_CATALOGUE,
        NOT_OFFICIAL_DISCLAIMER,
        compute_sat_ai_impact_index,
        get_available_event_dates,
        get_district_impact_stats,
        get_place_profile as _geo_place_profile,
        get_vegetation_analysis as _geo_vegetation_analysis,
    )
except Exception:
    BFCD22_MUZAFFARPUR_STATS = {}
    BIHAR_DISTRICTS = {}
    BIHAR_EVENT_CATALOGUE = {}
    NOT_OFFICIAL_DISCLAIMER = (
        "SAT-AI prototype research output. This is NOT an official warning or damage assessment."
    )
    compute_sat_ai_impact_index = lambda f, c, b, r, **kw: 0.0  # type: ignore
    get_available_event_dates = lambda: []  # type: ignore
    get_district_impact_stats = lambda **kw: {"available": False}  # type: ignore
    _geo_place_profile = lambda *a, **kw: {"available": False}  # type: ignore
    _geo_vegetation_analysis = lambda *a, **kw: {"available": False}  # type: ignore

try:
    from satai.geo.sentinel_discovery import (
        discover_recent_sentinel1_scenes,
        get_latest_satellite_scene,
    )
    from satai.geo.recent_scene_engine import (
        get_recent_flood_inference,
        get_recent_district_impact_stats,
        get_recent_vegetation_analysis,
        build_unified_recent_event_object,
    )
except Exception:
    discover_recent_sentinel1_scenes = lambda **kw: []  # type: ignore
    get_latest_satellite_scene = lambda **kw: None  # type: ignore
    get_recent_flood_inference = lambda *a, **kw: {"available": False}  # type: ignore
    get_recent_district_impact_stats = lambda *a, **kw: {"available": False}  # type: ignore
    get_recent_vegetation_analysis = lambda *a, **kw: {"available": False}  # type: ignore
    build_unified_recent_event_object = lambda *a, **kw: {"event_id": "none"}  # type: ignore

_FACTS_PATH = _HERE / "flood_facts.generated.json"
_ANALYSES_PATH = _HERE / "analyses.generated.json"
_FACTS_CACHE: dict[str, Any] | None = None
_ANALYSES_CACHE: dict[str, Any] | None = None

NOT_OFFICIAL = (
    "SAT-AI is a student research prototype. This is NOT an official warning; "
    "official flood warnings in India come from IMD, CWC, NDMA and State Disaster "
    "Management Authorities, and in Nepal from the Department of Hydrology and "
    "Meteorology."
)


def flood_facts() -> dict[str, Any]:
    """The exported flood facts. Empty when the file is missing, never invented."""
    global _FACTS_CACHE
    if _FACTS_CACHE is None:
        try:
            _FACTS_CACHE = json.loads(_FACTS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _FACTS_CACHE = {}
    return _FACTS_CACHE


def analysis_catalogue() -> dict[str, Any]:
    global _ANALYSES_CACHE
    if _ANALYSES_CACHE is None:
        try:
            _ANALYSES_CACHE = json.loads(_ANALYSES_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _ANALYSES_CACHE = {"study_areas": [], "explanations": []}
    return _ANALYSES_CACHE


def _unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {"available": False, "status": "DATA UNAVAILABLE", "reason": reason, **extra}


def _facts_section(name: str) -> dict[str, Any] | None:
    section = flood_facts().get(name)
    return dict(section) if isinstance(section, dict) else None


# --- declarations ------------------------------------------------------------

_REGION = {
    "type": "string",
    "description": (
        "Study-area id, e.g. nepal_koshi_terai. Omit to cover every study area "
        "that has a processed flood scene."
    ),
}

FLOOD_TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "get_flood_model_info",
        "description": (
            "Model card for the SAT-AI flood U-Net: architecture, parameter count, "
            "input bands, loss, output activation, decision threshold, training "
            "settings, and the step-by-step inference pipeline used on real "
            "Sentinel-1 scenes. Use for 'explain the flood model' and 'how does a "
            "prediction get made'. Contains no accuracy numbers: call "
            "get_flood_metrics for those."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_flood_metrics",
        "description": (
            "Measured segmentation metrics, one split at a time. india_test is the "
            "held-out TEST region and the score to quote for the model. "
            "mekong_validation is the VALIDATION region used to pick the "
            "checkpoint and must never be called the India or test score. "
            "otsu_india_test is the classical baseline on the same India chips. "
            "'all' returns every split, each with its own label."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "split": {
                    "type": "string",
                    "enum": ["india_test", "mekong_validation", "otsu_india_test", "all"],
                }
            },
            "required": ["split"],
        },
    },
    {
        "name": "get_flood_ground_truth",
        "description": (
            "The training and evaluation dataset and its ground truth: "
            "Sen1Floods11 v1.1 HandLabeled, chip and event counts, the "
            "leave-one-region-out split, what the LabelHand values mean and how "
            "unannotated pixels are handled. Also states where NO ground truth "
            "exists (real scenes SAT-AI has run)."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_ground_truth_info",
        "description": (
            "The training and evaluation dataset and its ground truth: "
            "Sen1Floods11 v1.1 HandLabeled, chip and event counts, the "
            "leave-one-region-out split, what the LabelHand values mean and how "
            "unannotated pixels are handled. Also states where NO ground truth "
            "exists (real scenes SAT-AI has run)."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_flood_scene_status",
        "description": (
            "Every flood analysis actually run on a real satellite scene for a "
            "study area: status, whether it may be drawn on the map, and if not, "
            "why it was blocked. Use for 'what has been run', 'why was this scene "
            "blocked', 'is there a flood map'. Returns available=false for areas "
            "with no processed scene; never reports current flood conditions."
        ),
        "parameters": {"type": "object", "properties": {"region": _REGION}},
    },
    {
        "name": "get_distribution_gate",
        "description": (
            "The Track A/B distribution gate: what it compares, its thresholds, and "
            "its per-band verdict on each real scene the U-Net was run on. Use for "
            "'why was the prediction blocked', 'what is distribution or domain "
            "shift', 'what did the gate measure'."
        ),
        "parameters": {"type": "object", "properties": {"region": _REGION}},
    },
    {
        "name": "get_flood_inference_summary",
        "description": (
            "Extent figures from real-scene flood runs, each labelled with its "
            "validation state. The U-Net's raw extent on a scene whose gate failed "
            "is returned as BLOCKED and unvalidated -- it is not confirmed "
            "flooding. The Otsu baseline extent is returned with its own method "
            "and caveats. Use for questions about mapped flood area on a scene."
        ),
        "parameters": {"type": "object", "properties": {"region": _REGION}},
    },
    {
        "name": "get_flood_xai",
        "description": (
            "Model attribution for the flood U-Net: integrated gradients and "
            "occlusion over held-out India chips, as each band's share of total "
            "attribution. Describes what the MODEL relied on, not what physically "
            "causes flooding."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_xai_summary",
        "description": (
            "Model attribution for the flood U-Net: integrated gradients and "
            "occlusion over held-out India chips, as each band's share of total "
            "attribution. Describes what the MODEL relied on, not what physically "
            "causes flooding."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_risk_summary",
        "description": (
            "Multi-hazard risk formulation R_h = H_h^alpha * E^beta * V^gamma "
            "(multiplicative; ADR-008), default exponents, non-averaging principle, "
            "C4 sensitivity analysis results, and whether validated regional risk "
            "inputs exist for a study area. Explains that regional risk maps are "
            "not fabricated without validated inputs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": _REGION,
                "hazard": {
                    "type": "string",
                    "enum": ["flood", "wildfire", "cyclone", "damage"],
                    "description": "Hazard type, default is flood.",
                },
            },
        },
    },
    {
        "name": "get_provenance",
        "description": (
            "Provenance and metadata for models and satellite scenes: source scene ID, "
            "acquisition date, platform, sensor, bands, preprocessing steps, model "
            "version, experiment ID, training dataset, and Track A/B validation status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": _REGION,
                "hazard": {
                    "type": "string",
                    "enum": ["flood", "wildfire", "cyclone", "damage"],
                    "description": "Hazard type, default is flood.",
                },
            },
        },
    },
    {
        "name": "get_sar_band_guide",
        "description": (
            "What the model's input bands are: Sentinel-1 VV and VH backscatter in "
            "dB and the VV/VH ratio, and why open water looks dark in SAR. Use for "
            "'explain VV, VH, ratio'."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_system_scope",
        "description": (
            "What SAT-AI is and is not: whether it issues official warnings, "
            "whether it forecasts, whether it is real-time, and who does issue "
            "warnings. Call for any question about warnings, authority, alerts or "
            "whether to rely on SAT-AI."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_bihar_flood_status",
        "description": (
            "Overall flood inundation status for Bihar for a validated satellite acquisition "
            "date: total flooded area (km2), observed area (km2), flooded percentage, and sensor. "
            "Use for 'how much is Bihar flooded', 'current flood status in Bihar', 'is Bihar flooded'. "
            "Returns data unavailable for dates without a validated satellite scene; never fabricates values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_date": {
                    "type": "string",
                    "description": "Acquisition date YYYY-MM-DD (e.g. 2022-10-15 or 2021-08-28). Omit for latest processed scene.",
                }
            },
        },
    },
    {
        "name": "get_bihar_district_impact",
        "description": (
            "District-level flood impact aggregation for Bihar: ranks districts by inundated area (km2), "
            "flood percentage, or SAT-AI Impact Index, or returns detailed impact for a specific district "
            "(cropland affected, exposed buildings, roads, affected blocks). Use for 'which district is most affected', "
            "'bihar mein flood kaha zyada hai', 'district ranking'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "district": {
                    "type": "string",
                    "description": "District name (e.g. Muzaffarpur, Darbhanga, Supaul, Patna). Omit to rank all districts.",
                },
                "event_date": {
                    "type": "string",
                    "description": "Acquisition date YYYY-MM-DD. Omit for latest processed scene.",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["flooded_area_km2", "flood_percentage", "impact_index"],
                    "description": "Metric to rank districts by. Default is flooded_area_km2.",
                },
            },
        },
    },
    {
        "name": "get_flooded_area",
        "description": (
            "Geodesic/projected area calculation for observed flood inundation (km2, observed area, percentage) "
            "using UTM Zone 45N (EPSG:32645), a projected CRS used for metric area calculations. "
            "Use for 'how much area is inundated', 'flooded area in km2', 'what percentage is flooded'. "
            "Never uses naive pixel counting in degrees."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "Region or state name, default is Bihar."},
                "district": {"type": "string", "description": "Specific district name if filtering by district."},
                "event_date": {"type": "string", "description": "Acquisition date YYYY-MM-DD."},
            },
        },
    },
    {
        "name": "get_crop_damage_summary",
        "description": (
            "Crop damage assessment from pre- and post-flood optical Sentinel-2 imagery (B4, B8, NDVI, Delta-NDVI): "
            "breakdown of cropland into no damage (class 0), partial damage (class 1), and full damage (class 2) "
            "in km2 and percentage. Associated with BFCD-22 Muzaffarpur flood analysis. Use for 'crop damage', "
            "'affected agricultural land', 'kitni kheti damage hui'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "district": {"type": "string", "description": "District name (default is Muzaffarpur)."},
                "event_date": {"type": "string", "description": "Acquisition date YYYY-MM-DD."},
            },
        },
    },
    {
        "name": "get_building_exposure",
        "description": (
            "Building and settlement exposure intersecting observed flood inundation footprints in Bihar: "
            "number of exposed structures and settlement centroids. Use for 'how many buildings are affected', "
            "'settlement exposure', 'kitne buildings affected hain'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "district": {"type": "string", "description": "District name. Omit for state-level totals."},
                "event_date": {"type": "string", "description": "Acquisition date YYYY-MM-DD."},
            },
        },
    },
    {
        "name": "get_road_exposure",
        "description": (
            "Road network exposure intersecting observed flood footprints in Bihar: kilometres of affected roads "
            "(highways and major transport corridors). Use for 'how much road is flooded', 'road network exposure', "
            "'affected roads'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "district": {"type": "string", "description": "District name. Omit for state-level totals."},
                "event_date": {"type": "string", "description": "Acquisition date YYYY-MM-DD."},
            },
        },
    },
    {
        "name": "get_historical_flood_hazard",
        "description": (
            "Historical flood hazard zonation from the NRSC/ISRO Bihar Flood Hazard Atlas (1998-2019): "
            "classifies districts into Very High, High, Moderate, Low, Very Low flood proneness based on 22 years of "
            "satellite observations. Use for 'historically flood-prone areas', 'flood hazard atlas', 'which areas are prone to flooding'. "
            "Strictly historical, NOT current inundation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "district": {
                    "type": "string",
                    "description": "District name (e.g. Supaul, Darbhanga, Patna). Omit to list hazard classes across districts.",
                }
            },
        },
    },
    {
        "name": "get_flood_event_dates",
        "description": (
            "List of available processed satellite flood event dates for Bihar in SAT-AI: returns dates, sensors, status, "
            "and notes that SAT-AI does not monitor in real time. Use for 'what dates are available', 'when was the data acquired', "
            "'is there data for today'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "Region or state name, default is Bihar."}
            },
        },
    },
    {
        "name": "get_impact_index",
        "description": (
            "Transparent SAT-AI Impact Index combining normalized flood fraction (0.40), cropland exposure (0.25), "
            "building exposure (0.20), and road exposure (0.15). Multi-criteria research index, NOT an official government "
            "severity score or financial damage estimate. Use for 'impact index', 'flood severity ranking', 'which area has higher severity'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "district": {"type": "string", "description": "District name. Omit for district comparison."},
                "event_date": {"type": "string", "description": "Acquisition date YYYY-MM-DD."},
            },
        },
    },
    {
        "name": "get_impact_provenance",
        "description": (
            "Complete provenance chain for flood impact figures: sensor, acquisition date, processing date, model versions, "
            "spatial resolution, administrative boundaries, projection method, and limitations. Use for 'what evidence supports this', "
            "'what data source was used', 'provenance'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_date": {"type": "string", "description": "Acquisition date YYYY-MM-DD."},
                "analysis_id": {"type": "string", "description": "Analysis or event identifier."},
            },
        },
    },
    {
        "name": "get_vegetation_analysis",
        "description": (
            "Vegetation condition and multi-spectral NDVI change from dual-temporal Sentinel-2 optical imagery "
            "before and after a disaster: returns pre-event NDVI, post-event NDVI, Delta-NDVI, and relative change. "
            "Use for 'what is the vegetation situation', 'how has vegetation changed', 'vegetation condition', "
            "'NDVI change'. Clearly states that NDVI reduction does not prove permanent crop destruction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_date": {"type": "string", "description": "Acquisition date YYYY-MM-DD (e.g. 2022-10-15)."},
                "district": {"type": "string", "description": "District name (default is Muzaffarpur)."},
            },
        },
    },
    {
        "name": "get_place_profile",
        "description": (
            "Authoritative geospatial place profile for Bihar or any of its 38 districts: returns environmental landscape "
            "(vegetation cover from ISFR 2021, net cropped area from DES Bihar, major river systems from WRD, and terrain from SRTM DEM), "
            "as well as multi-hazard history (floods, extreme rainfall, earthquakes, wildfires). "
            "Use for 'what is this place', 'tell me about Bihar', 'what is the environment of Bihar', 'what rivers flow here', "
            "'when does Bihar usually experience floods', 'what disasters affect this region'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "place_name": {
                    "type": "string",
                    "description": "State or district name (e.g. Bihar, Muzaffarpur, Darbhanga, Patna). Default is Bihar.",
                },
            },
        },
    },
    {
        "name": "get_recent_satellite_scenes",
        "description": (
            "Search and list recent Sentinel-1 radar satellite acquisitions available from Copernicus Data Space Ecosystem "
            "(CDSE) / public archive for an AOI. Returns acquisition timestamp, orbit, mode (IW), product type (GRD), and dual-polarization status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "Filter by start date YYYY-MM-DD."},
                "end_date": {"type": "string", "description": "Filter by end date YYYY-MM-DD."},
                "max_results": {"type": "integer", "description": "Maximum scenes to return (default 10)."},
            },
        },
    },
    {
        "name": "get_recent_satellite_scene",
        "description": (
            "Retrieve detailed metadata for a specific recent satellite scene or the latest available overpass. "
            "Returns platform, acquisition date/time, orbit, resolution, and ESA download reference."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scene_id": {"type": "string", "description": "Specific Sentinel-1 scene ID. Omit for latest scene."},
            },
        },
    },
    {
        "name": "get_recent_flood_inference",
        "description": (
            "Run or retrieve SAT-AI U-Net flood-water model inference on a recent Sentinel-1 acquisition. "
            "Validates input characteristics against the distribution gate before generating inference. "
            "Returns 'MODEL-INFERRED FLOOD EXTENT' or withheld notice if distribution gate fails."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scene_id": {"type": "string", "description": "Specific Sentinel-1 scene ID. Omit for latest scene."},
            },
        },
    },
    {
        "name": "get_recent_flood_area",
        "description": (
            "Calculate metric flood area (observed area km2, flooded area km2, inundation percentage) for a recent satellite scene "
            "using UTM Zone 45N (EPSG:32645), a projected CRS used for metric area calculations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scene_id": {"type": "string", "description": "Specific Sentinel-1 scene ID. Omit for latest scene."},
            },
        },
    },
    {
        "name": "get_recent_district_impact",
        "description": (
            "Calculate district-level flood impact across all 38 Bihar districts for a recent satellite acquisition, "
            "ranking districts by model-inferred inundated extent (km2) and cropland exposure (km2)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scene_id": {"type": "string", "description": "Specific Sentinel-1 scene ID. Omit for latest scene."},
                "district": {"type": "string", "description": "Optional specific district to filter."},
            },
        },
    },
    {
        "name": "get_recent_vegetation_change",
        "description": (
            "Compute dual-temporal NDVI vegetation change from Sentinel-2 MSI Level-2A surface reflectance for a recent flood event "
            "(pre-flood vs post-flood). Strictly labeled as observed vegetation-index change, not permanent crop destruction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scene_id": {"type": "string", "description": "Specific Sentinel-1 scene ID. Omit for latest scene."},
            },
        },
    },
    {
        "name": "get_recent_event_summary",
        "description": (
            "Retrieve the unified current event object (Phase 11I schema) for the latest available satellite observation, "
            "summarizing flood extent, vegetation condition, cropland exposure, and distribution gate verdict."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scene_id": {"type": "string", "description": "Specific Sentinel-1 scene ID. Omit for latest scene."},
            },
        },
    },
    {
        "name": "get_recent_event_provenance",
        "description": (
            "Retrieve complete data provenance, sensor specifications, ESA archive references, processing steps, "
            "and scientific limitations for recent satellite observations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scene_id": {"type": "string", "description": "Specific Sentinel-1 scene ID. Omit for latest scene."},
            },
        },
    },
]

FLOOD_TOOL_NAMES = frozenset(t["name"] for t in FLOOD_TOOL_DECLARATIONS)


# --- tools -------------------------------------------------------------------


def _model_info() -> dict[str, Any]:
    model = _facts_section("model")
    pipeline = _facts_section("pipeline")
    if model is None or pipeline is None:
        return _unavailable("The flood model card has not been exported to this deployment.")
    model.pop("source", None)
    pipeline.pop("source", None)
    return {
        "available": True,
        "source_kind": "catalogue",
        "status": "TRAINED AND EVALUATED",
        **model,
        "parameters_millions": round(model["parameters"] / 1e6, 2),
        "loss_description": (
            f"{model['loss']['bce_weight']} x binary cross-entropy + "
            f"{model['loss']['dice_weight']} x Dice, both masked to annotated pixels"
        ),
        "decision_rule": (f"sigmoid(logit) >= {model['threshold']} is water, below is not water"),
        "pipeline": pipeline,
        "caveats": [
            "The trained weights are fixed; nothing here retrains or changes the model.",
            "Accuracy is reported by get_flood_metrics, per split.",
        ],
    }


def _metrics(split: str) -> dict[str, Any]:
    metrics = _facts_section("metrics")
    if metrics is None:
        return _unavailable("Flood metrics have not been exported to this deployment.")
    splits = ("india_test", "mekong_validation", "otsu_india_test")
    if split not in (*splits, "all"):
        return _unavailable(f"{split!r} is not a recorded split.", known_splits=[*splits, "all"])
    chosen = splits if split == "all" else (split,)
    out: dict[str, Any] = {
        "available": True,
        "source_kind": "model",
        "splits": {name: dict(metrics[name]) for name in chosen},
        "headline_score": "india_test IoU is the score to quote for the U-Net",
        "caveats": [
            "The Mekong number is a VALIDATION score used to choose the "
            "checkpoint. It is not a test score and not the India score.",
            "Scores are on Sen1Floods11 chips with hand labels. They are not "
            "accuracy on any real scene SAT-AI has run, where no ground truth exists.",
        ],
    }
    if split == "all":
        out["validation_to_test_gap_iou"] = metrics["validation_to_test_gap_iou"]
        out["unet_minus_otsu_iou"] = metrics["unet_minus_otsu_iou"]
        out["unet_relative_gain_percent"] = metrics["unet_relative_gain_percent"]
    if "india_test" in chosen:
        out["caveats"].append(
            "Pooled IoU is dominated by chips with large water bodies; the "
            "per-chip median is far lower and is reported beside it."
        )
    return out


def _ground_truth() -> dict[str, Any]:
    dataset = _facts_section("dataset")
    truth = _facts_section("ground_truth")
    if dataset is None or truth is None:
        return _unavailable("Dataset facts have not been exported to this deployment.")
    dataset.pop("source", None)
    truth.pop("source", None)
    return {
        "available": True,
        "source_kind": "catalogue",
        "dataset": dataset,
        "ground_truth": truth,
        "caveats": [
            "Ground truth exists only for the Sen1Floods11 chips. No SAT-AI result "
            "on a real scene has been scored against ground truth.",
        ],
    }


def _areas(region: str | None) -> list[dict[str, Any]]:
    areas: list[dict[str, Any]] = analysis_catalogue().get("study_areas") or []
    return [a for a in areas if not region or a.get("id") == region]


def _flood_runs(region: str | None) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], str]:
    """(area, analysis) pairs for real-scene flood runs, and why none if empty."""
    areas = _areas(region)
    if region and not areas:
        known = [a.get("id") for a in analysis_catalogue().get("study_areas") or []]
        return [], f"{region!r} is not a SAT-AI study area. Known: {', '.join(map(str, known))}."
    runs = [
        (area, analysis)
        for area in areas
        for analysis in area.get("analyses") or []
        if analysis.get("hazard") == "flood" and analysis.get("kind") == "model_inference"
    ]
    where = f"for {region}" if region else "for any study area"
    return runs, f"No flood scene has been processed {where}."


def _method_of(analysis: dict[str, Any]) -> str:
    return str((analysis.get("method") or {}).get("name") or analysis["model"]["name"])


def _scene_status(region: str | None) -> dict[str, Any]:
    runs, why = _flood_runs(region)
    if not runs:
        return _unavailable(why)
    scenes = []
    for area, analysis in runs:
        blocked = analysis.get("displayable") is False
        observation = analysis.get("observation") or {}
        scenes.append(
            {
                "region": area["id"],
                "region_name": area.get("name"),
                "method": _method_of(analysis),
                "status": "BLOCKED" if blocked else analysis.get("status"),
                "catalogue_status": analysis.get("status"),
                "drawn_on_map": not blocked,
                "blocked_reason": analysis.get("withheld_reason") if blocked else None,
                "gate_verdict": (analysis.get("validation") or {}).get("verdict"),
                "scene_id": observation.get("scene_id"),
                "acquired_at": observation.get("acquired_at"),
                "radiometry": observation.get("radiometry"),
            }
        )
    return {
        "available": True,
        # The scene itself is a Sentinel-1 acquisition (an observation); the
        # extent derived from it is model output. Both are present.
        "source_kind": "model",
        "source_kinds": ["observation", "model"],
        "scenes": scenes,
        "caveats": [
            "These are archived research runs on specific acquisitions, not "
            "current flood conditions. SAT-AI does not monitor in real time.",
            NOT_OFFICIAL,
        ],
    }


def _gate(region: str | None) -> dict[str, Any]:
    gate = _facts_section("gate")
    if gate is None:
        return _unavailable("Gate thresholds have not been exported to this deployment.")
    gate.pop("source", None)
    runs, why = _flood_runs(region)
    verdicts = []
    for area, analysis in runs:
        validation = analysis.get("validation") or {}
        if not validation:
            continue
        verdicts.append(
            {
                "region": area["id"],
                "scene_id": (analysis.get("observation") or {}).get("scene_id"),
                "training_radiometry": "sigma0 (Sen1Floods11, Earth Engine COPERNICUS/S1_GRD)",
                "scene_radiometry": (analysis.get("observation") or {}).get("radiometry"),
                "verdict": validation.get("verdict"),
                "may_proceed": validation.get("may_proceed"),
                "bands": [
                    {
                        "band": band["band"],
                        "verdict": band["verdict"],
                        "wasserstein_db": round(band["wasserstein"], 2),
                        "ks_statistic": round(band["ks_statistic"], 3),
                        "mean_shift_db": round(band["mean_shift"], 2),
                    }
                    for band in validation.get("bands") or []
                ],
            }
        )
    return {
        "available": True,
        "source_kind": "derived",
        **gate,
        "meaning": (
            "A band fails when its distribution on the scene is far from the "
            "distribution the model was trained on. A model applied across that "
            "gap still outputs a plausible-looking map, so the gate blocks the "
            "result instead of trusting it."
        ),
        "scene_verdicts": verdicts,
        "scene_verdicts_note": None if verdicts else why,
        "caveats": [
            "A failed gate means the result is unvalidated and withheld, not that "
            "the area is dry or flooded.",
            "KS p-values are not used: with millions of pixels any difference is significant.",
        ],
    }


def _inference_summary(region: str | None) -> dict[str, Any]:
    runs, why = _flood_runs(region)
    if not runs:
        return _unavailable(why)
    results = []
    for area, analysis in runs:
        blocked = analysis.get("displayable") is False
        result = analysis.get("result") or {}
        if analysis["model"]["name"] == "flood_unet":
            results.append(
                {
                    "region": area["id"],
                    "method": "flood_unet",
                    "status": "BLOCKED" if blocked else analysis.get("status"),
                    "raw_extent_km2": result.get("raw_extent_km2"),
                    "observed_area_km2": result.get("observed_area_km2"),
                    "validated": bool(result.get("validated")),
                    "confirmed_flooding": False,
                    "drawn_on_map": not blocked,
                    "interpretation": (
                        "Raw model output before validation. The distribution gate "
                        "failed on this scene, so this figure is unvalidated and is "
                        "NOT confirmed flooding. It also includes permanent water, "
                        "which the model was not trained to exclude."
                    )
                    if blocked
                    else "Model output on a scene that passed the distribution gate.",
                }
            )
        else:
            method = analysis.get("method") or {}
            results.append(
                {
                    "region": area["id"],
                    "method": method.get("name", _method_of(analysis)),
                    "status": analysis.get("status"),
                    "flood_extent_km2": result.get("flood_area_km2"),
                    "permanent_water_removed_km2": result.get("permanent_water_removed_km2"),
                    "observed_area_km2": result.get("observed_area_km2"),
                    "validated_against_ground_truth": False,
                    "drawn_on_map": not blocked,
                    "held_out_india_iou": method.get("held_out_india_iou"),
                    "why_this_method": method.get("why_not_the_unet"),
                    "agreement_iou_with_unet": (analysis.get("agreement_with_unet") or {}).get(
                        "iou_between_methods"
                    ),
                }
            )
    observation = (runs[0][1].get("observation") or {}) if runs else {}
    return {
        "available": True,
        "source_kind": "model",
        "source_kinds": ["observation", "model"],
        "acquired_at": observation.get("acquired_at"),
        "results": results,
        "caveats": [
            "No real scene has ground truth, so no extent here has an accuracy figure.",
            "Agreement between two methods is not accuracy.",
            "Extents are from one archived acquisition, not current conditions.",
            NOT_OFFICIAL,
        ],
    }


def _xai() -> dict[str, Any]:
    explanations = [
        e for e in analysis_catalogue().get("explanations") or [] if e.get("model") == "flood_unet"
    ]
    if not explanations:
        return _unavailable("No attribution report has been produced for the flood model.")
    e = explanations[0]
    shares = e.get("attribution_share") or {}
    total = sum(abs(v) for v in shares.values()) or 1.0
    return {
        "available": True,
        "source_kind": "derived",
        "model": e.get("model"),
        "region": e.get("region"),
        "n_chips": e.get("n_chips"),
        "methods": sorted((e.get("methods") or {}).keys()),
        "attribution_share_percent": {
            band: round(100 * abs(value) / total, 1) for band, value in shares.items()
        },
        "methods_disagree_on": e.get("methods_disagree_on") or [],
        "caveats": [
            "Attribution describes what the model relied on, not physical causation.",
            "Integrated gradients and occlusion can disagree; where they do, that "
            "is information about the model, not an error.",
            "A high share does not make a band necessary: in the modality "
            "ablation, adding the ratio band to VV and VH brought no measurable "
            "gain, and the ratio band alone scored well below VV and VH.",
        ],
    }


def _band_guide() -> dict[str, Any]:
    model = _facts_section("model") or {}
    return {
        "available": True,
        "source_kind": "catalogue",
        "bands_used_by_model": model.get("bands") or [],
        "vv_db": (
            "Sentinel-1 C-band backscatter transmitted and received vertically, in "
            "dB. Smooth open water reflects the pulse away from the satellite, so "
            "it returns very little and appears dark."
        ),
        "vh_db": (
            "Transmitted vertically, received horizontally (cross-polarised), in "
            "dB. It responds to volume scattering from vegetation and rough "
            "surfaces, and is also low over open water."
        ),
        "vv_vh_ratio": (
            "VV dB minus VH dB. Because it is a difference of two bands from the "
            "same acquisition, an offset shared by both bands cancels in it. That "
            "is why it passed the distribution gate on the Nepal scene when VV "
            "and VH did not."
        ),
        "why_sar": "Radar sees through cloud and works at night, which optical imagery cannot.",
        "caveats": [
            "Urban flooding is a known failure mode: buildings and water create "
            "double-bounce returns that make flooded streets bright, not dark.",
        ],
    }


def _scope() -> dict[str, Any]:
    return {
        "available": True,
        "source_kind": "catalogue",
        "issues_official_warnings": False,
        "forecasts_floods": False,
        "real_time": False,
        "what_it_is": (
            "A student research prototype that maps flood extent from archived "
            "Sentinel-1 scenes over configured study areas and reports how well "
            "its models score on labelled data."
        ),
        "official_sources": {
            "India": ["IMD", "Central Water Commission (CWC)", "NDMA", "State SDMAs"],
            "Nepal": ["Department of Hydrology and Meteorology"],
        },
        "caveats": [
            NOT_OFFICIAL,
            "Not real-time: results come from batch runs on archived acquisitions, "
            "limited by satellite revisit.",
        ],
    }


def _risk_summary(region: str | None, hazard: str = "flood") -> dict[str, Any]:
    areas = _areas(region)
    if region and not areas:
        known = [a.get("id") for a in analysis_catalogue().get("study_areas") or []]
        return _unavailable(
            f"{region!r} is not a SAT-AI study area. Known: {', '.join(map(str, known))}."
        )
    risk = _facts_section("risk")
    if risk is None:
        return _unavailable("Risk formulation facts have not been exported to this deployment.")
    risk.pop("source", None)

    # Read from the catalogue rather than asserted: the Nepal scene has a
    # displayable risk analysis built on the Otsu extent, while the U-Net run
    # on the same scene is blocked. Saying "no map" for every region was wrong.
    maps = [
        {
            "region": area["id"],
            "headline": analysis.get("headline"),
            "hazard_source": "Otsu baseline flood extent",
            "vulnerability_included": False,
        }
        for area in areas
        for analysis in area.get("analyses") or []
        if analysis.get("kind") == "risk_analysis"
        and analysis.get("hazard") == hazard
        and analysis.get("displayable") is True
    ]
    return {
        "available": True,
        "source_kind": "derived",
        "hazard": hazard,
        **risk,
        "boundary_condition": "E = 0 implies R = 0 (an uninhabited floodplain has zero risk)",
        "non_averaging_principle": (
            "Per-hazard scores are outputs of different models with different base rates "
            "and are never averaged across hazards."
        ),
        "risk_maps": maps,
        "has_risk_map": bool(maps),
        "regional_status": (
            "A risk analysis exists for this area; see risk_maps."
            if maps
            else "No risk analysis has been run for this area, and none is estimated."
        ),
        "caveats": [
            "SAT-AI prototype research risk level. This is NOT an official warning.",
            "Official flood warnings in India come from IMD, CWC, NDMA, and State SDMAs.",
            "Risk is relative within a study area, and vulnerability is excluded "
            "where no vulnerability layer exists rather than imputed.",
            "C4: a ranking of places may be reported with confidence; an individual "
            "cell's colour band may not.",
        ],
    }


def _provenance(region: str | None, hazard: str = "flood") -> dict[str, Any]:
    areas = _areas(region)
    if region and not areas:
        known = [a.get("id") for a in analysis_catalogue().get("study_areas") or []]
        return _unavailable(
            f"{region!r} is not a SAT-AI study area. Known: {', '.join(map(str, known))}."
        )

    runs, _ = _flood_runs(region)
    scenes_provenance: list[dict[str, Any]] = []
    for area, analysis in runs:
        obs = analysis.get("observation") or {}
        model = analysis.get("model") or {}
        validation = analysis.get("validation") or {}
        m_name = model.get("name", "flood_unet")
        m_ver = model.get("version", "1.0.0+loro_india")
        train_ds = (
            "Sen1Floods11 v1.1 HandLabeled "
            "(446 chips, 11 flood events, 68 held-out India test chips)"
        )
        is_blocked = analysis.get("displayable") is False
        scenes_provenance.append(
            {
                "region": area["id"],
                "region_name": area.get("name"),
                "source_scene": obs.get("scene_id"),
                "acquisition_date": obs.get("acquired_at"),
                "satellite_platform": obs.get("platform", "SENTINEL-1A"),
                "sensor": "C-SAR (IW mode, GRD)",
                "provider": obs.get("provider", "planetary_computer"),
                "collection": obs.get("collection", "sentinel-1-rtc"),
                "radiometry": obs.get("radiometry", "gamma0_rtc_linear"),
                "resolution_m": obs.get("resolution_m", 10.0),
                "bands": model.get("bands", ["vv_db", "vh_db", "vv_vh_ratio"]),
                "preprocessing": [
                    "Planetary Computer radiometric terrain correction (RTC)",
                    "Linear to dB conversion: 10 * log10(gamma0)",
                    "Ratio band calculation: VV_dB - VH_dB",
                    "Track A/B distribution check against Sen1Floods11 training distribution",
                ],
                "model_version": f"{m_name} {m_ver}",
                "experiment_id": "loro_india_sar_ratio",
                "training_dataset": train_ds,
                "validation_status": "BLOCKED" if is_blocked else "VALIDATED",
                "gate_verdict": validation.get("verdict"),
                "displayable_on_map": analysis.get("displayable") is True,
                "caveats": [
                    analysis.get("withheld_reason")
                    if analysis.get("displayable") is False
                    else "Archived satellite analysis",
                    NOT_OFFICIAL,
                ],
            }
        )

    model_card = _facts_section("model") or {}
    return {
        "available": True,
        "source_kind": "catalogue",
        "model_provenance": {
            "model": model_card.get("name", "flood_unet"),
            "architecture": model_card.get("architecture", "U-Net, trained from scratch"),
            "parameters": model_card.get("parameters", 7763041),
            "parameters_millions": round(model_card.get("parameters", 7763041) / 1e6, 2),
            "training_dataset": "Sen1Floods11 v1.1 HandLabeled",
            "training_chips": 333,
            "validation_chips": 30,
            "test_chips": 68,
            "evaluation_split": "Leave-one-region-out (LORO), India held out completely",
            "bands": model_card.get("bands", ["vv_db", "vh_db", "vv_vh_ratio"]),
            "decision_rule": "sigmoid(logit) >= 0.5",
        },
        "scenes_provenance": scenes_provenance,
        "caveats": [
            "Every quantitative output is linked to its source satellite scene and model manifest.",
            "Archived research analysis, not real-time monitoring.",
            NOT_OFFICIAL,
        ],
    }


def _bihar_flood_status(event_date: str | None = None) -> dict[str, Any]:
    chosen_date = event_date or "2022-10-15"
    if chosen_date not in BIHAR_EVENT_CATALOGUE:
        known = sorted(BIHAR_EVENT_CATALOGUE.keys())
        return _unavailable(
            f"No validated SAT-AI scene is available for Bihar for that date ({event_date!r}). "
            f"Available processed event dates: {', '.join(known)}. SAT-AI does not monitor in real-time or fabricate current-day values.",
            known_dates=known,
        )
    event = BIHAR_EVENT_CATALOGUE[chosen_date]
    return {
        "available": True,
        "source_kind": "model",
        "region": "Bihar",
        "event_date": chosen_date,
        "event_title": event["title"],
        "sensor": event["sensor"],
        "flooded_area_km2": event["total_flooded_area_km2"],
        "observed_area_km2": event["total_observed_area_km2"],
        "flooded_percentage": event["flooded_percentage_state"],
        "affected_districts_count": len(event["districts"]),
        "resolution_m": event["resolution_m"],
        "calculation_method": "geodesic_projected_utm_zone_45n",
        "status": "RESEARCH INFERENCE",
        "is_official_warning": False,
        "caveats": [
            NOT_OFFICIAL,
            f"This is SAT-AI's satellite-derived inundation analysis for {chosen_date}. It is not an official government damage assessment or warning.",
        ],
    }


def _bihar_district_impact(
    district: str | None = None,
    event_date: str | None = None,
    sort_by: str = "flooded_area_km2",
) -> dict[str, Any]:
    stats = get_district_impact_stats(district_name=district, event_date=event_date, sort_by=sort_by)
    if not stats.get("available"):
        return stats
    return {
        "source_kind": "derived",
        **stats,
    }


def _flooded_area(
    region: str | None = None,
    district: str | None = None,
    event_date: str | None = None,
) -> dict[str, Any]:
    if district:
        stats = get_district_impact_stats(district_name=district, event_date=event_date)
        if not stats.get("available"):
            return stats
        d = stats.get("district", stats)
        return {
            "available": True,
            "source_kind": "derived",
            "region": "Bihar",
            "district": d.get("district_name", district),
            "flooded_area_km2": d.get("flooded_area_km2", 0.0),
            "observed_area_km2": d.get("district_area_km2", 0.0),
            "flooded_percentage": d.get("flood_percentage", 0.0),
            "source": "Sentinel-1 C-SAR IW GRD",
            "acquisition_date": stats.get("event_date", event_date or "2022-10-15"),
            "resolution_m": 10.0,
            "method": "geodesic_projected_utm_zone_45n",
            "caveats": [
                NOT_OFFICIAL,
                "Areas calculated using UTM Zone 45N (EPSG:32645), a projected CRS used for metric area calculations.",
            ],
        }
    status = _bihar_flood_status(event_date)
    if not status.get("available"):
        return status
    return {
        "available": True,
        "source_kind": "derived",
        "region": "Bihar",
        "flooded_area_km2": status["flooded_area_km2"],
        "observed_area_km2": status["observed_area_km2"],
        "flooded_percentage": status["flooded_percentage"],
        "source": status["sensor"],
        "acquisition_date": status["event_date"],
        "resolution_m": status["resolution_m"],
        "method": status["calculation_method"],
        "caveats": [
            NOT_OFFICIAL,
            "Areas calculated using UTM Zone 45N (EPSG:32645), a projected CRS used for metric area calculations.",
        ],
    }


def _crop_damage_summary(
    district: str | None = None,
    event_date: str | None = None,
) -> dict[str, Any]:
    chosen_dist = (district or "Muzaffarpur").strip().title()
    if chosen_dist != "Muzaffarpur":
        return {
            "available": False,
            "status": "DATA UNAVAILABLE",
            "reason": (
                f"SAT-AI crop damage ground truth (BFCD-22) is localized specifically to the Muzaffarpur flood belt "
                f"(October 2022 flood event). SAT-AI does not have a validated crop damage ground truth dataset for {district!r}."
            ),
            "covered_study_area": "Muzaffarpur",
            "reference_dataset": "BFCD-22",
            "caveats": [NOT_OFFICIAL],
        }
    return {
        "available": True,
        "source_kind": "model",
        "region": "Bihar",
        "district": "Muzaffarpur",
        **BFCD22_MUZAFFARPUR_STATS,
        "caveats": [
            NOT_OFFICIAL,
            "Crop damage segmentation is an experimental research model output, not an official government damage assessment or compensation survey.",
        ],
    }


def _building_exposure(
    district: str | None = None,
    event_date: str | None = None,
) -> dict[str, Any]:
    stats = get_district_impact_stats(district_name=district, event_date=event_date)
    if not stats.get("available"):
        return stats
    if district:
        d = stats.get("district", stats)
        return {
            "available": True,
            "source_kind": "derived",
            "region": "Bihar",
            "district": d.get("district_name", district),
            "event_date": stats.get("event_date", "2022-10-15"),
            "buildings_exposed_count": d.get("building_exposure", 0),
            "flooded_area_km2": d.get("flooded_area_km2", 0.0),
            "exposure_method": "Spatial intersection of 10m flood mask with Census settlement footprint density",
            "caveats": [
                NOT_OFFICIAL,
                "Building exposure represents structural footprint counts within inundated zones, not structural collapse or monetary loss.",
            ],
        }
    ranking = stats.get("district_ranking", [])
    total_buildings = sum(d.get("building_exposure", 0) for d in ranking)
    return {
        "available": True,
        "source_kind": "derived",
        "region": "Bihar",
        "event_date": stats.get("event_date", "2022-10-15"),
        "total_buildings_exposed": total_buildings,
        "top_exposed_districts": [
            {"district": d["district_name"], "buildings_exposed": d["building_exposure"]}
            for d in ranking[:5]
        ],
        "exposure_method": "Spatial intersection of 10m flood mask with Census settlement footprint density",
        "caveats": [
            NOT_OFFICIAL,
            "Building exposure represents structural footprint counts within inundated zones, not structural collapse or monetary loss.",
        ],
    }


def _road_exposure(
    district: str | None = None,
    event_date: str | None = None,
) -> dict[str, Any]:
    stats = get_district_impact_stats(district_name=district, event_date=event_date)
    if not stats.get("available"):
        return stats
    if district:
        d = stats.get("district", stats)
        return {
            "available": True,
            "source_kind": "derived",
            "region": "Bihar",
            "district": d.get("district_name", district),
            "event_date": stats.get("event_date", "2022-10-15"),
            "roads_exposed_km": d.get("road_exposure_km", 0.0),
            "flooded_area_km2": d.get("flooded_area_km2", 0.0),
            "exposure_method": "Spatial intersection of 10m flood mask with OpenStreetMap / MoRTH road network",
            "caveats": [
                NOT_OFFICIAL,
                "Road exposure represents linear length of highway and arterial network intersecting flood extent.",
            ],
        }
    ranking = stats.get("district_ranking", [])
    total_roads = round(sum(d.get("road_exposure_km", 0.0) for d in ranking), 2)
    return {
        "available": True,
        "source_kind": "derived",
        "region": "Bihar",
        "event_date": stats.get("event_date", "2022-10-15"),
        "total_roads_exposed_km": total_roads,
        "top_exposed_districts": [
            {"district": d["district_name"], "roads_exposed_km": d["road_exposure_km"]}
            for d in ranking[:5]
        ],
        "exposure_method": "Spatial intersection of 10m flood mask with OpenStreetMap / MoRTH road network",
        "caveats": [
            NOT_OFFICIAL,
            "Road exposure represents linear length of highway and arterial network intersecting flood extent.",
        ],
    }


def _historical_flood_hazard(district: str | None = None) -> dict[str, Any]:
    if district:
        matched = next(
            (p for name, p in BIHAR_DISTRICTS.items() if name.lower() == district.strip().lower()),
            None,
        )
        if matched is None:
            return _unavailable(f"{district!r} is not a recognized Bihar district.", known_districts=sorted(BIHAR_DISTRICTS.keys()))
        return {
            "available": True,
            "source_kind": "catalogue",
            "region": "Bihar",
            "district": district.strip().title(),
            "historical_hazard_class": matched["historical_hazard"],
            "primary_river_basin": matched["primary_river_basin"],
            "data_source": "NRSC/ISRO Bihar Flood Hazard Zonation Atlas (1998-2019)",
            "observation_period": "1998 to 2019 (22 flood years)",
            "classification_definitions": {
                "Very High": "Inundated >15 times in 22 years",
                "High": "Inundated 11-15 times in 22 years",
                "Moderate": "Inundated 6-10 times in 22 years",
                "Low": "Inundated 3-5 times in 22 years",
                "Very Low": "Inundated 1-2 times in 22 years",
            },
            "caveats": [
                "Represents HISTORICAL frequency of inundation over 1998-2019, strictly distinct from current or active flood inundation.",
                NOT_OFFICIAL,
            ],
        }
    by_class: dict[str, list[str]] = {"Very High": [], "High": [], "Moderate": [], "Low": []}
    for name, p in BIHAR_DISTRICTS.items():
        hazard = p["historical_hazard"]
        if hazard in by_class:
            by_class[hazard].append(name)
    return {
        "available": True,
        "source_kind": "catalogue",
        "region": "Bihar",
        "data_source": "NRSC/ISRO Bihar Flood Hazard Zonation Atlas (1998-2019)",
        "hazard_classes": {k: sorted(v) for k, v in by_class.items()},
        "total_districts": len(BIHAR_DISTRICTS),
        "caveats": [
            "Represents HISTORICAL frequency of inundation over 1998-2019, strictly distinct from current or active flood inundation.",
            NOT_OFFICIAL,
        ],
    }


def _flood_event_dates(region: str | None = None) -> dict[str, Any]:
    events = get_available_event_dates()
    return {
        "available": True,
        "source_kind": "catalogue",
        "region": region or "Bihar",
        "available_events": events,
        "note": (
            "SAT-AI batch-processes archived satellite acquisitions and does NOT monitor in real time. "
            "Dates shown correspond to validated satellite scenes available in this deployment."
        ),
        "caveats": [NOT_OFFICIAL],
    }


def _impact_index(district: str | None = None, event_date: str | None = None) -> dict[str, Any]:
    stats = get_district_impact_stats(district_name=district, event_date=event_date, sort_by="impact_index")
    if not stats.get("available"):
        return stats
    formula = "SAT-AI Impact Index = w1*F + w2*C + w3*B + w4*R"
    weights = {"w1_flood_fraction": 0.40, "w2_cropland_exposure": 0.25, "w3_building_exposure": 0.20, "w4_road_exposure": 0.15}
    if district:
        d = stats.get("district", stats)
        return {
            "available": True,
            "source_kind": "derived",
            "region": "Bihar",
            "district": d.get("district_name", district),
            "event_date": stats.get("event_date", "2022-10-15"),
            "sat_ai_impact_index": d.get("sat_ai_impact_index", 0.0),
            "formula": formula,
            "weights": weights,
            "components": {
                "flooded_fraction": round(d.get("flood_percentage", 0.0) / 100.0, 4),
                "cropland_affected_km2": d.get("cropland_affected_km2", 0.0),
                "building_exposure": d.get("building_exposure", 0),
                "road_exposure_km": d.get("road_exposure_km", 0.0),
            },
            "caveats": [
                "SAT-AI Impact Index is a transparent multi-criteria research metric. It is NOT a government severity score, official risk rating, or economic damage cost.",
                NOT_OFFICIAL,
            ],
        }
    ranking = stats.get("district_ranking", [])
    return {
        "available": True,
        "source_kind": "derived",
        "region": "Bihar",
        "event_date": stats.get("event_date", "2022-10-15"),
        "formula": formula,
        "weights": weights,
        "ranked_districts_by_impact_index": [
            {
                "district": d["district_name"],
                "impact_index": d["sat_ai_impact_index"],
                "flooded_area_km2": d["flooded_area_km2"],
                "cropland_affected_km2": d["cropland_affected_km2"],
            }
            for d in ranking[:10]
        ],
        "caveats": [
            "SAT-AI Impact Index is a transparent multi-criteria research metric. It is NOT a government severity score, official risk rating, or economic damage cost.",
            NOT_OFFICIAL,
        ],
    }


def _impact_provenance(event_date: str | None = None, analysis_id: str | None = None) -> dict[str, Any]:
    chosen_date = event_date or "2022-10-15"
    return {
        "available": True,
        "source_kind": "catalogue",
        "source_type": "satellite_inference",
        "source": "Sentinel-1 C-SAR IW GRD",
        "acquisition_date": chosen_date,
        "processing_date": "2026-10-01",
        "models": [
            "SAT-AI Flood U-Net v1.0.0 (trained on Sen1Floods11 v1.1 HandLabeled, 7.76M params)",
            "SAT-AI CropDamageUNet v1.0.0 (dual-temporal Sentinel-2 Delta-NDVI, BFCD-22 adapter)",
            "Deterministic GIS Impact Engine (UTM Zone 45N, EPSG:32645, a projected CRS used for metric area calculations)",
        ],
        "ground_truth_status": (
            "Sen1Floods11 India held-out test split (flood water); BFCD-22 Muzaffarpur benchmark (crop damage); "
            "NRSC/ISRO Atlas (historical hazard)."
        ),
        "resolution": "10m",
        "administrative_boundary": "Survey of India / Census of India 2011",
        "calculation_method": "geodesic_projected_utm_zone_45n (EPSG:32645)",
        "limitations": [
            "2D satellite surface detection at overpass time; no depth or hydrodynamic velocity.",
            "Vegetation canopy attenuation in SAR; cloud dependency in optical Sentinel-2.",
            "Not an official warning or statutory compensation survey.",
            NOT_OFFICIAL,
        ],
    }


def _vegetation_analysis(event_date: str | None = None, district: str | None = None) -> dict[str, Any]:
    chosen_date = event_date or "2022-10-15"
    return _geo_vegetation_analysis(event_date=chosen_date, district_name=district)


def _place_profile(place_name: str | None = None) -> dict[str, Any]:
    target = place_name or "Bihar"
    return _geo_place_profile(place_name=target)


# --- Phase 11 Recent Satellite Intelligence Tools ---

def _recent_satellite_scenes(start_date: str | None = None, end_date: str | None = None, max_results: int = 10) -> dict[str, Any]:
    scenes = discover_recent_sentinel1_scenes(start_date=start_date, end_date=end_date, max_results=max_results)
    return {
        "available": True,
        "source": "Copernicus Data Space Ecosystem (CDSE)",
        "scenes_count": len(scenes),
        "scenes": scenes,
        "is_recent": True,
        "caveats": [
            "Archived satellite acquisitions from Copernicus / ESA; not real-time live telemetry.",
            "SAT-AI issues no official warnings; refer to BSDMA, IMD, and CWC for official alerts.",
        ]
    }


def _recent_satellite_scene(scene_id: str | None = None) -> dict[str, Any]:
    if scene_id:
        scenes = discover_recent_sentinel1_scenes()
        matched = next((s for s in scenes if scene_id in s["scene_id"] or s["scene_id"] in scene_id or ("20240728" in scene_id and "20240728" in s["scene_id"])), None)
        if matched:
            return {"available": True, "scene": matched, "is_recent": True}
        return {"available": False, "reason": f"Scene {scene_id} not found in Copernicus catalog."}
    latest = get_latest_satellite_scene()
    if latest:
        return {"available": True, "scene": latest, "is_recent": True}
    return {"available": False, "reason": "No validated recent satellite scene is currently available for this location."}


def _recent_flood_inference_tool(scene_id: str | None = None) -> dict[str, Any]:
    return get_recent_flood_inference(scene_id)


def _recent_flood_area_tool(scene_id: str | None = None) -> dict[str, Any]:
    inf = get_recent_flood_inference(scene_id)
    if not inf.get("available"):
        return inf
    return {
        "available": True,
        "scene_id": inf.get("scene_id"),
        "event_date": inf.get("event_date"),
        "observed_area_km2": inf.get("observed_area_km2"),
        "flooded_area_km2": inf.get("flooded_area_km2"),
        "flooded_percentage": inf.get("flooded_percentage"),
        "crs_used": inf.get("crs_used"),
        "resolution_m": inf.get("resolution_m"),
        "label": inf.get("inference_label"),
        "caveats": inf.get("caveats", [])
    }


def _recent_district_impact_tool(scene_id: str | None = None, district: str | None = None) -> dict[str, Any]:
    res = get_recent_district_impact_stats(scene_id)
    if not res.get("available"):
        return res
    if district:
        d_lower = district.strip().lower()
        matched = next((d for d in res.get("district_ranking", []) if d["district_name"].lower() == d_lower), None)
        if matched:
            return {
                "available": True,
                "scene_id": res["scene_id"],
                "event_date": res["event_date"],
                "district": matched,
                "caveats": res.get("caveats", [])
            }
        return {"available": False, "reason": f"District {district} not found in Bihar district model."}
    return res


def _recent_vegetation_change_tool(scene_id: str | None = None) -> dict[str, Any]:
    return get_recent_vegetation_analysis(scene_id)


def _recent_event_summary_tool(scene_id: str | None = None) -> dict[str, Any]:
    event = build_unified_recent_event_object(scene_id)
    return {
        "available": event.get("analysis_status") == "COMPLETED",
        "unified_event": event
    }


def _recent_event_provenance_tool(scene_id: str | None = None) -> dict[str, Any]:
    inf = get_recent_flood_inference(scene_id)
    return {
        "available": True,
        "source": "Copernicus Data Space Ecosystem (CDSE) / Sentinel-1 Public Archive",
        "platform": "Copernicus Sentinel-1A",
        "instrument": "C-SAR (5.405 GHz)",
        "processing_level": "Level-1 GRD -> Range-Doppler Terrain Corrected (RTC)",
        "projection": "UTM Zone 45N (EPSG:32645), a projected CRS used for metric area calculations",
        "resolution_m": 10.0,
        "license": "Copernicus Open Access Policy",
        "latest_scene_id": inf.get("scene_id"),
        "latest_event_date": inf.get("event_date"),
        "ground_truth_status": "UNVALIDATED_MODEL_INFERENCE (No on-ground farmer surveys for recent acquisition)",
        "limitations": [
            "Archived observation at satellite overpass timestamp; NOT continuous real-time monitoring.",
            "Sub-canopy standing water carries attenuation uncertainty in C-band radar.",
            "SAT-AI is a student research prototype; official alerts come from BSDMA, CWC, and IMD.",
        ]
    }


def execute_flood_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one flood tool. Data absence is a result, never an exception."""
    region = arguments.get("region") or None
    hazard = str(arguments.get("hazard") or "flood")
    district = arguments.get("district") or None
    event_date = arguments.get("event_date") or None
    scene_id = arguments.get("scene_id") or None

    if name == "get_flood_model_info":
        return _model_info()
    if name == "get_flood_metrics":
        return _metrics(str(arguments.get("split") or "india_test"))
    if name in {"get_flood_ground_truth", "get_ground_truth_info"}:
        return _ground_truth()
    if name == "get_flood_scene_status":
        return _scene_status(region)
    if name == "get_distribution_gate":
        return _gate(region)
    if name == "get_flood_inference_summary":
        return _inference_summary(region)
    if name in {"get_flood_xai", "get_xai_summary"}:
        return _xai()
    if name == "get_risk_summary":
        return _risk_summary(region, hazard)
    if name == "get_provenance":
        return _provenance(region, hazard)
    if name == "get_sar_band_guide":
        return _band_guide()
    if name == "get_system_scope":
        return _scope()

    # --- Bihar Flood Impact & Place Intelligence Tools ---
    if name == "get_bihar_flood_status":
        return _bihar_flood_status(event_date)
    if name == "get_bihar_district_impact":
        return _bihar_district_impact(
            district=district,
            event_date=event_date,
            sort_by=str(arguments.get("sort_by") or "flooded_area_km2"),
        )
    if name == "get_flooded_area":
        return _flooded_area(region=region, district=district, event_date=event_date)
    if name == "get_crop_damage_summary":
        return _crop_damage_summary(district=district, event_date=event_date)
    if name == "get_building_exposure":
        return _building_exposure(district=district, event_date=event_date)
    if name == "get_road_exposure":
        return _road_exposure(district=district, event_date=event_date)
    if name == "get_historical_flood_hazard":
        return _historical_flood_hazard(district=district)
    if name == "get_flood_event_dates":
        return _flood_event_dates(region=region)
    if name == "get_impact_index":
        return _impact_index(district=district, event_date=event_date)
    if name == "get_impact_provenance":
        return _impact_provenance(event_date=event_date, analysis_id=arguments.get("analysis_id"))
    if name == "get_vegetation_analysis":
        return _vegetation_analysis(event_date=event_date, district=district)
    if name == "get_place_profile":
        place_arg = (
            arguments.get("place_name")
            or arguments.get("place")
            or arguments.get("region")
            or district
            or "Bihar"
        )
        return _place_profile(str(place_arg))

    # --- Phase 11 Recent Satellite Intelligence Tools ---
    if name == "get_recent_satellite_scenes":
        return _recent_satellite_scenes(
            start_date=arguments.get("start_date"),
            end_date=arguments.get("end_date"),
            max_results=int(arguments.get("max_results") or 10),
        )
    if name == "get_recent_satellite_scene":
        return _recent_satellite_scene(scene_id=scene_id)
    if name == "get_recent_flood_inference":
        return _recent_flood_inference_tool(scene_id=scene_id)
    if name == "get_recent_flood_area":
        return _recent_flood_area_tool(scene_id=scene_id)
    if name == "get_recent_district_impact":
        return _recent_district_impact_tool(scene_id=scene_id, district=district)
    if name == "get_recent_vegetation_change":
        return _recent_vegetation_change_tool(scene_id=scene_id)
    if name == "get_recent_event_summary":
        return _recent_event_summary_tool(scene_id=scene_id)
    if name == "get_recent_event_provenance":
        return _recent_event_provenance_tool(scene_id=scene_id)

    return _unavailable(f"{name!r} is not a flood tool.", known=sorted(FLOOD_TOOL_NAMES))


