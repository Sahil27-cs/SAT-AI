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


def execute_flood_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one flood tool. Data absence is a result, never an exception."""
    region = arguments.get("region") or None
    hazard = str(arguments.get("hazard") or "flood")
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
    return _unavailable(f"{name!r} is not a flood tool.", known=sorted(FLOOD_TOOL_NAMES))
