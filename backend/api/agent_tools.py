"""Tools Gemini may call, and the map actions it may trigger.

Every project-specific number the assistant states has to come through one of
these. That is the whole mechanism behind contribution C1: the union of what
the tools returned in a turn is the set of values the answer is permitted to
contain, and a number outside it is a violation the validator counts.

So each tool returns data **and** the provenance of that data, and each one has
an explicit unavailable form. There is no code path that returns a plausible
default: a region with no completed run yields ``available: false`` with the
reason and what would produce it, never a zero. A zero would arrive in the
language layer indistinguishable from a measurement.

``show_on_map`` is different in kind and is kept here deliberately. It returns
no data -- it returns an *intent* the frontend executes: activate these layers,
fly to this region. Letting the model express that as a tool call rather than
as prose is what connects the assistant to the map, and routing it through the
same declaration surface means it is recorded in the same audit trail as every
other call.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MAP_LAYERS",
    "TOOL_DECLARATIONS",
    "execute_tool",
    "groundable_values",
]

#: Layers the assistant may switch on. Constrained to this list so a model that
#: invents a layer name produces a rejected call rather than a silent no-op the
#: user reads as the map being broken.
MAP_LAYERS: tuple[str, ...] = (
    "aoi",
    "flood",
    "wildfire",
    "rainfall",
    "terrain",
    "risk",
    "damage",
)

_REGION_ENUM = [
    "bihar_ganga",
    "mumbai_mmr",
    "assam_brahmaputra",
    "kerala_periyar",
    "nepal_koshi_terai",
    "odisha_mahanadi",
    "uttarakhand_kumaon",
]

#: Gemini function declarations. Descriptions are written for the model, not
#: for a developer: each one states what the tool returns AND what it does not,
#: because a description that oversells its tool is the cheapest way to get a
#: model to call it for the wrong question.
TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "get_study_area",
        "description": (
            "Static facts about a configured SAT-AI study area: bounding box, area, "
            "country, primary hazards, study role, and any verified historical event "
            "with confirmed satellite coverage. This is catalogue data, not a "
            "measurement and not a prediction. Use it to answer 'what is this region' "
            "and to check whether a region is covered at all."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "enum": _REGION_ENUM},
            },
            "required": ["region"],
        },
    },
    {
        "name": "get_hazard_result",
        "description": (
            "The latest computed hazard result for a region: risk index, risk band, "
            "confidence, mapped extent and the satellite scenes behind it. Returns "
            "available=false when no batch run has produced a result, which is "
            "currently the case for every region. Never returns a zero to mean "
            "'no hazard'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "enum": _REGION_ENUM},
                "hazard": {
                    "type": "string",
                    "enum": ["flood", "wildfire", "cyclone", "damage"],
                },
            },
            "required": ["region", "hazard"],
        },
    },
    {
        "name": "get_model_info",
        "description": (
            "Model card for a SAT-AI model: architecture, training dataset, "
            "evaluation protocol, held-out test region, and measured metrics where "
            "an actual run produced them. Use this for 'how accurate is it', 'what "
            "data was it trained on', and any question about model performance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": ["flood_unet", "otsu_baseline"],
                    "description": (
                        "flood_unet is the deep model; otsu_baseline is the "
                        "classical method it is compared against."
                    ),
                }
            },
            "required": ["model"],
        },
    },
    {
        "name": "get_experiment",
        "description": (
            "Status and results of a research contribution C1-C4. Returns the "
            "measured numbers for experiments that have run and an explicit "
            "not-executed status with the blocking reason for those that have not."
        ),
        "parameters": {
            "type": "object",
            "properties": {"contribution": {"type": "string", "enum": ["C1", "C2", "C3", "C4"]}},
            "required": ["contribution"],
        },
    },
    {
        "name": "show_on_map",
        "description": (
            "Control the map the user is looking at: switch to a study area and "
            "activate layers. Call this whenever the user asks to see, show, "
            "display or zoom to something. Returns no data -- it moves the map. "
            "Call a data tool as well if the user also asked what the values are."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "enum": _REGION_ENUM},
                "layers": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(MAP_LAYERS)},
                    "description": (
                        "Layers to activate. Unavailable layers stay off and "
                        "the response says which."
                    ),
                },
            },
            "required": ["region"],
        },
    },
]


async def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    query_fn: Any,
    study_areas: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run one tool call. Never raises for a data-absence case.

    An exception here would become a 500 on a question the system can answer
    perfectly well with "that is not available". Genuine faults still raise.
    """
    if name == "get_study_area":
        return _study_area(arguments.get("region", ""), study_areas)
    if name == "get_hazard_result":
        return await _hazard_result(
            arguments.get("region", ""), arguments.get("hazard", "flood"), query_fn
        )
    if name == "get_model_info":
        return _model_info(arguments.get("model", ""))
    if name == "get_experiment":
        return _experiment(arguments.get("contribution", ""))
    if name == "show_on_map":
        return _show_on_map(arguments, study_areas)

    return {
        "available": False,
        "error": "unknown_tool",
        "message": (
            f"{name!r} is not a SAT-AI tool. Available: "
            f"{', '.join(t['name'] for t in TOOL_DECLARATIONS)}"
        ),
    }


def _study_area(region: str, study_areas: list[dict[str, Any]]) -> dict[str, Any]:
    area = next((a for a in study_areas if a["id"] == region), None)
    if area is None:
        return {
            "available": False,
            "reason": (
                f"{region!r} is not a SAT-AI study area. The system covers only its "
                f"configured regions and does not estimate anything elsewhere."
            ),
            "configured_regions": [a["id"] for a in study_areas],
        }

    verified = [e for e in area.get("events", []) if e.get("verified")]
    return {
        "available": True,
        "source_kind": "catalogue",
        "region": area["name"],
        "country": area["country"],
        "area_km2": area["areaKm2"],
        "bbox": area["bbox"],
        "primary_hazards": area["primaryHazards"],
        "study_role": area["studyRole"],
        # Declared, not obtained: see satai.geo.aoi. Named this way because
        # the model reads it and turns it into a sentence -- "has ground
        # truth" would become a claim that labels exist for the area.
        "label_source_declared": area["labelSourcesDeclared"],
        "verified_events": [
            {
                "name": e["name"],
                "date": e["occurredOn"],
                "sensor": e["sensor"],
                "status": e["displayStatus"],
            }
            for e in verified
        ],
        "caveats": [
            "Catalogue data: a study-area definition, not a measurement.",
            *(
                []
                if verified
                else [
                    "No verified hazard event is configured for this region, so no "
                    "historical analysis can be offered for it."
                ]
            ),
        ],
    }


async def _hazard_result(region: str, hazard: str, query_fn: Any) -> dict[str, Any]:
    rows = await query_fn(
        "latest_hazard_results",
        {"select": "*", "region_id": f"eq.{region}", "hazard": f"eq.{hazard}"},
    )
    if not rows:
        return {
            "available": False,
            "status": "MODEL RESULT NOT COMPUTED",
            "reason": (
                f"No {hazard} analysis has been computed for {region}. SAT-AI "
                f"produces results on a batch schedule tied to satellite revisit, "
                f"and this region has no completed run."
            ),
            "what_would_produce_it": (
                "A Track B acquisition followed by ml/flood/predict.py over this AOI."
            ),
        }

    row = rows[0]
    values = {
        key: row[key]
        for key in (
            "risk_index",
            "risk_band",
            "confidence",
            "flooded_area_km2",
            "population_exposed",
        )
        if row.get(key) is not None
    }
    return {
        "available": True,
        "source_kind": row.get("source_kind", "model"),
        "region": region,
        "hazard": hazard,
        **values,
        "scene_ids": row.get("scene_ids") or [],
        "observed_at": row.get("observed_at"),
        "caveats": [
            "SAT-AI prototype risk level. NOT an official warning. Official "
            "warnings for India come from IMD, NDMA and State Disaster "
            "Management Authorities.",
            *(row.get("caveats") or []),
        ],
    }


#: Measured on real runs. Every number here traces to a file in ml/experiments/;
#: a model with no completed evaluation carries status and no metrics rather
#: than placeholders.
_MODEL_CARDS: dict[str, dict[str, Any]] = {
    "flood_unet": {
        "available": True,
        "source_kind": "model",
        "name": "Flood segmentation U-Net",
        "architecture": "U-Net, 4 levels, base width 32, trained from scratch",
        "parameters": 7763041,
        "bands": ["vv_db", "vh_db", "vv_vh_ratio"],
        "training_dataset": "Sen1Floods11 v1.1 HandLabeled, 446 chips, 11 flood events",
        "split_protocol": "leave-one-region-out (ADR-009)",
        "test_region": "India (68 chips), held out from training and selection",
        "status": "TRAINED AND EVALUATED",
        "metrics": {
            "iou": 0.523,
            "f1": 0.6868,
            "precision": 0.7506,
            "recall": 0.6331,
            "threshold": 0.5,
        },
        "caveats": [
            "Validation IoU on the fold's validation region (Mekong) was 0.8679; "
            "the test score on India is 0.5230. The gap is what quoting a "
            "validation number as the model's score would overstate it by.",
            "Per-chip median IoU is 0.223 against a pooled 0.523: the pooled "
            "figure is dominated by chips with large water bodies.",
            "Known failure mode: urban double-bounce RAISES backscatter over "
            "flooded streets, inverting the signature the model relies on.",
            "Trained on 11 flood events; generalisation beyond those regimes is untested.",
        ],
    },
    "otsu_baseline": {
        "available": True,
        "source_kind": "model",
        "name": "Otsu classical baseline",
        "architecture": "Otsu threshold on Sentinel-1 VV with a unimodality guard",
        "training_dataset": "None -- unsupervised, threshold derived per chip",
        "split_protocol": "leave-one-region-out selection of the guard, 11 regions",
        "test_region": "India (68 chips)",
        "status": "EVALUATED",
        "metrics": {"iou": 0.3754, "f1": 0.5459, "precision": 0.7293, "recall": 0.4362},
        "caveats": [
            "No HAND terrain mask: Sen1Floods11 ships none, so the mask that "
            "removes false positives over tarmac and dry sand is absent. These "
            "are a lower bound on the method as operationally deployed.",
            "This is the bar the deep model has to clear, and does: 0.523 vs 0.375.",
        ],
    },
}


def _model_info(model: str) -> dict[str, Any]:
    card = _MODEL_CARDS.get(model)
    if card is None:
        return {
            "available": False,
            "reason": f"{model!r} is not a SAT-AI model.",
            "known_models": sorted(_MODEL_CARDS),
        }
    return dict(card)


_EXPERIMENTS: dict[str, dict[str, Any]] = {
    "C1": {
        "available": False,
        "status": "NOT EXECUTED",
        "title": "Measured grounding of an agent layer over Earth-observation outputs",
        "reason": (
            "The benchmark and the grounding validator exist and the validator "
            "scored 100% over 15 labelled cases in experiment 9a, but the "
            "28-question run has not been executed."
        ),
    },
    "C2": {
        "available": False,
        "status": "NOT EXECUTED",
        "title": "Quantified degradation under modality loss",
        "reason": (
            "A SAR-only ablation against SAR+ratio is running; rainfall and DEM "
            "arms need co-registered ancillary rasters that are not yet acquired."
        ),
    },
    "C3": {
        "available": False,
        "status": "NOT EXECUTED",
        "title": "Rural to urban domain-transfer gap for SAR flood segmentation",
        "reason": (
            "Requires labelled urban Indian flood imagery. Sen1Floods11's India "
            "chips are not urban, and no Mumbai ground truth has been acquired."
        ),
    },
    "C4": {
        "available": True,
        "status": "EXECUTED",
        "title": "Reproducible, sensitivity-analysed multi-hazard risk implementation",
        "source_kind": "index",
        "result": {
            "min_spearman_rho": 0.957,
            "max_cells_changing_band_percent": 18.7,
        },
        "interpretation": (
            "Exponents barely change which places rank riskiest but move up to "
            "18.7% of cells between colour bands. A ranking may be reported with "
            "confidence; a cell's band may not."
        ),
    },
}


def _experiment(contribution: str) -> dict[str, Any]:
    entry = _EXPERIMENTS.get(contribution.upper())
    if entry is None:
        return {
            "available": False,
            "reason": f"{contribution!r} is not a SAT-AI contribution.",
            "known": sorted(_EXPERIMENTS),
        }
    return dict(entry)


#: Which layers actually have data. Everything but the study-area outlines
#: depends on a batch run that has not happened, so the map action reports what
#: it could not switch on rather than silently activating an empty layer -- a
#: user who toggles FLOOD and sees nothing cannot tell "no flooding" from
#: "never computed".
_LAYER_AVAILABILITY: dict[str, str | None] = {
    "aoi": None,
    "flood": (
        "No flood extent raster exists: the inference pipeline has not been run for any region."
    ),
    "wildfire": "FIRMS ingestion requires FIRMS_MAP_KEY, which is not configured.",
    "rainfall": "GPM ingestion requires NASA Earthdata credentials, which are not configured.",
    "terrain": "DEM acquisition runs in the batch plane and has not been executed.",
    "risk": (
        "The risk engine needs hazard, exposure and vulnerability rasters it has not been given."
    ),
    "damage": "Change detection needs a pre/post image pair around a specific event.",
}


def _show_on_map(arguments: dict[str, Any], study_areas: list[dict[str, Any]]) -> dict[str, Any]:
    region = arguments.get("region", "")
    requested = [str(x) for x in (arguments.get("layers") or ["aoi"])]

    area = next((a for a in study_areas if a["id"] == region), None)
    if area is None:
        return {
            "available": False,
            "reason": f"{region!r} is not a SAT-AI study area; the map was not moved.",
        }

    activated = [layer for layer in requested if _LAYER_AVAILABILITY.get(layer) is None]
    unavailable = {
        layer: _LAYER_AVAILABILITY[layer]
        for layer in requested
        if _LAYER_AVAILABILITY.get(layer) is not None
    }
    if "aoi" not in activated:
        activated.append("aoi")

    return {
        "available": True,
        "source_kind": "catalogue",
        # The frontend reads this block and acts on it. Kept flat and explicit
        # so a change here is visible in the audit log rather than buried.
        "map_action": {
            "region": region,
            "bbox": area["bbox"],
            "activate_layers": activated,
        },
        "region_name": area["name"],
        "layers_activated": activated,
        "layers_unavailable": unavailable,
        "message": (
            f"Map moved to {area['name']}."
            + (f" Could not activate: {', '.join(unavailable)}." if unavailable else "")
        ),
    }


def groundable_values(results: list[dict[str, Any]]) -> list[float]:
    """Numbers the answer is permitted to state, from this turn's tool results.

    Walks nested dicts because metrics arrive one level down in a model card.
    Booleans are excluded: ``True`` is ``1`` in Python and would silently
    license the model to state "1" anywhere.
    """
    out: list[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            out.append(float(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    for result in results:
        if result.get("available"):
            walk(result)
    return out
