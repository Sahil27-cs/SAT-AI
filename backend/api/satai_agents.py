"""Agent plane for the deployed serving layer, on Google Gemini.

A tool-using geospatial agent, not a chatbot. Gemini receives function
declarations, decides which SAT-AI tools to call, receives their results, and
writes an answer from them. It may also move the map, which it does by calling
``show_on_map`` rather than by describing what the user should click.

The behavioural contract is the one the benchmark measures:

* values come from tools, never from the model;
* observations and model outputs stay distinct;
* the system refuses what it has no authority or capability to answer;
* when the language layer is unreachable the endpoint degrades to structured
  tool output and says so, rather than failing or inventing prose (req. 39).

**Refusals are checked before any tool call and before any model call.** A
refusal must not depend on a database lookup or an API round trip succeeding:
the case where the system is degraded is exactly the case where someone might
be asking whether to evacuate.

**On the grounding check.** This module used to validate only that numbers were
traceable, while ``satai.agents.grounding`` -- the validator scored in
experiment 9a -- also checked fabricated authority, observation/prediction
confusion and dropped caveats. The measurement reported for C1 was therefore
made by a stricter instrument than the one in front of users. All four checks
are implemented here with the same constants and semantics, and
``tests/test_grounding_parity.py`` runs both over a shared corpus and fails if
their verdicts diverge.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

from agent_tools import TOOL_DECLARATIONS, execute_tool, groundable_values
from gemini import (
    GeminiError,
    GeminiNotConfigured,
    call_args,
    call_gemini,
    call_name,
    describe_configuration,
    generation_settings,
    is_configured,
    model_turn,
    tool_turn,
    user_turn,
)

if TYPE_CHECKING:  # pragma: no cover
    from index import ChatRequest, ChatResponse

DISCLAIMER = (
    "SAT-AI prototype risk level. Research output from a student research "
    "prototype. This is NOT an official warning and does not replace IMD, NDMA "
    "or State Disaster Management Authority advisories."
)

#: How many times the model may call tools before it must answer. Three covers
#: the realistic chains (move the map, fetch a result, fetch the model card)
#: without letting a confused turn loop until the function times out. The
#: default; GEMINI_MAX_TOOL_ROUNDS overrides it within the bounds in gemini.py.
MAX_TOOL_ROUNDS = 3

FALLBACK_REGIONS = (
    "bihar_ganga",
    "mumbai_mmr",
    "assam_brahmaputra",
    "kerala_periyar",
    "nepal_koshi_terai",
    "odisha_mahanadi",
    "uttarakhand_kumaon",
)

SYSTEM_PROMPT = """\
You are the SAT-AI Flood Assistant, the analyst interface to SAT-AI, a student
research prototype that maps flood extent from Sentinel-1 radar and reports how
well its flood models score on labelled data. You answer FLOOD questions only.
For wildfire, cyclone, earthquake or anything else, say this assistant covers
flooding only.

You have tools. Use them. Every project-specific fact -- dataset, counts,
metrics, parameters, bands, loss, threshold, scene results, gate verdicts,
attribution shares -- must come from a tool result in this turn. Call the tool
first and answer from what it returns. General background (what radar is, what
IoU means) may be explained in words, without numbers of your own.

WHICH TOOL
- the model, architecture, loss, threshold, prediction pipeline: get_flood_model_info
- accuracy, IoU, F1, precision, recall: get_flood_metrics
- dataset, ground truth, labels, chips, events, split:
  get_ground_truth_info (or get_flood_ground_truth)
- what has been run on a scene, whether it is on the map, why it was blocked:
  get_flood_scene_status
- distribution gate, domain shift, sigma0 vs gamma0: get_distribution_gate
- mapped flood area on a scene, the raw U-Net extent: get_flood_inference_summary
- explainability, attribution, which band matters: get_xai_summary (or get_flood_xai)
- multi-hazard risk engine, formulation, sensitivity, exposure/vulnerability: get_risk_summary
- provenance, satellite scenes, sensors, metadata, audit trail: get_provenance
- VV, VH, the VV/VH ratio: get_sar_band_guide
- warnings, alerts, authority, forecasting, real-time: get_system_scope
- the Otsu baseline: get_flood_metrics (otsu_india_test) and
  get_flood_inference_summary
Call more than one when a question spans several.

ABSOLUTE RULES
1. Never state a number that is not in a tool result from this turn. Do not
   round a value into a different one, do not convert units, do not recall a
   figure. If a tool did not return it, say it is not available.
2. When a tool returns available=false, say DATA UNAVAILABLE and give its
   reason. Never substitute a plausible value.
3. Metric splits are different things. The India test score is the model's
   score (0.523 IoU). The Mekong number is a VALIDATION score (0.868 IoU) used to choose the
   checkpoint: never call it the India score, the test score or the model's
   accuracy. Always name the split next to any metric.
4. A BLOCKED or unvalidated result is not a finding about the ground. The raw
   U-Net extent on a scene whose distribution gate failed is unvalidated model
   output: never call it confirmed flooding, observed flooding or the flooded
   area. For the Nepal Koshi scene (74.8 km2 raw extent over 9,310.1 km2 scene area),
   state:
   "The model generated a raw 74.8 km² inference, but the distribution gate
   rejected the scene because its input distribution differed from the training
   distribution. Therefore the result is not treated as a validated flood extent
   and is not displayed as confirmed flooding."
   Never bypass the gate.
5. Keep observations and model outputs distinct. A Sentinel-1 acquisition is
   an observation. A flood extent is a model output. Neither is a forecast.
6. SAT-AI does not forecast floods, is not real-time, and gives NO official
   warnings. Whenever you report a scene result or answer about warnings, say
   it is not an official warning and name the official sources the tool gives
   (IMD, CWC, NDMA, State SDMAs).
7. Never invent current flood status, a risk value, a warning, an evacuation
   instruction, a phone number or a satellite observation.
8. Explainability is model attribution: what the model relied on, NOT physical
   causation.
9. For risk questions: the formulation is R_h = H_h^alpha * E^beta * V^gamma
   (multiplicative; where zero exposure implies zero risk). Never average across hazards.
   If validated regional exposure or vulnerability data is missing, do not
   fabricate a regional risk map or risk value.
10. Carry through the caveats attached to tool results.
11. When the user asks to see, show or zoom to something, call show_on_map.
    A blocked result is never drawn; say so if they ask for it.
12. DATASET SIZE vs TRAINING SIZE: get_flood_ground_truth returns both the
    total chip count (dataset.n_chips) and the split (train_chips,
    validation_chips, test_chips, reserved_chips). "How much data was it trained
    on" is train_chips, not the total. Name which one you are giving.
13. ANSWER STYLE:
    Write concise, natural responses (2-5 sentences for simple factual
    questions). Do not dump raw tool output, JSON keys or diagnostics unless the
    user asks to see the tool output or evidence.

Answer in short paragraphs or a few bullets. State what the data shows, what it
does not, and how far it can be trusted."""

# --- routing ---------------------------------------------------------------

_AGENT_TERMS: dict[str, dict[str, float]] = {
    "emergency": {
        "evacuat": 3.0,
        "emergency": 2.5,
        "helpline": 3.0,
        "shelter": 2.5,
        "rescue": 2.5,
        "warning issued": 3.0,
        "who should i call": 3.0,
        "affected population": 1.5,
        "respond": 1.0,
    },
    "recovery": {
        "damage": 2.5,
        "destroyed": 2.5,
        "before and after": 2.5,
        "recovery": 2.5,
        "rebuilt": 2.0,
        "post-event": 2.0,
        "aftermath": 2.0,
        "what changed": 2.0,
    },
    "risk_analyst": {
        "risk": 2.0,
        "probability": 2.0,
        "why": 2.0,
        "explain": 2.0,
        "confidence": 1.5,
        "which satellite": 2.0,
        "scene": 1.5,
        "model": 1.5,
        "compare": 1.5,
        "factor": 1.5,
        "contributed": 2.0,
    },
}

_HAZARD_TERMS = {
    "flood": ("flood", "inundat", "submerg"),
    "wildfire": ("fire", "burn", "wildfire", "smoke"),
    "cyclone": ("cyclone", "storm", "wind", "landfall"),
    "earthquake": ("earthquake", "seismic", "quake"),
    "damage": ("damage", "destroyed", "collapsed"),
}

#: Questions SAT-AI must decline, with the reason.
#:
#: **Order is load-bearing.** The first match wins, so hazard-specific rules
#: come before the generic timing rule. Without that ordering, "will there be an
#: earthquake next month" matches the flood-timing rule first and the user is
#: told about flood susceptibility -- a fluent answer to a question they did not
#: ask, omitting the only thing that matters, which is that SAT-AI performs no
#: earthquake prediction at all.
_REFUSALS: tuple[tuple[str, str], ...] = (
    (
        r"\b(should|must|do)\s+(we|i|they|residents|people).{0,20}evacuat|evacuat\w*\s*\?",
        "SAT-AI cannot advise on evacuation. It is a research prototype with no "
        "authority to issue emergency instructions. Evacuation guidance comes from "
        "your State Disaster Management Authority and local administration.",
    ),
    (
        r"\b(helpline|emergency (number|contact)|phone number|contact number)\b|"
        r"\b(who|whom|what|which)\b[^?.]{0,30}\bnumber\b[^?.]{0,20}\bcall\b|"
        r"\b(who|whom)\s+(should|do|can|would)\s+i\s+(call|contact)\b|"
        r"\bnumber\s+(should|do|can)\s+i\s+call\b",
        "SAT-AI does not hold emergency contact information and will not generate "
        "a number. Please obtain emergency contacts from official state or district "
        "disaster management sources.",
    ),
    (
        r"\bearthquake\b.{0,40}(predict|expect|forecast|when|happen|magnitude|next)|"
        r"(predict|expect|forecast|will there be).{0,40}\bearthquake\b",
        "SAT-AI performs no earthquake prediction under any circumstances. "
        "Earthquake prediction is not scientifically established. SAT-AI's "
        "earthquake scope is limited to post-event damage assessment from "
        "pre/post imagery.",
    ),
    (
        r"\b(where|when).{0,30}\bcyclone\b.{0,30}(landfall|hit|make landfall)|"
        r"\bcyclone\b.{0,30}(track|path|intensity).{0,20}(forecast|predict)",
        "SAT-AI does not forecast cyclone tracks or landfall. Track forecasting "
        "requires assimilated observations and numerical weather prediction, which "
        "IMD operates. SAT-AI assesses exposure along an observed or supplied track.",
    ),
    (
        r"\b(will|is|are) (it|there|we|they)\b.{0,40}\b(fire|wildfire|burn|ignite)\b|"
        r"\b(predict|forecast)\b.{0,30}\b(fire|wildfire|ignition)\b",
        "SAT-AI does not predict where or when a fire will start. It estimates "
        "fire danger from weather and fuel-state proxies, and ingests FIRMS "
        "detections, which are observations at satellite overpass time.",
    ),
    (
        r"\b(has|did)\s+(imd|ndma|sdma|the government)\b.{0,30}(issue|declare|warn)|"
        r"\bofficial warning\b.{0,20}(issued|in effect)",
        "SAT-AI has no access to official warning status and will not assert one. "
        "Check IMD (mausam.imd.gov.in) and your State Disaster Management "
        "Authority directly.",
    ),
    (
        r"\b(economic|monetary|financial)\s+(damage|loss|cost)|\bin (rupees|dollars|inr|usd)\b|"
        r"\bhow much.{0,20}(cost|worth|rupees)\b",
        "SAT-AI produces no monetary damage estimate. It has no asset-value data, "
        "and generating a figure without it would be fabrication.",
    ),
    (
        r"\b(will it|is it going to|when will).{0,30}(flood|rain|burn)\b|"
        r"\b(predict|forecast|expect)\b.{0,40}\b(flood|flooding)\b|"
        r"\b(can you predict|will\s+\w+\s+flood|flood tomorrow)\b|"
        r"\bnext (week|month|day|year)\b|\btomorrow\b",
        "SAT-AI does not forecast hazard timing or occurrence. For flooding it "
        "estimates susceptibility from terrain and rainfall state and maps observed "
        "extent from satellite imagery after the fact. Forecasting is IMD's role.",
    ),
)


def route(query: str) -> tuple[str, float, str | None]:
    lowered = query.lower()
    scores = {
        agent: sum(w for term, w in terms.items() if term in lowered)
        for agent, terms in _AGENT_TERMS.items()
    }
    best_agent = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    confidence = round(scores[best_agent] / total, 3) if total else 0.0
    if scores[best_agent] == 0:
        best_agent = "risk_analyst"
    hazard = next(
        (h for h, terms in _HAZARD_TERMS.items() if any(t in lowered for t in terms)), None
    )
    return best_agent, confidence, hazard


def check_refusal(query: str) -> str | None:
    for pattern, message in _REFUSALS:
        if re.search(pattern, query, re.IGNORECASE):
            return message
    return None


# --- grounding -------------------------------------------------------------
#
# Ported from satai.agents.grounding. Constants are duplicated rather than
# imported because importing the research package would pull pydantic-settings
# and numpy into a serverless bundle with a size budget. The price of that
# duplication is drift; the mechanism that pays it is
# tests/test_grounding_parity.py.

_NUMBER_RE = re.compile(
    r"""(?<![\w.])
    (?P<value>
        [-+]?
        (?:\d{1,3}(?:,\d{3})+ | \d+)
        (?:\.\d+)?
    )
    \s*(?P<suffix>%|percent|km2|km²|sq\s?km|km|m|dB|hours?|days?|hrs?)?
    (?![\w.]\d)""",
    re.VERBOSE | re.IGNORECASE,
)

#: Exempt ONLY when bare. These were once exempt unconditionally, so "flooding
#: lasted 7 days" passed the deployed check and failed the validated one -- a
#: fabricated duration reported as fact.
_EXEMPT_EXACT = frozenset({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0})
_YEAR_RANGE = (1900.0, 2100.0)

_STRUCTURAL_CONTEXT = re.compile(
    r"(sentinel[\s-]?|landsat[\s-]?|modis[\s-]?|viirs[\s-]?|noaa[\s-]?|"
    r"era|gpm|epsg:?\s?|adr[\s-]?|phase\s|step\s|band\s|figure\s|table\s|"
    r"^\s*\d+[.)]\s)",
    re.IGNORECASE,
)

_FABRICATED_AUTHORITY = (
    r"\bevacuat(e|ion)\s+(order|notice|is\s+(?:now\s+)?(?:in\s+effect|mandatory))",
    r"\bofficial\s+(warning|alert|advisory)\s+(has been|is)\s+issued",
    r"\b(call|dial|contact)\s+(?:on\s+)?(?=[-+()\s]*\d{3})[-+()\d\s]{3,}",
    r"\b(IMD|NDMA|SDMA|NDRF)\s+has\s+(issued|declared|ordered)",
    r"\byou\s+(must|should)\s+evacuate\b",
)

_CONFUSION_PATTERNS = (
    (r"\b(predict\w*|forecast\w*|will\s+occur|expected\s+to\s+(?:flood|burn))\b", "observation"),
    (r"\b(observed|detected|measured|recorded)\b", "prediction"),
)

_NEGATION_RE = re.compile(
    r"\b(not|never|isn'?t|aren'?t|no|rather\s+than|instead\s+of|"
    r"cannot|can'?t|does\s+not|doesn'?t|without)\b",
    re.IGNORECASE,
)
_SENTENCE_BREAK_RE = re.compile(r"[.!?;]\s|\n")

_CRITICAL_CAVEATS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "official_warning",
        ("not an official warning", "not official warnings", "prototype risk level"),
        ("not an official warning", "not official", "prototype", "imd", "ndma"),
    ),
    (
        "observation_not_prediction",
        ("not predictions", "not a complete census"),
        ("observation", "observed", "overpass", "not a prediction", "not predictions"),
    ),
    ("proxy_vulnerability", ("proxy",), ("proxy", "does not incorporate", "omits")),
    (
        "not_real_time",
        ("revisit-limited", "not real-time"),
        ("revisit", "not real-time", "days old", "batch"),
    ),
)

_TOLERANCE = 0.02


def _expand_allowed(allowed: list[float]) -> list[float]:
    """Values plus the safe restatements of them.

    A model writing "87 %" for a returned 0.87 is faithful; counting that as a
    violation would make the metric measure formatting rather than fidelity.
    """
    expanded: list[float] = []
    for value in allowed:
        expanded += [value, round(value, 2)]
        if 0.0 <= value <= 1.0:
            expanded += [value * 100.0, round(value * 100.0, 1)]
    return expanded


def _is_grounded_value(value: float, expanded: list[float]) -> bool:
    for permitted in expanded:
        if permitted == 0.0:
            if abs(value) < 1e-9:
                return True
            continue
        if abs(value - permitted) <= _TOLERANCE * abs(permitted):
            return True
    return False


def _is_exempt(value: float, suffix: str | None, context: str) -> bool:
    if _YEAR_RANGE[0] <= value <= _YEAR_RANGE[1] and float(value).is_integer():
        return True
    if value in _EXEMPT_EXACT and suffix is None:
        return True
    return bool(_STRUCTURAL_CONTEXT.search(context))


def _is_negated(text: str, start: int) -> bool:
    """Whether the term at ``start`` sits inside a negation.

    Scoped to the sentence: "not predictions of where a fire will occur" is an
    ordinary way to draw the distinction correctly, and the matching term sits
    at the far end of it.
    """
    breaks = [m.end() for m in _SENTENCE_BREAK_RE.finditer(text[:start])]
    return bool(_NEGATION_RE.search(text[breaks[-1] if breaks else 0 : start]))


def validate_grounding(response: str, allowed: list[float]) -> tuple[bool, list[str]]:
    """Reject any number not traceable to a tool result. Tolerance 2 %."""
    expanded = _expand_allowed(allowed)
    ungrounded: list[str] = []
    for match in _NUMBER_RE.finditer(response):
        raw = match.group("value")
        value = float(raw.replace(",", ""))
        suffix = (match.group("suffix") or "").strip().lower() or None
        context = response[max(0, match.start() - 48) : match.end() + 48]
        if _is_exempt(value, suffix, context):
            continue
        if not _is_grounded_value(value, expanded):
            ungrounded.append(raw)
    return not ungrounded, ungrounded


def validate_response(
    response: str,
    allowed: list[float],
    *,
    has_observation: bool = False,
    has_model: bool = False,
    caveats: list[str] | None = None,
) -> tuple[bool, list[tuple[str, str]]]:
    """The full check: numbers, authority, observation/prediction, caveats."""
    violations: list[tuple[str, str]] = []

    _, ungrounded = validate_grounding(response, allowed)
    violations += [
        ("ungrounded_value", f"'{raw}' is not traceable to any tool result") for raw in ungrounded
    ]

    for pattern in _FABRICATED_AUTHORITY:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            violations.append(
                (
                    "fabricated_authority",
                    f"asserts official authority the system does not have: '{match.group(0)}'",
                )
            )

    for pattern, misrepresents in _CONFUSION_PATTERNS:
        for match in re.finditer(pattern, response, re.IGNORECASE):
            if _is_negated(response, match.start()):
                continue
            if misrepresents == "observation" and has_observation and not has_model:
                violations.append(
                    (
                        "observation_prediction_confusion",
                        f"'{match.group(0)}' describes an observation as a prediction; "
                        f"the tools returned only measurements",
                    )
                )
                break
            if misrepresents == "prediction" and has_model and not has_observation:
                violations.append(
                    (
                        "observation_prediction_confusion",
                        f"'{match.group(0)}' describes a model output as an observation; "
                        f"the tools returned only model results",
                    )
                )
                break

    lowered = response.lower()
    seen: set[str] = set()
    for caveat in caveats or []:
        caveat_lower = caveat.lower()
        for name, triggers, acknowledgements in _CRITICAL_CAVEATS:
            if name in seen or not any(t in caveat_lower for t in triggers):
                continue
            seen.add(name)
            if not any(a in lowered for a in acknowledgements):
                violations.append(
                    (
                        "missing_caveat",
                        f"response omits the '{name}' caveat carried by the "
                        f"tool result: '{caveat[:70]}...'",
                    )
                )

    return not violations, violations


def is_hard_failure(violations: list[tuple[str, str]]) -> bool:
    """Fabricated authority is never regenerated -- it is returned as a refusal.

    Handing an invented helpline back with "try again" risks a second,
    differently-worded invention. There is nothing to recover.
    """
    return any(kind == "fabricated_authority" for kind, _ in violations)


# --- helpers ---------------------------------------------------------------


def _evidence(results: list[dict[str, Any]]) -> tuple[bool, bool, list[str]]:
    """What kinds of source this turn drew on, and the caveats they carry."""
    # A result may carry several kinds: a scene record is a Sentinel-1
    # acquisition (an observation) and the extent derived from it (a model
    # output). Counting only one would make the confusion check flag the
    # correct word for the other.
    kinds = {
        kind
        for r in results
        if r.get("available")
        for kind in (r.get("source_kinds") or [r.get("source_kind")])
    }
    caveats = [c for r in results for c in (r.get("caveats") or [])]
    return "observation" in kinds, "model" in kinds, caveats


def _map_actions(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map intents the model expressed, for the frontend to execute."""
    return [r["map_action"] for r in results if r.get("map_action")]


#: Why the prose layer was dropped. Three genuinely different things, and
#: reporting all three as "unavailable" is a false statement about the system:
#: in two of them the model answered perfectly well and was *rejected*.
UNAVAILABLE = "Language layer unavailable — returning verified tool output directly."
UNGROUNDED = (
    "The language layer produced an answer whose values could not all be traced "
    "back to a tool result, so it was rejected. Verified tool output follows."
)
#: Sent back to the model once when a draft fails the check for a reason it can
#: fix. The second draft is validated exactly like the first; nothing is waived.
REPAIR_PROMPT = (
    "Your previous answer failed SAT-AI's grounding check: {problems}. Rewrite it. "
    "Use only numbers that appear verbatim in the tool results above, leave out "
    "any number you cannot find there, keep the caveats, and do not call any tool."
)

#: Violations a rewrite can fix. Fabricated authority is not among them: it is
#: never regenerated (see ``is_hard_failure``).
_REPAIRABLE = frozenset({"ungrounded_value", "missing_caveat", "observation_prediction_confusion"})

OVERREACHED = (
    "The language layer asserted an authority SAT-AI does not have, so its "
    "answer was rejected. Verified tool output follows."
)


def _by_tool(results: list[dict[str, Any]], *names: str) -> dict[str, Any] | None:
    """The first available result from any of the named tools."""
    return next((r for r in results if r.get("_tool") in names and r.get("available")), None)


def _synthesis(results: list[dict[str, Any]]) -> list[str]:
    """A readable summary of this turn's tool results, built only from their values.

    This text is shown without the grounding check, because it is not model
    output. That is exactly why it may not contain a typed number: every figure
    below is read from a tool result, so it is the same figure the tool returned
    and the same one the reports hold. An earlier version typed them in and got
    five wrong.
    """
    out: list[str] = []

    model = _by_tool(results, "get_flood_model_info")
    if model:
        out.append(
            f"- **Model:** {model.get('architecture')}, "
            f"{model.get('parameters_millions')}M parameters, inputs "
            f"{', '.join(model.get('bands') or [])}; loss {model.get('loss_description')}; "
            f"{model.get('decision_rule')}."
        )

    truth = _by_tool(results, "get_flood_ground_truth", "get_ground_truth_info")
    if truth:
        dataset, split = truth["dataset"], truth["dataset"]["split"]
        out.append(
            f"- **Ground truth:** {truth['ground_truth']['name']} from {dataset['name']}: "
            f"{dataset['n_chips']} chips from {dataset['n_events']} flood events. Held-out "
            f"test region {', '.join(split['test_region'])} ({split['test_chips']} chips); "
            f"validation region {', '.join(split['validation_region'])} "
            f"({split['validation_chips']} chips)."
        )

    metrics = _by_tool(results, "get_flood_metrics")
    if metrics:
        for name, split in metrics["splits"].items():
            figures = ", ".join(
                f"{k.upper() if k == 'f1' else k.capitalize() if k != 'iou' else 'IoU'} {split[k]}"
                for k in ("iou", "f1", "precision", "recall")
                if k in split
            )
            out.append(f"- **{split['label']}** ({name}): {figures}.")

    gate = _by_tool(results, "get_distribution_gate")
    for verdict in (gate or {}).get("scene_verdicts") or []:
        bands = "; ".join(
            f"{b['band']} {b['verdict']} (Wasserstein {b['wasserstein_db']} dB)"
            for b in verdict["bands"]
        )
        out.append(
            f"- **Distribution gate, {verdict['region']}:** {verdict['verdict'].upper()}. "
            f"Trained on {verdict['training_radiometry']}; scene is "
            f"{verdict['scene_radiometry']}. Per band: {bands}."
        )

    summary = _by_tool(results, "get_flood_inference_summary")
    for item in (summary or {}).get("results") or []:
        if item.get("method") == "flood_unet":
            out.append(
                f"- **U-Net raw extent, {item['region']}:** {item.get('raw_extent_km2')} km² "
                f"— status {item.get('status')}, unvalidated, NOT confirmed flooding, "
                f"not drawn on the map."
            )
        else:
            out.append(
                f"- **{item.get('method')} extent, {item['region']}:** "
                f"{item.get('flood_extent_km2')} km² of {item.get('observed_area_km2')} km² "
                f"observed; no ground truth on this scene."
            )

    xai = _by_tool(results, "get_flood_xai", "get_xai_summary")
    if xai:
        shares = ", ".join(
            f"{band} {share}%" for band, share in xai["attribution_share_percent"].items()
        )
        out.append(
            f"- **Attribution ({' + '.join(xai['methods'])}, {xai['n_chips']} "
            f"{xai.get('region')} chips):** {shares}. Model attribution, not physical causation."
        )

    risk = _by_tool(results, "get_risk_summary")
    if risk:
        c4 = risk["c4_sensitivity"]
        out.append(
            f"- **Risk:** {risk['formulation']}. C4 sensitivity over {c4['n_settings']} "
            f"settings: rank correlation at least {c4['min_spearman_rank_correlation']}, "
            f"up to {c4['max_band_reassignment_percent']}% of cells change band. "
            f"{risk['regional_status']}"
        )

    status = _by_tool(results, "get_flood_scene_status")
    for scene in (status or {}).get("scenes") or []:
        out.append(
            f"- **Scene {scene['region']} ({scene['method']}):** {scene['status']}"
            + (f" — {scene['blocked_reason']}" if scene.get("blocked_reason") else "")
        )

    if out:
        out.append(
            "- **Not an official warning.** Official sources: IMD, CWC, NDMA and State "
            "SDMAs (India); Department of Hydrology and Meteorology (Nepal)."
        )
    return out


def _plain(value: Any) -> str:
    """A value as readable text: no Python dict or list syntax in an answer."""
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_plain(v)}" for k, v in value.items() if v is not None)
    if isinstance(value, (list, tuple)):
        return ", ".join(_plain(v) for v in value)
    return str(value)


def _asks(msg: str, *terms: str) -> bool:
    """Whether the question contains any of these words, as words.

    Substring matching sent "evaluate" to the validation answer (it contains
    "val") and "update" to the dataset answer (it contains "data").
    """
    return any(re.search(rf"\b{re.escape(term)}", msg) for term in terms)


def _natural_fallback_answer(
    query: str,
    results: list[dict[str, Any]],
    region_ids: list[str] | None = None,
) -> str:
    """A short natural-language answer when the language layer is unavailable.

    Shown without the grounding check, because no model wrote it -- which is
    exactly why it may not contain a typed number. Every figure is read from
    the exported facts or a scene tool at answer time, so it is the figure the
    reports hold. An earlier version typed them in and stated a 3.48 dB gate
    shift (the scene's shifts are 2.48 and 2.54 dB), +39.5 % over Otsu (39.3 %)
    and an Adam optimiser (AdamW).
    """
    from flood_tools import execute_flood_tool, flood_facts

    msg = (query or "").lower().strip()
    facts = flood_facts()
    if not facts:
        return _degraded_answer(query, results, region_ids)

    if results and _asks(
        msg, "raw tool", "tool output", "show evidence", "telemetry", "raw output"
    ):
        return _degraded_answer(query, results, region_ids)

    dataset, split = facts["dataset"], facts["dataset"]["split"]
    model, metrics = facts["model"], facts["metrics"]
    india, mekong, otsu = (
        metrics["india_test"],
        metrics["mekong_validation"],
        metrics["otsu_india_test"],
    )
    relative = metrics["unet_relative_gain_percent"]
    test_region = ", ".join(split["test_region"])
    val_region = ", ".join(split["validation_region"])
    totals = (
        f"{dataset['n_chips']} hand-labelled {dataset['chip_size_px']}x{dataset['chip_size_px']} "
        f"chips from {dataset['n_events']} flood events"
    )
    split_lines = (
        f"- **{split['train_chips']} chips** for training\n"
        f"- **{split['validation_chips']} {val_region} chips** for validation "
        f"(checkpoint selection)\n"
        f"- **{split['test_chips']} {test_region} chips** held out for testing\n"
        f"- **{split['reserved_chips']} {split['reserved_region']} chips** reserved"
    )

    # Order matters: the most specific subjects first, so "the 74.8 km2 Nepal
    # result" is not answered as a metrics question because it mentions a score.
    if _asks(msg, "nepal", "koshi", "74.8", "gate", "blocked", "distribution", "domain shift"):
        gate = execute_flood_tool("get_distribution_gate", {"region": "nepal_koshi_terai"})
        summary = execute_flood_tool("get_flood_inference_summary", {"region": "nepal_koshi_terai"})
        verdict = (gate.get("scene_verdicts") or [{}])[0]
        unet: dict[str, Any] = next(
            (r for r in summary.get("results") or [] if r.get("method") == "flood_unet"), {}
        )
        bands = ", ".join(
            f"{b['band']} {b['verdict']} ({b['wasserstein_db']} dB)"
            for b in verdict.get("bands") or []
        )
        year = str(summary.get("acquired_at", ""))[:4]
        return (
            f"On the Nepal Koshi scene (Sentinel-1, {year}), the U-Net "
            f"produced a raw extent of {unet.get('raw_extent_km2')} km² over "
            f"{unet.get('observed_area_km2')} km² observed. The distribution gate's verdict "
            f"was **{str(verdict.get('verdict', 'unknown')).upper()}**: the model was "
            f"trained on {verdict.get('training_radiometry')}, and this scene is "
            f"{verdict.get('scene_radiometry')}. Per-band Wasserstein distance: {bands} "
            f"(fail threshold {gate.get('fail_wasserstein_db')} dB).\n\n"
            "So the extent is **blocked and unvalidated**. It is not confirmed flooding and is "
            "not drawn on the map. This is not an official warning."
        )

    if _asks(msg, "tomorrow", "forecast", "future", "will it flood", "warning", "alert", "evacuat"):
        return (
            "SAT-AI does not forecast floods and cannot say whether any place will flood "
            "tomorrow. It maps water in archived Sentinel-1 acquisitions after the fact, and "
            "it is not real-time.\n\n"
            "It issues **no official warnings**. In India those come from IMD, the Central "
            "Water Commission (CWC), NDMA and State Disaster Management Authorities; in "
            "Nepal, from the Department of Hydrology and Meteorology."
        )

    if _asks(msg, "xai", "attribution", "explainab", "vv", "vh", "ratio", "feature", "band"):
        xai = execute_flood_tool("get_flood_xai", {})
        if not xai.get("available"):
            return "No attribution report is available for the flood model."
        shares = xai["attribution_share_percent"]
        lines = "\n".join(f"- **{band}:** {share}%" for band, share in shares.items())
        return (
            f"Integrated Gradients over {xai['n_chips']} held-out {xai['region']} chips gives "
            f"these attribution shares:\n{lines}\n\n"
            "In radar, calm open water reflects the pulse away from the satellite, so it is dark "
            "in VV and VH; the VV/VH ratio is VV dB minus VH dB. The shares say how much the "
            "**model's output** moved with each input, not why water looks the way it does. "
            "Occlusion disagrees on the ratio band's sign, and in the modality ablation adding "
            "that band to VV and VH made almost no difference to test IoU. This is model "
            "attribution, not physical causation."
        )

    if _asks(
        msg,
        "iou",
        "metric",
        "performance",
        "accura",
        "f1",
        "precision",
        "recall",
        "score",
        "evaluat",
        "otsu",
        "baseline",
        "benchmark",
        "mekong",
    ):
        return (
            f"On the held-out **{test_region} test set** ({india['n_chips']} chips, never used "
            f"for training or selection) the U-Net scores:\n"
            f"- **IoU:** {india['iou']}\n- **F1:** {india['f1']}\n"
            f"- **Precision:** {india['precision']}\n- **Recall:** {india['recall']}\n\n"
            f"The classical Otsu baseline scores IoU {otsu['iou']} on the same chips, so the "
            f"U-Net is {metrics['unet_minus_otsu_iou']} IoU better (+{relative}% relative).\n\n"
            f"{val_region} IoU {mekong['iou']} is a **validation** score used to choose the "
            f"checkpoint. It is not the {test_region} score. The per-chip median IoU is "
            f"{india['per_chip_iou_median']}: the pooled figure is lifted by chips with a lot "
            "of water."
        )

    if _asks(
        msg,
        "dataset",
        "data",
        "chip",
        "train",
        "image",
        "sample",
        "split",
        "ground truth",
        "label",
        "test",
        "validation",
    ):
        if _asks(msg, "train"):
            lead = (
                f"The model was trained on **{split['train_chips']} chips**. The full experiment "
                f"uses {totals} (Sen1Floods11 v1.1 HandLabeled), split by region:"
            )
        else:
            lead = (
                f"SAT-AI uses **Sen1Floods11 v1.1 HandLabeled**: {totals}, split by region "
                "(leave-one-region-out):"
            )
        return (
            f"{lead}\n{split_lines}\n\n"
            f"Ground truth is the dataset's hand labels ({facts['ground_truth']['name']}). "
            "Unlabelled pixels are left out of the loss and every metric."
        )

    if _asks(
        msg,
        "model",
        "architecture",
        "u-net",
        "unet",
        "parameter",
        "network",
        "pipeline",
        "predict",
        "detect",
        "loss",
        "threshold",
    ):
        bands = ", ".join(model["bands"])
        return (
            f"SAT-AI uses a **U-Net** trained from scratch: depth {model['depth']}, base width "
            f"{model['base_width']}, **{round(model['parameters'] / 1e6, 2)}M parameters**.\n"
            f"- **Inputs:** {bands} (Sentinel-1)\n"
            f"- **Loss:** {model['loss']['bce_weight']} x binary cross-entropy + "
            f"{model['loss']['dice_weight']} x Dice, on labelled pixels only\n"
            f"- **Output:** a sigmoid water probability; >= {model['threshold']} is water\n"
            f"- **Training:** {model['epochs_run']} epochs, batch size {model['batch_size']}, "
            f"AdamW; checkpoint chosen on {', '.join(model['checkpoint_selected_on'])} "
            "validation IoU"
        )

    if results:
        synthesis = [s for s in _synthesis(results) if not s.startswith("###")]
        if synthesis:
            return "\n\n".join(synthesis)

    configured = ", ".join(region_ids or FALLBACK_REGIONS)
    return (
        "SAT-AI is a student research prototype that maps flood water from Sentinel-1 radar. "
        "Ask about the U-Net, its dataset and split, its India test scores, the Nepal scene "
        f"and its distribution gate, or the attribution results. Study areas: {configured}."
    )


def _degraded_answer(
    query: str,
    results: list[dict[str, Any]],
    region_ids: list[str] | None = None,
    *,
    reason: str = UNAVAILABLE,
) -> str:
    """Structured tool output with grounded scientific synthesis."""
    lines: list[str] = [reason, ""]

    if not results:
        configured = ", ".join(region_ids or FALLBACK_REGIONS)
        lines.append(
            f"No SAT-AI tool was called for this question. Configured study areas: {configured}."
        )
        return "\n".join(lines)

    synthesis = _synthesis(results)

    if synthesis:
        lines.append("### Scientific Assessment & Key Findings")
        lines.extend(synthesis)
        lines.append("")

    lines.append("### Verified Tool Telemetry")
    for result in results:
        tool_name = result.get("_tool", "tool")
        lines.append(f"#### Tool: `{tool_name}`")
        if not result.get("available"):
            reason_msg = result.get("reason") or result.get("message", "unavailable")
            lines.append(f"- **Status:** Unavailable — {reason_msg}")
            lines.append("")
            continue
        for key, value in result.items():
            if key.startswith("_") or key in {"available", "caveats", "source_kind", "map_action"}:
                continue
            if isinstance(value, dict):
                lines.append(f"- **{key}:**")
                lines += [f"  - `{sub_k}`: {_plain(sub_v)}" for sub_k, sub_v in value.items()]
            elif isinstance(value, list) and any(isinstance(v, dict) for v in value):
                lines.append(f"- **{key}:**")
                lines += [f"  - {_plain(v)}" for v in value]
            else:
                lines.append(f"- **{key}:** {_plain(value)}")
        if result.get("caveats"):
            lines.append("- **Caveats:**")
            lines += [f"  - {c}" for c in result["caveats"]]
        lines.append("")
    return "\n".join(lines)


# --- main entry point ------------------------------------------------------


async def answer(request: ChatRequest) -> ChatResponse:
    from index import ChatResponse, Provenance, _query, study_areas

    started = time.perf_counter()
    agent, confidence, _hazard = route(request.message)

    # Before any tool call and before any model call.
    refusal = check_refusal(request.message)
    if refusal:
        return ChatResponse(
            answer=refusal,
            agent=agent,
            route_confidence=confidence,
            route_method="refusal_rule",
            tools_called=[],
            grounded=True,
            provenance=[],
            degraded=False,
            map_actions=[],
            notes=[
                "Refused by policy before any tool call: a refusal must not "
                "depend on a database lookup or an API round trip succeeding."
            ],
        )

    areas = study_areas()
    region_ids = [a["id"] for a in areas]

    if not is_configured():
        return _degraded(
            request,
            agent,
            confidence,
            [],
            region_ids,
            [],
            "GEMINI_API_KEY is not configured on this deployment; "
            "degraded to verified tool output.",
        )

    contents: list[dict[str, Any]] = [user_turn(_framed(request))]
    called: list[str] = []
    results: list[dict[str, Any]] = []
    notes: list[str] = []
    text = ""

    max_rounds = int(generation_settings()["max_tool_rounds"])
    answered_by: str | None = None
    try:
        for round_index in range(max_rounds + 1):
            allow_tools = round_index < max_rounds
            text, calls, usage = await call_gemini(
                contents,
                system_instruction=SYSTEM_PROMPT,
                tools=TOOL_DECLARATIONS if allow_tools else None,
            )
            answered_by = str(usage.get("model") or "") or answered_by
            if usage.get("fallback_from"):
                notes.append(
                    f"{usage['fallback_from']} refused the call ({usage.get('fallback_reason')}); "
                    f"answered by {usage.get('model')}."
                )
            if not calls:
                break

            contents.append(model_turn(text, calls))
            for call in calls:
                # `call` is the model's whole part, so the tool name and
                # arguments come out through accessors rather than by indexing
                # into a shape that also carries the thought signature.
                name = call_name(call)
                arguments = call_args(call)
                called.append(name)
                result = await execute_tool(name, arguments, query_fn=_query, study_areas=areas)
                result["_tool"] = name
                results.append(result)
                contents.append(tool_turn(name, result))

            if round_index == max_rounds - 1:
                notes.append(
                    f"Tool budget of {max_rounds} rounds reached; the final "
                    f"turn was generated without further tool access."
                )
    except GeminiNotConfigured as exc:
        # The exception names the defect (unset, a line break, a space) and
        # never the value; "not configured" alone hid a key that was set but
        # unusable.
        return _degraded(
            request,
            agent,
            confidence,
            results,
            region_ids,
            called,
            f"Language layer unavailable: {exc}",
        )
    except GeminiError as exc:
        return _degraded(
            request,
            agent,
            confidence,
            results,
            region_ids,
            called,
            f"{exc}; degraded to verified tool output.",
        )

    has_observation, has_model, caveats = _evidence(results)
    allowed = groundable_values(results)

    def check(draft: str) -> tuple[bool, list[tuple[str, str]]]:
        return validate_response(
            draft,
            allowed,
            has_observation=has_observation,
            has_model=has_model,
            caveats=caveats,
        )

    grounded, violations = check(text)

    if (
        not grounded
        and not is_hard_failure(violations)
        and all(kind in _REPAIRABLE for kind, _ in violations)
    ):
        # The first draft is recorded as rejected either way -- that is the C1
        # draft-violation rate -- and the rewrite has to pass the same check.
        summary = "; ".join(f"[{k}] {d}" for k, d in violations[:5])
        notes.append(f"Grounding violation in first draft, rewrite requested: {summary}")
        try:
            contents.append(model_turn(text, []))
            contents.append(user_turn(REPAIR_PROMPT.format(problems=summary)))
            rewrite, _calls, usage = await call_gemini(
                contents, system_instruction=SYSTEM_PROMPT, tools=None
            )
            answered_by = str(usage.get("model") or "") or answered_by
            rewrite_ok, rewrite_violations = check(rewrite)
            if rewrite_ok:
                text, grounded, violations = rewrite, True, []
                notes.append("Rewrite passed the grounding check.")
            else:
                violations = rewrite_violations
        except GeminiError as exc:
            notes.append(f"Rewrite not obtained ({exc}).")

    if not grounded:
        summary = "; ".join(f"[{k}] {d}" for k, d in violations[:5])
        if is_hard_failure(violations):
            notes.append(f"Hard failure, not regenerated: {summary}")
            text = _degraded_answer(request.message, results, region_ids, reason=OVERREACHED)
        else:
            notes.append(f"Grounding violation: {summary}")
            text = _degraded_answer(request.message, results, region_ids, reason=UNGROUNDED)

    provenance = [
        Provenance(
            source_kind=r.get("source_kind", "catalogue"),
            source_id=r.get("_tool", "tool"),
            version="1.0.0",
            scene_ids=r.get("scene_ids") or [],
            observed_at=r.get("observed_at"),
            caveats=r.get("caveats") or [],
        )
        for r in results
        if r.get("available")
    ]

    return ChatResponse(
        answer=text,
        agent=agent,
        route_confidence=confidence,
        route_method="gemini+tools",
        tools_called=called,
        grounded=grounded,
        provenance=provenance,
        degraded=False,
        map_actions=_map_actions(results),
        notes=[
            *notes,
            f"latency {time.perf_counter() - started:.2f}s",
            f"model {answered_by or describe_configuration()['model']}",
        ],
    )


def _framed(request: ChatRequest) -> str:
    """The user's question, plus the region they are looking at.

    The selected region is context, not an instruction: a question that names a
    different region should still be answered about that one, so it is stated
    as what is on screen rather than as the subject.
    """
    if request.region:
        return (
            f"{request.message}\n\n"
            f"(The user is currently viewing the study area '{request.region}'. "
            f"Use it only if the question does not name a different one.)"
        )
    return request.message


def _execute_degraded_tools(request: ChatRequest) -> tuple[list[dict[str, Any]], list[str]]:
    from flood_tools import execute_flood_tool

    msg = request.message.lower()
    reg = (request.region or "bihar_ganga").lower()
    for known in [
        "bihar_ganga",
        "mumbai_mmr",
        "assam_brahmaputra",
        "kerala_periyar",
        "nepal_koshi_terai",
        "odisha_mahanadi",
        "uttarakhand_kumaon",
    ]:
        if known in msg:
            reg = known
            break

    results: list[dict[str, Any]] = []
    called: list[str] = []

    # 1. Nepal Koshi / 74.8 / gate
    if any(k in msg for k in ["74.8", "nepal", "koshi", "gate"]):
        r1 = execute_flood_tool("get_distribution_gate", {"region": "nepal_koshi_terai"})
        r1["_tool"] = "get_distribution_gate"
        results.append(r1)
        called.append("get_distribution_gate")
        r2 = execute_flood_tool("get_flood_inference_summary", {"scene_id": "nepal_koshi_202008"})
        r2["_tool"] = "get_flood_inference_summary"
        results.append(r2)
        called.append("get_flood_inference_summary")

    # 2. XAI / attribution / VV / VH / ratio
    elif any(k in msg for k in ["xai", "attribution", "attributions", "vv", "vh", "ratio"]):
        r = execute_flood_tool("get_xai_summary", {"model_id": "unet_sen1floods11"})
        r["_tool"] = "get_xai_summary"
        results.append(r)
        called.append("get_xai_summary")

    # 3. Model detection / ground truth / dataset / data / training / chips / images
    elif any(
        k in msg
        for k in [
            "model",
            "detection",
            "ground truth",
            "dataset",
            "data",
            "architecture",
            "train",
            "chip",
            "image",
        ]
    ):
        r1 = execute_flood_tool("get_flood_model_info", {"model_id": "unet_sen1floods11"})
        r1["_tool"] = "get_flood_model_info"
        results.append(r1)
        called.append("get_flood_model_info")
        r2 = execute_flood_tool("get_ground_truth_info", {"dataset": "sen1floods11"})
        r2["_tool"] = "get_ground_truth_info"
        results.append(r2)
        called.append("get_ground_truth_info")
        r3 = execute_flood_tool("get_flood_metrics", {"split": "india_test"})
        r3["_tool"] = "get_flood_metrics"
        results.append(r3)
        called.append("get_flood_metrics")

    # 4. India / IoU / metrics / performance / test
    elif any(
        k in msg
        for k in [
            "india",
            "iou",
            "metric",
            "metrics",
            "test",
            "testing",
            "performance",
            "accuracy",
            "otsu",
            "baseline",
        ]
    ):
        r = execute_flood_tool("get_flood_metrics", {"split": "india_test"})
        r["_tool"] = "get_flood_metrics"
        results.append(r)
        called.append("get_flood_metrics")
        r2 = execute_flood_tool("get_ground_truth_info", {"dataset": "sen1floods11"})
        r2["_tool"] = "get_ground_truth_info"
        results.append(r2)
        called.append("get_ground_truth_info")

    # 5. Risk / current risk / flood risk / bihar / general region risk
    elif any(k in msg for k in ["risk", "flood", "current", "hazard", "bihar"]):
        r1 = execute_flood_tool("get_risk_summary", {"region": reg, "hazard": "flood"})
        r1["_tool"] = "get_risk_summary"
        results.append(r1)
        called.append("get_risk_summary")
        r2 = execute_flood_tool("get_flood_scene_status", {"region": reg})
        r2["_tool"] = "get_flood_scene_status"
        results.append(r2)
        called.append("get_flood_scene_status")

    # 6. Provenance
    elif "provenance" in msg:
        r = execute_flood_tool("get_provenance", {})
        r["_tool"] = "get_provenance"
        results.append(r)
        called.append("get_provenance")

    return results, called


def _degraded(
    request: ChatRequest,
    agent: str,
    confidence: float,
    results: list[dict[str, Any]],
    region_ids: list[str],
    called: list[str],
    note: str,
) -> ChatResponse:
    from index import ChatResponse, Provenance

    if not results:
        results, _ = _execute_degraded_tools(request)

    answer_text = _natural_fallback_answer(request.message, results, region_ids)

    provenance = [
        Provenance(
            source_kind=r.get("source_kind", "catalogue"),
            source_id=r.get("_tool", "tool"),
            version="1.0.0",
            scene_ids=r.get("scene_ids") or [],
            observed_at=r.get("observed_at"),
            caveats=r.get("caveats") or [],
        )
        for r in results
        if r.get("available")
    ]

    return ChatResponse(
        answer=answer_text,
        agent=agent,
        route_confidence=confidence,
        route_method="degraded+tools" if called else "degraded",
        tools_called=called,
        grounded=True,
        provenance=provenance,
        degraded=True,
        map_actions=_map_actions(results),
        notes=[note],
    )


__all__ = [
    "MAX_TOOL_ROUNDS",
    "answer",
    "check_refusal",
    "is_hard_failure",
    "route",
    "validate_grounding",
    "validate_response",
]

#: Re-exported so /health can report the language layer without importing the
#: gemini module directly.
describe_llm = describe_configuration
