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
#: without letting a confused turn loop until the function times out.
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
You are the analyst interface to SAT-AI, a student research prototype that
assesses multi-hazard risk from satellite remote sensing over configured study
areas in India and Nepal.

You have tools. Use them. Never answer a question about SAT-AI's data, models
or regions from your own knowledge -- call the tool and answer from what it
returns.

ABSOLUTE RULES
1. Every project-specific number must come from a tool result in this turn. If
   a tool did not return a quantity, say plainly that it is not available.
   Never estimate, interpolate, or recall a number from general knowledge.
2. When a tool returns available=false, report that. Say DATA UNAVAILABLE or
   MODEL RESULT NOT COMPUTED and give the reason the tool gave. Do not
   substitute a plausible value and do not apologise at length.
3. Keep observations and model outputs distinct. A FIRMS thermal anomaly is a
   MEASUREMENT at overpass time. A flood extent is a MODEL OUTPUT. A risk index
   is a documented COMPOSITE INDEX. None of them is a forecast.
4. SAT-AI does not forecast: no earthquake prediction, no cyclone track, no
   wildfire ignition, no flood timing or depth. Say so if asked.
5. Risk levels are prototype research outputs, NOT official warnings. Say this
   whenever you report one. Official warnings come from IMD, NDMA and State
   Disaster Management Authorities.
6. Never invent an emergency phone number, an evacuation order, an official
   advisory, or the status of any government warning.
7. SAT-AI is not real-time. Sentinel-1 revisit is 6-12 days. Report data age.
8. Carry through the caveats attached to tool results. They are part of the
   answer, not optional decoration.
9. When the user asks to see, show or zoom to something, call show_on_map. Do
   not tell them which button to press.

Be concise. State what the data shows, what it does not, and how confident the
model is. Uncertainty is information, not a weakness."""

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
        r"\bnext (week|month|day|year)\b",
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
    kinds = {r.get("source_kind") for r in results if r.get("available")}
    caveats = [c for r in results for c in (r.get("caveats") or [])]
    return "observation" in kinds, "model" in kinds, caveats


def _map_actions(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map intents the model expressed, for the frontend to execute."""
    return [r["map_action"] for r in results if r.get("map_action")]


def _degraded_answer(
    query: str, results: list[dict[str, Any]], region_ids: list[str] | None = None
) -> str:
    """Structured tool output, with no generated prose layered over it."""
    lines = ["Language layer unavailable — returning verified tool output directly.", ""]
    if not results:
        configured = ", ".join(region_ids or FALLBACK_REGIONS)
        lines.append(
            f"No SAT-AI tool was called for this question. Configured study areas: {configured}."
        )
        return "\n".join(lines)

    for result in results:
        lines.append(f"[{result.get('_tool', 'tool')}]")
        if not result.get("available"):
            lines += [f"  {result.get('reason') or result.get('message', 'unavailable')}", ""]
            continue
        for key, value in result.items():
            if key.startswith("_") or key in {"available", "caveats", "source_kind", "map_action"}:
                continue
            lines.append(f"  {key}: {value}")
        if result.get("caveats"):
            lines.append("  caveats:")
            lines += [f"    - {c}" for c in result["caveats"]]
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
        return ChatResponse(
            answer=_degraded_answer(request.message, [], region_ids),
            agent=agent,
            route_confidence=confidence,
            route_method="heuristic",
            tools_called=[],
            grounded=True,
            provenance=[],
            degraded=True,
            map_actions=[],
            notes=[
                "GEMINI_API_KEY is not configured on this deployment, so the "
                "language layer is unavailable. The data plane is unaffected "
                "(requirement 39)."
            ],
        )

    contents: list[dict[str, Any]] = [user_turn(_framed(request))]
    called: list[str] = []
    results: list[dict[str, Any]] = []
    notes: list[str] = []
    text = ""

    try:
        for round_index in range(MAX_TOOL_ROUNDS + 1):
            allow_tools = round_index < MAX_TOOL_ROUNDS
            text, calls, _usage = await call_gemini(
                contents,
                system_instruction=SYSTEM_PROMPT,
                tools=TOOL_DECLARATIONS if allow_tools else None,
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

            if round_index == MAX_TOOL_ROUNDS - 1:
                notes.append(
                    f"Tool budget of {MAX_TOOL_ROUNDS} rounds reached; the final "
                    f"turn was generated without further tool access."
                )
    except GeminiNotConfigured:
        return _degraded(
            request,
            agent,
            confidence,
            results,
            region_ids,
            called,
            "GEMINI_API_KEY is not configured.",
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
    grounded, violations = validate_response(
        text,
        allowed,
        has_observation=has_observation,
        has_model=has_model,
        caveats=caveats,
    )

    if not grounded:
        summary = "; ".join(f"[{k}] {d}" for k, d in violations[:5])
        if is_hard_failure(violations):
            notes.append(f"Hard failure, not regenerated: {summary}")
            text = _degraded_answer(request.message, results, region_ids) + (
                "\n\n[The language layer asserted authority SAT-AI does not have; "
                "returning verified tool output instead.]"
            )
        else:
            notes.append(f"Grounding violation: {summary}")
            text = _degraded_answer(request.message, results, region_ids) + (
                "\n\n[The language layer stated values not traceable to a tool "
                "result; returning verified tool output instead.]"
            )

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
            f"model {describe_configuration()['model']}",
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


def _degraded(
    request: ChatRequest,
    agent: str,
    confidence: float,
    results: list[dict[str, Any]],
    region_ids: list[str],
    called: list[str],
    note: str,
) -> ChatResponse:
    from index import ChatResponse

    return ChatResponse(
        answer=_degraded_answer(request.message, results, region_ids),
        agent=agent,
        route_confidence=confidence,
        route_method="degraded",
        tools_called=called,
        grounded=True,
        provenance=[],
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
