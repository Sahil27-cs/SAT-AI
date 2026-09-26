"""Agent plane for the deployed serving layer.

A deliberately small, dependency-light mirror of ``satai.agents`` so the
serverless bundle stays inside its size budget. The behavioural contract is the
same one the research code enforces and the benchmark measures:

* values come from tools, never from the model;
* observations and model outputs stay distinct;
* the system refuses what it has no authority or capability to answer;
* when the LLM is unreachable the endpoint degrades to structured tool output
  and says so, rather than failing or inventing prose (requirement 39).

The heuristic router runs unconditionally. Routing with the LLM is an
optimisation; routing *without* it is what keeps the interface usable when the
agent plane is down.

**On the grounding check, and why it is no longer a weaker one.** This module
used to validate only that numbers were traceable, while
``satai.agents.grounding`` -- the validator scored in experiment 9a and reported
as the C1 instrument -- also checked fabricated authority, observation/prediction
confusion and dropped critical caveats. So the measurement described in the
paper was not the measurement running in production, and the deployed check was
the weaker of the two in exactly the categories that matter most: an invented
helpline is worse than an invented number. The four checks are now ported here
in full, with the same constants and the same semantics, and
``tests/test_grounding_parity.py`` runs both implementations over a shared
corpus and fails if their verdicts diverge. That test is the thing keeping this
file honest; porting without it would just restart the drift.
"""

from __future__ import annotations

import os
import re
import time
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:  # pragma: no cover
    from index import ChatRequest, ChatResponse

DISCLAIMER = (
    "SAT-AI prototype risk level. Research output from a student research "
    "prototype. This is NOT an official warning and does not replace IMD, NDMA "
    "or State Disaster Management Authority advisories."
)

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

#: Fallback study-area list for the degraded answer, used only when the
#: catalogue lookup that would supply the real ones has itself failed.
FALLBACK_REGIONS = ("bihar_ganga", "mumbai_mmr", "assam_brahmaputra", "kerala_periyar")

SYSTEM_PROMPT = """\
You are an analyst interface to SAT-AI, a student research prototype that
assesses multi-hazard risk from satellite remote sensing over a small number of
configured study areas in India.

ABSOLUTE RULES
1. Every project-specific number must come from the TOOL RESULTS given to you.
   If a quantity is not there, say it is not available. Never estimate it.
2. Keep observations and model outputs distinct. FIRMS detections are
   MEASUREMENTS at overpass time. Flood extent is a MODEL OUTPUT. A risk index
   is a composite INDEX. None of them is a forecast.
3. SAT-AI does not forecast: no earthquake prediction, no cyclone track, no
   wildfire ignition, no flood timing or depth. Say so if asked.
4. Risk levels are prototype research outputs, NOT official warnings. Say this
   whenever you report one. Official warnings come from IMD, NDMA and State
   Disaster Management Authorities.
5. Never invent an emergency phone number, an evacuation order, an official
   advisory, or the status of any government warning.
6. SAT-AI is not real-time. Sentinel-1 revisit is 6-12 days. Report data age.
7. If a region is not covered, say so rather than estimating from general
   knowledge.
8. Carry through the caveats attached to tool results. They are part of the
   answer, not optional decoration.
Be concise. State what the data shows, what it does not, and how confident the
model is."""

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

#: Questions SAT-AI must decline, with the reason. Checked before any tool call:
#: a refusal should not depend on a database lookup succeeding.
#:
#: **Order is load-bearing.** The first match wins, so hazard-specific rules come
#: before the generic timing rule. Without that ordering, "will there be an
#: earthquake next month" matches the flood-timing rule first and the user is
#: told about flood susceptibility — a fluent answer to a question they did not
#: ask, and one that omits the only thing that matters, which is that SAT-AI
#: performs no earthquake prediction at all.
_REFUSALS: tuple[tuple[str, str], ...] = (
    (
        r"\b(should|must|do)\s+(we|i|they|residents|people).{0,20}evacuat|evacuat\w*\s*\?",
        "SAT-AI cannot advise on evacuation. It is a research prototype with no "
        "authority to issue emergency instructions. Evacuation guidance comes from "
        "your State Disaster Management Authority and local administration.",
    ),
    (
        # Indian emergency numbers are short and are asked for in many ways.
        # "what number should I call" is the phrasing most likely to draw an
        # invented helpline out of a language model, so it is matched explicitly.
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
        # Generic timing, checked last so a hazard-specific rule above can claim
        # the query first and give the reason that actually applies to it.
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
# Ported from satai.agents.grounding. The constants below are duplicated rather
# than imported because importing the research package would pull pydantic-
# settings and numpy into a serverless bundle that has a size budget. The price
# of that duplication is drift, and the mechanism that pays it is
# tests/test_grounding_parity.py, which asserts both implementations agree.

#: Numbers with optional thousands separators, decimals, percentages and signs.
#: The suffix group is what makes the exemption table safe: "7" on its own is
#: rhetorical, "7 days" is a claim about the world.
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

#: Small integers used rhetorically ("three factors") and 100 for percentages.
#: Exempt ONLY when bare. Previously these were exempt unconditionally, so
#: "flooding lasted 7 days" passed the deployed check and failed the validated
#: one -- a fabricated duration reported as fact.
_EXEMPT_EXACT = frozenset({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0})
_YEAR_RANGE = (1900.0, 2100.0)

#: Contexts in which a number is structural rather than a claim about the world.
_STRUCTURAL_CONTEXT = re.compile(
    r"(sentinel[\s-]?|landsat[\s-]?|modis[\s-]?|viirs[\s-]?|noaa[\s-]?|"
    r"era|gpm|epsg:?\s?|adr[\s-]?|phase\s|step\s|band\s|figure\s|table\s|"
    r"^\s*\d+[.)]\s)",
    re.IGNORECASE,
)

#: Phrases that assert official status. The system is a research prototype; it
#: has no authority to issue any of these, so they are hard failures and are
#: never regenerated.
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

#: Critical caveat classes: the trigger that identifies one in a tool result,
#: and the phrases that count as carrying it through into the answer.
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

    Scoped to the sentence, not a fixed window: "not predictions of where a fire
    will occur" is an ordinary way to draw the distinction correctly, and the
    term that matches is at the far end of it.
    """
    breaks = [m.end() for m in _SENTENCE_BREAK_RE.finditer(text[:start])]
    return bool(_NEGATION_RE.search(text[breaks[-1] if breaks else 0 : start]))


def validate_grounding(response: str, allowed: list[float]) -> tuple[bool, list[str]]:
    """Reject any number not traceable to a tool result. Tolerance 2 %.

    Returns ``(grounded, ungrounded_raw_values)``. This is the numeric check
    only; :func:`validate_response` adds the three categorical checks.
    """
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
    """The full check: numbers, authority, observation/prediction, caveats.

    Returns ``(grounded, [(kind, detail), ...])`` using the same violation-kind
    vocabulary as :class:`satai.agents.grounding.ViolationKind`, so the two
    implementations' reports are directly comparable.
    """
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
    """Fabricated authority is never regenerated — it is returned as a refusal.

    Handing an invented helpline back to the model with "try again" risks a
    second, differently-worded invention. There is no value to recover here, so
    the turn falls straight through to verified tool output.
    """
    return any(kind == "fabricated_authority" for kind, _ in violations)


# --- tools -----------------------------------------------------------------


def _tool_name(hazard: str) -> str:
    """The tool a hazard query maps to.

    Previously the name appended to ``tools_called`` was computed from the
    hazard while the result payload hardcoded ``get_flood_prediction``, so a
    wildfire query reported one tool in its trace and a different one in its
    provenance. Tool-invocation accuracy is a C1 metric, so a trace that does
    not match what ran corrupts the measurement, not just the display.
    """
    return "get_flood_prediction" if hazard == "flood" else f"get_{hazard}_prediction"


async def fetch_tools(
    query_fn: Any, region: str | None, hazard: str | None
) -> tuple[list[str], list[dict[str, Any]], list[float]]:
    """Call the tools this query needs. Returns (names, results, allowed values)."""
    called: list[str] = []
    results: list[dict[str, Any]] = []
    allowed: list[float] = []

    if not region:
        return called, results, allowed

    regions = await query_fn("regions", {"select": "*", "id": f"eq.{region}"})
    if not regions:
        called.append("get_location_statistics")
        results.append(
            {
                "tool": "get_location_statistics",
                "available": False,
                "message": (
                    f"{region} is not a SAT-AI study area. The system covers only "
                    f"its configured regions and does not estimate risk elsewhere."
                ),
            }
        )
        return called, results, allowed

    info = regions[0]
    called.append("get_location_statistics")
    results.append(
        {
            "tool": "get_location_statistics",
            "available": True,
            "region": info["name"],
            "area_km2": info["area_km2"],
            "study_role": info["study_role"],
            "source_kind": "catalogue",
            "caveats": info.get("caveats") or [],
        }
    )
    allowed.append(float(info["area_km2"]))

    target = hazard or "flood"
    tool_name = _tool_name(target)
    rows = await query_fn(
        "latest_hazard_results",
        {"select": "*", "region_id": f"eq.{region}", "hazard": f"eq.{target}"},
    )
    called.append(tool_name)
    if rows:
        row = rows[0]
        results.append(
            {
                "tool": tool_name,
                "available": True,
                "hazard": target,
                "risk_index": row["risk_index"],
                "risk_band": row["risk_band"],
                "confidence": row.get("confidence"),
                "flooded_area_km2": row.get("flooded_area_km2"),
                "population_exposed": row.get("population_exposed"),
                "source_kind": row.get("source_kind", "model"),
                "scene_ids": row.get("scene_ids") or [],
                "observed_at": row.get("observed_at"),
                "caveats": [DISCLAIMER, *(row.get("caveats") or [])],
            }
        )
        allowed += [
            float(v)
            for v in (
                row["risk_index"],
                row.get("confidence"),
                row.get("flooded_area_km2"),
                row.get("population_exposed"),
            )
            if v is not None
        ]
    else:
        results.append(
            {
                "tool": tool_name,
                "available": False,
                "hazard": target,
                "message": (
                    f"No {target} analysis has been computed for {info['name']}. "
                    f"SAT-AI produces results on a batch schedule tied to satellite "
                    f"revisit; this region has no completed run."
                ),
            }
        )
    return called, results, allowed


def _evidence(results: list[dict[str, Any]]) -> tuple[bool, bool, list[str]]:
    """What kinds of source this turn drew on, and the caveats they carry."""
    kinds = {r.get("source_kind") for r in results if r.get("available")}
    caveats = [c for r in results for c in (r.get("caveats") or [])]
    return "observation" in kinds, "model" in kinds, caveats


# --- main entry point ------------------------------------------------------


async def answer(request: ChatRequest) -> ChatResponse:
    from index import ChatResponse, Provenance, _query

    started = time.perf_counter()
    agent, confidence, hazard = route(request.message)

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
            notes=[
                "Refused by policy before any tool call: a refusal must not "
                "depend on a database lookup succeeding."
            ],
        )

    region = request.region
    known = await _query("regions", {"select": "id,name"})
    if not region:
        for row in known:
            if row["id"].lower() in request.message.lower() or (
                row["name"].split(",")[0].lower() in request.message.lower()
            ):
                region = row["id"]
                break

    called, tool_results, allowed = await fetch_tools(_query, region, hazard)
    region_ids = [str(r["id"]) for r in known] or list(FALLBACK_REGIONS)

    provenance = [
        Provenance(
            source_kind=r.get("source_kind", "catalogue"),
            source_id=r["tool"],
            version="1.0.0",
            scene_ids=r.get("scene_ids") or [],
            observed_at=r.get("observed_at"),
            caveats=r.get("caveats") or [],
        )
        for r in tool_results
        if r.get("available")
    ]

    if not ANTHROPIC_KEY:
        return ChatResponse(
            answer=_degraded_answer(request.message, region, tool_results, region_ids),
            agent=agent,
            route_confidence=confidence,
            route_method="heuristic",
            tools_called=called,
            grounded=True,
            provenance=provenance,
            degraded=True,
            notes=[
                "No ANTHROPIC_API_KEY is configured on this deployment, so the "
                "language layer is unavailable. This response is structured tool "
                "output. The data plane is unaffected (requirement 39).",
            ],
        )

    text, grounded, notes = await _generate(request.message, tool_results, allowed, region_ids)
    return ChatResponse(
        answer=text,
        agent=agent,
        route_confidence=confidence,
        route_method="heuristic+llm",
        tools_called=called,
        grounded=grounded,
        provenance=provenance,
        degraded=False,
        notes=[*notes, f"latency {time.perf_counter() - started:.2f}s"],
    )


def _degraded_answer(
    query: str,
    region: str | None,
    results: list[dict[str, Any]],
    region_ids: list[str] | None = None,
) -> str:
    """Structured tool output, with no generated prose layered over it."""
    lines = ["Language layer unavailable — returning verified tool output directly.", ""]
    if not region:
        configured = ", ".join(region_ids or FALLBACK_REGIONS)
        lines.append(
            f"No SAT-AI study area was identified in your question. Configured "
            f"regions: {configured}."
        )
        return "\n".join(lines)

    for result in results:
        lines.append(f"[{result['tool']}]")
        if not result.get("available"):
            lines += [f"  {result['message']}", ""]
            continue
        for key, value in result.items():
            if key in {"tool", "available", "caveats", "source_kind"}:
                continue
            lines.append(f"  {key}: {value}")
        if result.get("caveats"):
            lines.append("  caveats:")
            lines += [f"    - {c}" for c in result["caveats"]]
        lines.append("")
    return "\n".join(lines)


async def _generate(
    query: str,
    results: list[dict[str, Any]],
    allowed: list[float],
    region_ids: list[str] | None = None,
) -> tuple[str, bool, list[str]]:
    """Generate, validate, regenerate once on a recoverable violation."""
    import json

    has_observation, has_model, caveats = _evidence(results)
    prompt = (
        f"User question: {query}\n\n"
        f"TOOL RESULTS (the only source of project-specific values):\n"
        f"{json.dumps(results, indent=2, default=str)}\n\n"
        f"Answer using only these values. If a quantity is not present, say it "
        f"is unavailable. Carry through the caveats attached to the results."
    )
    notes: list[str] = []

    def _fallback(reason: str) -> tuple[str, bool, list[str]]:
        return _degraded_answer(query, "unknown", results, region_ids), True, [reason]

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": ANTHROPIC_MODEL,
                        "max_tokens": 900,
                        "temperature": 0.2,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
            if response.status_code >= 400:
                return _fallback(f"LLM returned {response.status_code}; degraded to tool output.")
            text = "".join(block.get("text", "") for block in response.json().get("content", []))
        except (httpx.HTTPError, TimeoutError) as exc:
            return _fallback(f"LLM unreachable ({type(exc).__name__}); degraded to tool output.")

        grounded, violations = validate_response(
            text,
            allowed,
            has_observation=has_observation,
            has_model=has_model,
            caveats=caveats,
        )
        if grounded:
            if attempt:
                notes.append("Regenerated once after a grounding violation.")
            return text, True, notes

        if is_hard_failure(violations):
            # Never regenerated. See is_hard_failure.
            detail = next(d for k, d in violations if k == "fabricated_authority")
            return (
                _degraded_answer(query, "unknown", results, region_ids)
                + "\n\n[The language layer asserted authority SAT-AI does not have; "
                "returning verified tool output instead.]",
                False,
                [*notes, f"Hard failure, not regenerated: fabricated authority — {detail}"],
            )

        if attempt == 0:
            summary = "; ".join(f"[{k}] {d}" for k, d in violations[:5])
            notes.append(f"Grounding violation on first attempt: {summary}")
            permitted = ", ".join(f"{v:g}" for v in sorted(set(allowed))[:30])
            prompt += (
                f"\n\nYour previous answer had these problems: {summary}. "
                f"Permitted values: {permitted}. Rewrite using only those values, "
                f"carry through the caveats, and if the tools did not return a "
                f"quantity the question asks for, say so plainly."
            )

    return (
        _degraded_answer(query, "unknown", results, region_ids)
        + "\n\n[The language layer produced ungrounded values twice; "
        "returning verified tool output instead.]",
        False,
        [*notes, "Two grounding violations; fell back to structured output."],
    )
