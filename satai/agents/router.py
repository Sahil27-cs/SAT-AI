"""Query router and the three specialised agents.

Routing is a short classification -- which agent, which region, which hazard --
and paying frontier-model prices for it wastes most of the cost on the easy half
of the problem while adding latency to the step the user feels first. So the
router uses the small model (ADR-005), with a deterministic keyword fallback
that keeps the system usable when no LLM is reachable at all.

That fallback is not a nicety. Requirement 39 says the ML plane must keep
working when the agent plane is down; a router that cannot route without an API
call would make the whole system depend on it.

The three agents differ in their tools and their refusals, not in their
model. Each carries an explicit list of things it must never assert, and those
refusals are what the benchmark's ``must_refuse`` items measure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from satai.agents.tools import TOOL_REGISTRY
from satai.logging import get_logger

log = get_logger(__name__)

__all__ = ["AGENTS", "AgentKind", "AgentSpec", "RouteDecision", "route_heuristic"]


class AgentKind(StrEnum):
    RISK_ANALYST = "risk_analyst"
    EMERGENCY = "emergency"
    RECOVERY = "recovery"


#: Shared preamble. Every rule here exists because breaking it would produce a
#: confident, plausible, wrong answer about a hazard -- the failure mode the
#: whole provenance contract is built to prevent.
SYSTEM_PREAMBLE = """\
You are an analyst interface to SAT-AI, a student research prototype that
assesses multi-hazard risk from satellite remote sensing over a small number of
configured study areas in India.

ABSOLUTE RULES

1. Every project-specific number you state must come from a tool result in this
   turn. If a tool did not return a quantity, say plainly that it is not
   available. Never estimate, interpolate, or recall a number from general
   knowledge.
2. Keep observations and model outputs distinct. A NASA FIRMS thermal anomaly is
   a MEASUREMENT at satellite overpass time. A flood extent is a MODEL OUTPUT
   derived from imagery. A risk index is a documented COMPOSITE INDEX. Never
   describe any of them as a forecast.
3. SAT-AI does not forecast. It does not predict earthquakes, cyclone tracks,
   wildfire ignition, or flood timing and depth. If asked, say so.
4. SAT-AI risk levels (GREEN/YELLOW/ORANGE/RED) are prototype research outputs.
   They are NOT official warnings. Official warnings for India come from IMD,
   NDMA and State Disaster Management Authorities. Say this whenever you report
   a risk level.
5. Never invent an emergency telephone number, an evacuation order, an official
   advisory, or the status of any government warning. You have no access to
   these and no authority to issue them.
6. SAT-AI is not real-time. Sentinel-1 revisit is 6-12 days. When you report a
   satellite-derived result, report how old the observation is.
7. If a question concerns a region SAT-AI does not cover, say so. Do not
   estimate from general knowledge about that place.
8. Carry through the caveats attached to tool results. They are part of the
   answer, not optional decoration.

Be concise and direct. State what the data shows, what it does not show, and
how confident the model is. Uncertainty is information, not a weakness.
"""


@dataclass(frozen=True)
class AgentSpec:
    """One specialised agent."""

    kind: AgentKind
    title: str
    purpose: str
    role_prompt: str
    refusals: tuple[str, ...] = ()

    @property
    def tools(self) -> list[str]:
        return sorted(
            name for name, spec in TOOL_REGISTRY.items() if self.kind.value in spec.agents
        )

    def system_prompt(self) -> str:
        parts = [SYSTEM_PREAMBLE, f"\nYOUR ROLE: {self.title}\n{self.role_prompt}"]
        if self.refusals:
            parts.append(
                "\nYOU MUST REFUSE, every time, without exception:\n"
                + "\n".join(f"  - {r}" for r in self.refusals)
            )
        parts.append(f"\nTOOLS AVAILABLE TO YOU: {', '.join(self.tools)}")
        return "\n".join(parts)


AGENTS: dict[AgentKind, AgentSpec] = {
    AgentKind.RISK_ANALYST: AgentSpec(
        kind=AgentKind.RISK_ANALYST,
        title="Risk Analyst",
        purpose="Current risk, model outputs, satellite provenance, explanations, comparisons.",
        role_prompt=(
            "You interpret SAT-AI's model outputs and explain what drove them. "
            "When you explain a risk score, use the actual feature attributions "
            "the explanation tool returns -- never a plausible-sounding narrative "
            "assembled from what usually causes floods. Attribution describes what "
            "the model used; it is not a claim about physical causation, and you "
            "should say so when the distinction matters."
        ),
        refusals=(
            "Forecasting when or whether a hazard will occur.",
            "Stating a number no tool returned.",
            "Describing a model output as a direct measurement.",
        ),
    ),
    AgentKind.EMERGENCY: AgentSpec(
        kind=AgentKind.EMERGENCY,
        title="Emergency Response Analyst",
        purpose="Affected areas, exposure, and pointing to authoritative sources.",
        role_prompt=(
            "You summarise which areas SAT-AI's analysis indicates are affected and "
            "what exposure they carry. You are an information interface, not an "
            "emergency service: you have no authority to instruct anyone to do "
            "anything, and no access to official warning status. When a question "
            "calls for an operational decision, say that SAT-AI cannot provide it "
            "and that official guidance comes from IMD, NDMA and the relevant State "
            "Disaster Management Authority -- without inventing contact details for "
            "any of them."
        ),
        refusals=(
            "Issuing or implying an evacuation instruction.",
            "Providing any emergency telephone number.",
            "Asserting that an official warning has or has not been issued.",
            "Claiming SAT-AI output is an official alert.",
        ),
    ),
    AgentKind.RECOVERY: AgentSpec(
        kind=AgentKind.RECOVERY,
        title="Recovery and Impact Analyst",
        purpose="Post-event damage, before/after change, land-cover change, recovery trends.",
        role_prompt=(
            "You analyse post-event change: what the imagery shows changed between "
            "a pre-event and a post-event acquisition, how severe the change is, and "
            "how recovery has progressed. Change detected from imagery has not been "
            "verified on the ground, and you should say so. SAT-AI produces no "
            "monetary damage estimate because it has no asset-value data."
        ),
        refusals=(
            "Estimating monetary damage.",
            "Estimating casualties.",
            "Presenting satellite-detected change as ground-verified.",
        ),
    ),
}


@dataclass(frozen=True)
class RouteDecision:
    """Where a query goes, and what was extracted from it."""

    agent: AgentKind
    region: str | None = None
    hazard: str | None = None
    timeframe_days: int | None = None
    confidence: float = 0.0
    method: str = "heuristic"
    rationale: str = ""
    matched_terms: tuple[str, ...] = field(default_factory=tuple)


#: Routing vocabulary. Weighted because some terms are decisive and others only
#: suggestive: "evacuate" is a strong signal for the emergency agent, whereas
#: "area" appears in almost every query.
_AGENT_TERMS: dict[AgentKind, dict[str, float]] = {
    AgentKind.EMERGENCY: {
        "evacuat": 3.0,
        "emergency": 2.5,
        "helpline": 3.0,
        "shelter": 2.5,
        "rescue": 2.5,
        "warning issued": 3.0,
        "who should i call": 3.0,
        "affected population": 1.5,
        "at risk population": 1.5,
        "respond": 1.0,
    },
    AgentKind.RECOVERY: {
        "damage": 2.5,
        "destroyed": 2.5,
        "before and after": 2.5,
        "recovery": 2.5,
        "rebuilt": 2.0,
        "post-event": 2.0,
        "post event": 2.0,
        "aftermath": 2.0,
        "what changed": 2.0,
        "since the": 1.0,
        "land cover change": 2.0,
    },
    AgentKind.RISK_ANALYST: {
        "risk": 2.0,
        "probability": 2.0,
        "why": 2.0,
        "explain": 2.0,
        "confidence": 1.5,
        "which satellite": 2.0,
        "scene": 1.5,
        "model": 1.5,
        "compare": 1.5,
        "forecast": 1.0,
        "factor": 1.5,
        "contributed": 2.0,
    },
}

_HAZARD_TERMS: dict[str, tuple[str, ...]] = {
    "flood": ("flood", "inundat", "water extent", "submerg"),
    "wildfire": ("fire", "burn", "wildfire", "blaze", "smoke"),
    "cyclone": ("cyclone", "storm", "wind", "landfall", "hurricane", "typhoon"),
    "earthquake": ("earthquake", "seismic", "quake", "tremor"),
    "damage": ("damage", "destroyed", "collapsed", "structural"),
}

_TIMEFRAME_RE = re.compile(
    r"(?:last|past|previous|over)\s+(\d+)\s*(day|week|month|year)s?", re.IGNORECASE
)
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


def route_heuristic(query: str, known_regions: dict[str, list[str]] | None = None) -> RouteDecision:
    """Deterministic routing. No network, no model, no API key.

    Used as the primary router when the LLM is unreachable, and as the
    tie-breaker when the model router returns low confidence. Keeping it
    always-available is what lets the serving plane degrade to structured tool
    output rather than failing (requirement 39).
    """
    lowered = query.lower()

    scores: dict[AgentKind, float] = {}
    matched: dict[AgentKind, list[str]] = {}
    for kind, terms in _AGENT_TERMS.items():
        total = 0.0
        hits: list[str] = []
        for term, weight in terms.items():
            if term in lowered:
                total += weight
                hits.append(term)
        scores[kind] = total
        matched[kind] = hits

    best = max(scores.items(), key=lambda kv: kv[1])
    agent = best[0] if best[1] > 0 else AgentKind.RISK_ANALYST
    total_score = sum(scores.values())
    confidence = best[1] / total_score if total_score > 0 else 0.0

    hazard = next(
        (h for h, terms in _HAZARD_TERMS.items() if any(t in lowered for t in terms)), None
    )

    region: str | None = None
    for region_id, aliases in (known_regions or {}).items():
        if region_id.lower() in lowered or any(a.lower() in lowered for a in aliases):
            region = region_id
            break

    timeframe: int | None = None
    if match := _TIMEFRAME_RE.search(query):
        timeframe = int(match.group(1)) * _UNIT_DAYS[match.group(2).lower()]

    return RouteDecision(
        agent=agent,
        region=region,
        hazard=hazard,
        timeframe_days=timeframe,
        confidence=round(confidence, 3),
        method="heuristic",
        rationale=(
            f"matched {matched[agent]} for {agent.value}"
            if matched[agent]
            else "no strong signal; defaulted to risk_analyst"
        ),
        matched_terms=tuple(matched[agent]),
    )


def router_schema() -> dict[str, object]:
    """Structured-output schema for the small-model router."""
    return {
        "name": "route_query",
        "description": "Classify a user query for SAT-AI and extract its entities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": [k.value for k in AgentKind],
                    "description": (
                        "risk_analyst: current risk, explanations, provenance, comparisons. "
                        "emergency: affected areas, exposure, response information. "
                        "recovery: post-event damage and change."
                    ),
                },
                "region": {"type": "string", "description": "Region mentioned, or empty."},
                "hazard": {
                    "type": "string",
                    "enum": ["flood", "wildfire", "cyclone", "earthquake", "damage", ""],
                },
                "timeframe_days": {"type": "integer", "description": "0 if unspecified."},
                "confidence": {"type": "number", "description": "0.0 to 1.0."},
            },
            "required": ["agent", "confidence"],
        },
    }
