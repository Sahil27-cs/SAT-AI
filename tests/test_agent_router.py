"""Query routing and the agent system prompts.

Routing is heuristic first and model-assisted second, and that ordering is a
design decision rather than a cost saving: the deterministic router needs no
network, no API key and no model, so a question still reaches the right tools
when the language layer is unreachable. Requirement 39 asks the system to
degrade to structured tool output rather than fail, and that is only possible
if routing never depended on the model in the first place.

The system prompts are tested too. They are the only thing standing between a
capable model and a fluent answer about next week's flooding.
"""

from __future__ import annotations

import pytest

from satai.agents.router import (
    AGENTS,
    SYSTEM_PREAMBLE,
    AgentKind,
    route_heuristic,
    router_schema,
)

REGIONS = {
    "bihar_ganga": ["Middle Ganga plain, Bihar", "Bihar"],
    "mumbai_mmr": ["Mumbai Metropolitan Region", "Mumbai"],
}


# --- routing -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Why is flood risk high in Bihar?", AgentKind.RISK_ANALYST),
        ("Which satellite scenes were used?", AgentKind.RISK_ANALYST),
        ("How confident is the model?", AgentKind.RISK_ANALYST),
        ("Should residents evacuate?", AgentKind.EMERGENCY),
        ("What is the affected population?", AgentKind.EMERGENCY),
        ("How much damage was there after the event?", AgentKind.RECOVERY),
        ("What changed before and after the flood?", AgentKind.RECOVERY),
    ],
)
def test_queries_reach_the_right_agent(query: str, expected: AgentKind) -> None:
    assert route_heuristic(query).agent is expected


def test_an_unrecognised_query_falls_back_to_the_analyst_with_low_confidence() -> None:
    """A fallback is correct; a refusal to route would strand the question."""
    decision = route_heuristic("Hello there")

    assert decision.agent is AgentKind.RISK_ANALYST
    assert decision.confidence == 0.0


def test_routing_needs_no_network_model_or_key() -> None:
    """The property requirement 39 depends on. Stated as a test, not a comment."""
    decision = route_heuristic("What is the flood risk in Bihar?")
    assert decision.method == "heuristic"
    assert decision.agent is AgentKind.RISK_ANALYST


def test_hazard_is_extracted_from_the_wording() -> None:
    assert route_heuristic("Is there flooding near the river?").hazard == "flood"
    assert route_heuristic("Any active fires this week?").hazard == "wildfire"
    assert route_heuristic("When will the cyclone make landfall?").hazard == "cyclone"


def test_an_earthquake_question_is_still_routed_rather_than_dropped() -> None:
    """Routing is not refusal. The agent refuses; the router delivers.

    Silently dropping the query would produce an unexplained non-answer, where
    what the user needs is the reason SAT-AI will not do this.
    """
    decision = route_heuristic("Will there be an earthquake in Delhi next month?")

    assert decision.agent in set(AgentKind)
    assert decision.hazard == "earthquake"


def test_a_region_is_matched_by_id_or_by_display_name() -> None:
    assert route_heuristic("flood risk in bihar_ganga", REGIONS).region == "bihar_ganga"
    assert route_heuristic("flood risk in Mumbai", REGIONS).region == "mumbai_mmr"


def test_an_unknown_place_yields_no_region_rather_than_a_guess() -> None:
    """'Chennai' is not a study area, and guessing the nearest one would be worse."""
    assert route_heuristic("What is the flood risk in Chennai?", REGIONS).region is None


@pytest.mark.parametrize(
    ("query", "days"),
    [
        ("fires over the last 7 days", 7),
        ("risk over the past 3 months", 90),
        ("changes in the last 2 weeks", 14),
        ("damage over the previous 1 year", 365),
    ],
)
def test_timeframes_are_normalised_to_days(query: str, days: int) -> None:
    assert route_heuristic(query).timeframe_days == days


def test_no_timeframe_is_none_not_a_default_window() -> None:
    """A silently assumed window would make an answer unreproducible."""
    assert route_heuristic("What is the flood risk?").timeframe_days is None


def test_confidence_reflects_how_decisive_the_matched_terms_were() -> None:
    decisive = route_heuristic("Should we evacuate immediately?")
    mixed = route_heuristic("What is the damage risk?")

    assert decisive.confidence > mixed.confidence
    assert decisive.matched_terms


# --- agent specifications ----------------------------------------------------


def test_every_agent_has_tools_and_a_prompt_carrying_the_shared_rules() -> None:
    assert set(AGENTS) == set(AgentKind)
    for spec in AGENTS.values():
        assert spec.tools
        assert SYSTEM_PREAMBLE in spec.system_prompt()


def test_agent_tools_are_all_registered() -> None:
    from satai.agents.tools import TOOL_REGISTRY

    for spec in AGENTS.values():
        for name in spec.tools:
            assert name in TOOL_REGISTRY, f"{spec} references unregistered tool {name}"


@pytest.mark.parametrize(
    "phrase",
    [
        "earthquake",
        "official warning",
        "tool",
        "forecast",
    ],
)
def test_the_preamble_states_each_boundary_explicitly(phrase: str) -> None:
    assert phrase in SYSTEM_PREAMBLE.lower()


def test_the_preamble_forbids_inventing_emergency_contacts() -> None:
    lowered = SYSTEM_PREAMBLE.lower()
    assert "never invent" in lowered or "never generate" in lowered
    assert "phone" in lowered or "number" in lowered


def test_the_preamble_requires_data_age_to_be_reported() -> None:
    """Six-to-twelve-day-old extent presented as current is the quiet failure."""
    lowered = SYSTEM_PREAMBLE.lower()
    assert "real-time" in lowered
    assert "revisit" in lowered or "age" in lowered


# --- model-assisted routing schema -------------------------------------------


def test_router_schema_constrains_the_agent_and_hazard_choices() -> None:
    schema = router_schema()
    properties = schema["input_schema"]["properties"]

    assert set(properties["agent"]["enum"]) == {k.value for k in AgentKind}
    assert "earthquake" in properties["hazard"]["enum"]
    assert schema["input_schema"]["required"] == ["agent", "confidence"]
