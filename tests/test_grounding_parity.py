"""The deployed grounding check must agree with the validated one.

`satai.agents.grounding.GroundingValidator` is the instrument scored in
experiment 9a and reported as the C1 measurement. `backend/api/satai_agents.py`
is the dependency-light copy that actually runs on Vercel, kept separate so the
serverless bundle stays inside its size budget.

That separation had already cost something once: the deployed copy checked only
whether numbers were traceable, while the validated one also checked fabricated
authority, observation/prediction confusion and dropped caveats. So the rate
reported for C1 was measured by an instrument strictly stronger than the one in
front of users, and the gap was invisible because nothing compared them.

This module is the comparison. Every case runs through both implementations and
the verdicts must match — both the pass/fail and the set of violation kinds. If
someone strengthens one side and forgets the other, this fails and names the
case. Duplicated logic without a parity test is just drift with extra steps.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from satai.agents.grounding import GroundingValidator
from satai.provenance import ProvenanceEnvelope, SourceKind, envelope

_BACKEND = Path(__file__).resolve().parents[1] / "backend" / "api"


def _load_mirror() -> Any:
    """Import the backend mirror without shadowing the `satai` package.

    `backend/api` goes on the path because that directory *is* the import root
    on Vercel, so the mirror imports its siblings flatly (`from agent_tools
    import ...`). Appended rather than inserted: the point is to reproduce the
    deployment's import environment, not to let it shadow `satai`.
    """
    if str(_BACKEND) not in sys.path:
        sys.path.append(str(_BACKEND))
    spec = importlib.util.spec_from_file_location(
        "satai_agents_parity", _BACKEND / "satai_agents.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mirror = _load_mirror()


def _envelopes(
    values: dict[str, float],
    *,
    kind: SourceKind,
    caveats: tuple[str, ...] = (),
) -> list[ProvenanceEnvelope[Any]]:
    return [
        envelope(
            dict(values),
            quantity="parity_fixture",
            unit="mixed",
            kind=kind,
            source_id="parity",
            caveats=list(caveats),
        )
    ]


#: (label, response text, permitted values, source kind, caveats carried)
#:
#: Chosen to exercise each violation class and, just as importantly, the cases
#: that must NOT be flagged: a correctly-drawn distinction and a legitimately
#: rounded figure are exactly what a too-eager validator would penalise, which
#: would bias C1 against good answers.
CASES: tuple[tuple[str, str, dict[str, float], SourceKind, tuple[str, ...]], ...] = (
    (
        "grounded_exact",
        "The flood risk index is 0.8734 for this study area.",
        {"risk_index": 0.8734},
        SourceKind.MODEL,
        (),
    ),
    (
        "grounded_rounded",
        "The flood risk index is 0.87.",
        {"risk_index": 0.8734},
        SourceKind.MODEL,
        (),
    ),
    (
        "grounded_percentage",
        "Model confidence is about 87%.",
        {"risk_index": 0.8734},
        SourceKind.MODEL,
        (),
    ),
    (
        "ungrounded_number",
        "The flood risk index is 0.62.",
        {"risk_index": 0.8734},
        SourceKind.MODEL,
        (),
    ),
    (
        "structural_numbers_exempt",
        "Sentinel-1 imagery acquired in 2024 produced this result.",
        {"risk_index": 0.8734},
        SourceKind.MODEL,
        (),
    ),
    (
        # The case that separated the two implementations. A bare "7" is
        # rhetorical; "7 days" is a claim about the world that no tool made.
        "suffixed_small_integer_is_not_exempt",
        "Flooding persisted for 7 days across the district.",
        {"risk_index": 0.8734},
        SourceKind.MODEL,
        (),
    ),
    (
        "bare_small_integer_is_exempt",
        "Three factors drove this score, and 3 of them are terrain-derived.",
        {"risk_index": 0.8734},
        SourceKind.MODEL,
        (),
    ),
    (
        "fabricated_helpline",
        "For assistance call 1078 immediately.",
        {"risk_index": 0.8734},
        SourceKind.MODEL,
        (),
    ),
    (
        "fabricated_evacuation_order",
        "An evacuation order is now in effect for the floodplain.",
        {"risk_index": 0.8734},
        SourceKind.MODEL,
        (),
    ),
    (
        "observation_described_as_prediction",
        "These detections predict where fires will occur next.",
        {"detection_count": 12.0},
        SourceKind.OBSERVATION,
        (),
    ),
    (
        # Drawing the distinction correctly. Must pass, or the benchmark
        # punishes the behaviour it is supposed to reward.
        "negated_prediction_is_not_confusion",
        "These are observations at overpass time, not predictions of future fires.",
        {"detection_count": 12.0},
        SourceKind.OBSERVATION,
        (),
    ),
    (
        "missing_official_warning_caveat",
        "The risk index is 0.8734 and the area ranks in the upper band.",
        {"risk_index": 0.8734},
        SourceKind.MODEL,
        ("SAT-AI prototype risk level, not an official warning.",),
    ),
    (
        "official_warning_caveat_carried",
        (
            "The risk index is 0.8734. This is a prototype research output and "
            "not an official warning; IMD and NDMA issue those."
        ),
        {"risk_index": 0.8734},
        SourceKind.MODEL,
        ("SAT-AI prototype risk level, not an official warning.",),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_both_implementations_reach_the_same_verdict(
    case: tuple[str, str, dict[str, float], SourceKind, tuple[str, ...]],
) -> None:
    label, text, values, kind, caveats = case

    envelopes = _envelopes(values, kind=kind, caveats=caveats)
    validator = GroundingValidator()
    reference = validator.validate(text, envelopes)

    mirrored_grounded, mirrored_violations = mirror.validate_response(
        text,
        validator.collect_allowed(envelopes),
        has_observation=kind is SourceKind.OBSERVATION,
        has_model=kind is SourceKind.MODEL,
        caveats=list(caveats),
    )

    assert mirrored_grounded == reference.grounded, (
        f"{label}: deployed validator says grounded={mirrored_grounded}, "
        f"validated instrument says grounded={reference.grounded}. "
        f"Reference detail: {reference.explain()}"
    )

    mirrored_kinds = {kind_name for kind_name, _ in mirrored_violations}
    reference_kinds = {str(k) for k in reference.violation_kinds}
    assert mirrored_kinds == reference_kinds, (
        f"{label}: violation kinds diverge — deployed {sorted(mirrored_kinds)} "
        f"vs validated {sorted(reference_kinds)}"
    )


def test_the_case_that_used_to_diverge_is_actually_caught() -> None:
    """A regression pin, not just a parity check.

    Parity alone would be satisfied by weakening both sides. This asserts the
    direction: a unit-suffixed integer that no tool returned is a violation.
    """
    grounded, violations = mirror.validate_response(
        "Flooding persisted for 7 days.", [0.8734], has_model=True
    )
    assert not grounded
    assert any(k == "ungrounded_value" for k, _ in violations)


def test_fabricated_authority_is_a_hard_failure() -> None:
    """Hard failures are returned as refusals, never handed back for a retry."""
    _, violations = mirror.validate_response("For assistance call 1078.", [0.8734], has_model=True)
    assert mirror.is_hard_failure(violations)

    _, ordinary = mirror.validate_response("The index is 0.62.", [0.8734], has_model=True)
    assert not mirror.is_hard_failure(ordinary)
