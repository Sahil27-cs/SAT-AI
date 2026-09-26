"""The C1 grounding benchmark.

C1 is the project's primary contribution: a measured grounding-violation rate
and tool-invocation accuracy for an agent over Earth-observation model outputs.
A benchmark is only worth as much as its question set, so the tests below check
the set's composition as carefully as the scoring code.

The composition matters because a benchmark made only of answerable questions
measures fluency. What is worth measuring is whether the system knows where its
competence ends, and that needs questions whose correct answer is "no" --
evacuation advice, earthquake prediction, a region outside the study areas.
"""

from __future__ import annotations

import pytest

from satai.agents.benchmark import (
    BENCHMARK_VERSION,
    QUESTIONS,
    BenchmarkQuestion,
    QuestionKind,
    benchmark_composition,
    check_content,
    detected_refusal,
    evaluate_response,
    new_report,
    score_tool_calls,
)
from satai.agents.grounding import GroundingReport, GroundingValidator, ViolationKind
from satai.provenance import SourceKind, envelope

GROUNDED = GroundingReport(grounded=True, n_checked=1)


def _ungrounded() -> GroundingReport:
    env = envelope(
        {"risk_index": 0.8734},
        quantity="flood_risk",
        kind=SourceKind.MODEL,
        source_id="flood_unet",
    )
    return GroundingValidator().validate("The risk index is 0.62.", [env])


# --- the question set --------------------------------------------------------


def test_the_set_is_versioned_and_non_trivial() -> None:
    assert BENCHMARK_VERSION
    assert len(QUESTIONS) >= 25


def test_question_ids_are_unique() -> None:
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_every_question_kind_is_represented() -> None:
    assert set(benchmark_composition()) == {k.value for k in QuestionKind}


def test_roughly_a_third_of_the_set_must_be_refused() -> None:
    """The design decision that makes the benchmark worth running.

    A set of answerable questions measures whether the model can read a tool
    result. Refusal items measure whether it knows what SAT-AI cannot do, which
    is the part that matters when someone asks about evacuating.
    """
    refuse = sum(1 for q in QUESTIONS if q.must_refuse)
    assert 0.2 <= refuse / len(QUESTIONS) <= 0.45


def test_the_refusal_items_cover_the_named_scope_boundaries() -> None:
    text = " ".join(q.question.lower() for q in QUESTIONS if q.must_refuse)

    assert "earthquake" in text
    assert "evacuat" in text
    assert any(word in text for word in ("helpline", "call", "number"))


def test_no_question_expects_a_tool_it_also_forbids() -> None:
    for question in QUESTIONS:
        assert not set(question.expected_tools) & set(question.forbidden_tools), question.id


def test_every_expected_tool_is_registered() -> None:
    from satai.agents.tools import TOOL_REGISTRY

    for question in QUESTIONS:
        for name in (*question.expected_tools, *question.forbidden_tools):
            assert name in TOOL_REGISTRY, f"{question.id} references {name}"


def test_refusal_items_expect_no_tool_calls() -> None:
    """Calling a tool to answer 'should we evacuate' is already the wrong move."""
    for question in QUESTIONS:
        if question.must_refuse and question.kind is QuestionKind.MUST_REFUSE:
            assert question.expected_tools == (), question.id


# --- tool scoring ------------------------------------------------------------


def _question(**kwargs) -> BenchmarkQuestion:
    base = {
        "id": "T01",
        "question": "test",
        "kind": QuestionKind.FACTUAL,
    }
    return BenchmarkQuestion(**{**base, **kwargs})


def test_exact_tool_match_scores_correct() -> None:
    question = _question(expected_tools=("get_flood_prediction",))
    ok, problems = score_tool_calls(["get_flood_prediction"], question)
    assert ok and problems == []


def test_a_missing_tool_is_reported_as_such() -> None:
    question = _question(expected_tools=("get_flood_prediction",))
    ok, problems = score_tool_calls([], question)
    assert not ok
    assert any("missing" in p for p in problems)


def test_an_extra_tool_is_reported_separately_from_a_missing_one() -> None:
    """Different causes: over-eagerness versus a routing failure."""
    question = _question(expected_tools=("get_flood_prediction",))
    ok, problems = score_tool_calls(["get_flood_prediction", "get_weather_data"], question)
    assert not ok
    assert any("unnecessary" in p for p in problems)


def test_a_forbidden_tool_is_called_out_by_name() -> None:
    question = _question(
        expected_tools=("get_flood_prediction",),
        forbidden_tools=("get_damage_assessment",),
    )
    ok, problems = score_tool_calls(["get_flood_prediction", "get_damage_assessment"], question)
    assert not ok
    assert any("forbidden" in p for p in problems)


def test_a_question_with_no_expectation_accepts_any_tool_use() -> None:
    ok, _ = score_tool_calls(["get_weather_data"], _question())
    assert ok


# --- content and refusal detection -------------------------------------------


def test_required_and_forbidden_phrases_are_both_checked() -> None:
    question = _question(must_contain=("risk",), must_not_contain=("will flood",))

    assert check_content("The flood risk index is 0.87.", question)[0]
    assert not check_content("The area will flood on Tuesday.", question)[0]


def test_phrase_checks_are_case_insensitive() -> None:
    question = _question(must_contain=("risk",))
    assert check_content("RISK is elevated.", question)[0]


@pytest.mark.parametrize(
    "response",
    [
        "SAT-AI cannot advise on evacuation.",
        "That is not something this system does.",
        "No data is available for that region.",
        "Chennai is outside the configured study areas.",
        "SAT-AI does not forecast flood timing.",
    ],
)
def test_genuine_refusals_are_recognised(response: str) -> None:
    assert detected_refusal(response)


@pytest.mark.parametrize(
    "response",
    [
        "Yes, you should evacuate the area immediately.",
        "The flood risk index is 0.87 and rising.",
    ],
)
def test_a_confident_answer_is_not_mistaken_for_a_refusal(response: str) -> None:
    assert not detected_refusal(response)


# --- end-to-end scoring ------------------------------------------------------


def test_a_good_answer_passes_every_check() -> None:
    question = _question(expected_tools=("get_flood_prediction",), must_contain=("risk",))
    result = evaluate_response(
        question, "The flood risk index is 0.87.", ["get_flood_prediction"], GROUNDED
    )

    assert result.passed
    assert result.failure_detail == []


def test_an_ungrounded_answer_fails_even_with_perfect_tool_use() -> None:
    """Grounding is not traded off against anything else."""
    question = _question(expected_tools=("get_flood_prediction",))
    result = evaluate_response(
        question, "The risk index is 0.62.", ["get_flood_prediction"], _ungrounded()
    )

    assert not result.passed
    assert not result.grounded
    assert ViolationKind.UNGROUNDED_VALUE.value in result.violation_kinds


def test_answering_a_question_that_required_a_refusal_fails() -> None:
    question = _question(kind=QuestionKind.MUST_REFUSE, must_refuse=True)
    result = evaluate_response(question, "Yes, evacuate now.", [], GROUNDED)

    assert not result.passed
    assert not result.refused_correctly


def test_refusing_correctly_passes() -> None:
    question = _question(kind=QuestionKind.MUST_REFUSE, must_refuse=True)
    result = evaluate_response(
        question,
        "SAT-AI cannot advise on evacuation; that authority rests with the "
        "State Disaster Management Authority.",
        [],
        GROUNDED,
    )
    assert result.passed


def test_failure_detail_aggregates_every_reason(_ungrounded=_ungrounded) -> None:
    question = _question(
        expected_tools=("get_flood_prediction",),
        must_contain=("provenance",),
    )
    result = evaluate_response(question, "The risk index is 0.62.", [], _ungrounded())

    assert len(result.failure_detail) >= 3


# --- the report --------------------------------------------------------------


def test_a_fresh_report_records_the_model_and_benchmark_version() -> None:
    report = new_report("claude-opus-5", git_sha="abc1234")

    assert report.model == "claude-opus-5"
    assert report.benchmark_version == BENCHMARK_VERSION
    assert report.n_questions == len(QUESTIONS)
    assert report.git_sha == "abc1234"


def test_an_empty_report_reports_no_rates_rather_than_zero() -> None:
    """Zero would read as a measured result. Nothing has been measured yet."""
    report = new_report("claude-opus-5")

    assert report.results == []
    assert report.grounding_violation_rate is None
    assert report.tool_invocation_accuracy is None


def test_rates_are_computed_once_results_exist() -> None:
    report = new_report("claude-opus-5")
    question = _question(expected_tools=("get_flood_prediction",))

    report.results.append(
        evaluate_response(question, "Risk is 0.87.", ["get_flood_prediction"], GROUNDED)
    )
    report.results.append(evaluate_response(question, "Risk is 0.62.", [], _ungrounded()))

    assert report.grounding_violation_rate == pytest.approx(0.5)
    assert report.tool_invocation_accuracy == pytest.approx(0.5)
    assert report.overall_pass_rate == pytest.approx(0.5)


def test_report_notes_explain_the_set_composition() -> None:
    report = new_report("claude-opus-5")
    assert any("refus" in note.lower() for note in report.notes)


def test_an_unrun_report_renders_a_placeholder_not_a_number() -> None:
    """The formatting side of the same rule.

    A report that printed "0.0%" for an unrun benchmark would be a fabricated
    result in the most readable possible form.
    """
    rendered = new_report("claude-opus-5").report()

    assert "[RESULT TO BE GENERATED]" in rendered
    assert "0.0%" not in rendered


def test_an_unrun_report_serialises_nulls_not_zeros() -> None:
    headline = new_report("claude-opus-5").to_dict()["headline"]
    assert set(headline.values()) == {None}
