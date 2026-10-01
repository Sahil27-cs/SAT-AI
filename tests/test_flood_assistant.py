"""The flood assistant: its tools, and the loop that holds the model to them.

Two layers are tested separately because they fail differently. The tools can
return the wrong number (a typed constant drifting from its report, a
validation score filed as a test score). The loop can let the model state a
number no tool returned, or call a blocked extent confirmed flooding. Both are
checked here without a network: Gemini is replaced by a scripted stub that
plays back tool calls and drafts, and every assertion is on what reaches the
user.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="fastapi is a serving-plane dependency")

REPO = Path(__file__).resolve().parents[1]
_BACKEND = REPO / "backend" / "api"


def _load(name: str) -> Any:
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _BACKEND / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


index = _load("index")
flood = _load("flood_tools")
tools = _load("agent_tools")
agents = _load("satai_agents")
gemini = _load("gemini")


def run(name: str, **arguments: Any) -> dict[str, Any]:
    return flood.execute_flood_tool(name, arguments)


# --- the facts are copies of the reports ------------------------------------


def test_the_exported_facts_match_their_source_reports() -> None:
    """A number in the tools that disagrees with the report that measured it fails here."""
    spec = importlib.util.spec_from_file_location(
        "export_flood_facts", REPO / "scripts" / "export_flood_facts.py"
    )
    assert spec and spec.loader
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    assert exporter.main(["--check"]) == 0


def test_every_flood_tool_is_offered_and_the_stale_generic_ones_are_not() -> None:
    offered = {t["name"] for t in tools.TOOL_DECLARATIONS}
    assert offered >= flood.FLOOD_TOOL_NAMES
    # get_hazard_result reads a table no run has written and answered "not
    # computed" about a scene that has been processed.
    assert "get_hazard_result" not in offered
    assert {"get_study_area", "show_on_map"} <= offered


# --- model and metrics --------------------------------------------------------


def test_the_model_card_states_the_trained_configuration() -> None:
    card = run("get_flood_model_info")
    assert card["available"] is True
    assert card["parameters"] == 7763041
    assert card["parameters_millions"] == 7.76
    assert card["bands"] == ["vv_db", "vh_db", "vv_vh_ratio"]
    assert card["loss"]["bce_weight"] == 0.5
    assert card["loss"]["dice_weight"] == 0.5
    assert card["threshold"] == 0.5
    assert "sigmoid" in card["decision_rule"]
    assert "iou" not in str(card).lower().replace("iou_", "")  # no accuracy in the card


def test_the_india_test_score_is_the_india_test_score() -> None:
    result = run("get_flood_metrics", split="india_test")
    india = result["splits"]["india_test"]
    assert list(result["splits"]) == ["india_test"]
    assert india["region"] == ["India"]
    assert india["n_chips"] == 68
    assert (india["iou"], india["f1"], india["precision"], india["recall"]) == (
        0.523,
        0.6868,
        0.7506,
        0.6331,
    )
    assert india["label"].startswith("TEST")


def test_the_mekong_number_comes_back_labelled_as_validation_only() -> None:
    result = run("get_flood_metrics", split="mekong_validation")
    mekong = result["splits"]["mekong_validation"]
    assert list(result["splits"]) == ["mekong_validation"]
    assert mekong["region"] == ["Mekong"]
    assert mekong["iou"] == 0.8679
    assert mekong["label"].startswith("VALIDATION")
    assert "NOT the India score" in mekong["label"]
    # And the India figure is not smuggled in beside it.
    assert 0.523 not in tools.groundable_values([result])


def test_all_splits_stay_separate_and_each_keeps_its_label() -> None:
    result = run("get_flood_metrics", split="all")
    splits = result["splits"]
    assert set(splits) == {"india_test", "mekong_validation", "otsu_india_test"}
    assert splits["india_test"]["iou"] != splits["mekong_validation"]["iou"]
    assert splits["otsu_india_test"]["iou"] == 0.3754
    assert result["validation_to_test_gap_iou"] == 0.3449
    assert all("label" in s for s in splits.values())


def test_an_unknown_split_is_unavailable_not_defaulted() -> None:
    result = run("get_flood_metrics", split="bihar_test")
    assert result["available"] is False
    assert result["status"] == "DATA UNAVAILABLE"


# --- ground truth -------------------------------------------------------------


def test_ground_truth_is_sen1floods11_labelhand() -> None:
    result = run("get_flood_ground_truth")
    dataset, truth = result["dataset"], result["ground_truth"]
    assert dataset["name"] == "Sen1Floods11 v1.1 HandLabeled"
    assert dataset["n_chips"] == 446
    assert dataset["n_events"] == 11
    assert dataset["split"]["test_chips"] == 68
    assert dataset["split"]["test_region"] == ["India"]
    assert truth["name"] == "Sen1Floods11 LabelHand"
    assert truth["values"] == {"1": "water", "0": "not water", "-1": "not annotated"}
    assert "no ground truth" in truth["not_available_for"].lower()


def test_the_split_accounts_for_every_chip() -> None:
    split = run("get_flood_ground_truth")["dataset"]["split"]
    total = (
        split["train_chips"]
        + split["validation_chips"]
        + split["test_chips"]
        + split["reserved_chips"]
    )
    assert total == 446


# --- the blocked 74.8 km2 -----------------------------------------------------


def _unet_result(region: str | None = "nepal_koshi_terai") -> dict[str, Any]:
    summary = run("get_flood_inference_summary", region=region)
    return next(r for r in summary["results"] if r["method"] == "flood_unet")


def test_the_raw_unet_extent_is_returned_as_blocked_and_unvalidated() -> None:
    unet = _unet_result()
    assert unet["raw_extent_km2"] == 74.8
    assert unet["status"] == "BLOCKED"
    assert unet["validated"] is False
    assert unet["confirmed_flooding"] is False
    assert unet["drawn_on_map"] is False
    assert "NOT confirmed flooding" in unet["interpretation"]


def test_the_blocked_extent_is_never_reported_under_a_confirmed_name() -> None:
    summary = run("get_flood_inference_summary", region="nepal_koshi_terai")
    unet = next(r for r in summary["results"] if r["method"] == "flood_unet")
    assert "flood_extent_km2" not in unet
    assert "flooded_area_km2" not in unet


def test_the_scene_status_explains_why_it_was_blocked() -> None:
    status = run("get_flood_scene_status", region="nepal_koshi_terai")
    blocked = [s for s in status["scenes"] if s["status"] == "BLOCKED"]
    assert len(blocked) == 1
    assert blocked[0]["method"] == "flood_unet"
    assert blocked[0]["gate_verdict"] == "fail"
    assert "distribution gate" in blocked[0]["blocked_reason"]
    assert blocked[0]["drawn_on_map"] is False


# --- distribution gate --------------------------------------------------------


def test_the_gate_reports_the_failing_and_passing_bands() -> None:
    gate = run("get_distribution_gate", region="nepal_koshi_terai")
    assert gate["p_value_used"] is False
    assert gate["fail_wasserstein_db"] == 3.0
    verdict = gate["scene_verdicts"][0]
    assert verdict["verdict"] == "fail"
    assert verdict["may_proceed"] is False
    bands = {b["band"]: b["verdict"] for b in verdict["bands"]}
    assert bands == {"vh_db": "warn", "vv_db": "fail", "vv_vh_ratio": "pass"}


def test_the_gate_without_a_scene_still_explains_itself() -> None:
    gate = run("get_distribution_gate", region="bihar_ganga")
    assert gate["available"] is True
    assert gate["scene_verdicts"] == []
    assert "No flood scene" in gate["scene_verdicts_note"]


# --- unavailable data ---------------------------------------------------------


@pytest.mark.parametrize("name", ["get_flood_scene_status", "get_flood_inference_summary"])
def test_a_region_with_no_scene_is_unavailable(name: str) -> None:
    result = run(name, region="bihar_ganga")
    assert result["available"] is False
    assert result["status"] == "DATA UNAVAILABLE"
    assert tools.groundable_values([result]) == []


def test_an_unknown_region_is_unavailable() -> None:
    result = run("get_flood_scene_status", region="london")
    assert result["available"] is False
    assert "not a SAT-AI study area" in result["reason"]


def test_missing_facts_are_unavailable_not_invented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flood, "_FACTS_CACHE", {})
    for name in ("get_flood_model_info", "get_flood_metrics", "get_flood_ground_truth"):
        result = run(name, split="india_test")
        assert result["available"] is False, name


# --- explainability -----------------------------------------------------------


def test_xai_shares_are_attribution_not_causation() -> None:
    xai = run("get_flood_xai")
    assert xai["n_chips"] == 12
    assert xai["attribution_share_percent"] == {
        "vv_db": 29.8,
        "vh_db": 16.8,
        "vv_vh_ratio": 53.4,
    }
    assert xai["methods"] == ["integrated_gradients", "occlusion"]
    assert any("not physical causation" in c for c in xai["caveats"])


# --- official warnings ----------------------------------------------------------


def test_the_scope_says_no_official_warnings() -> None:
    scope = run("get_system_scope")
    assert scope["issues_official_warnings"] is False
    assert scope["forecasts_floods"] is False
    assert scope["real_time"] is False
    assert "IMD" in scope["official_sources"]["India"]


def test_an_answer_claiming_an_issued_warning_is_a_hard_failure() -> None:
    results = [{**run("get_system_scope"), "_tool": "get_system_scope"}]
    _, _, caveats = agents._evidence(results)
    ok, violations = agents.validate_response(
        "An official warning has been issued for the Koshi basin.",
        tools.groundable_values(results),
        caveats=caveats,
    )
    assert not ok
    assert agents.is_hard_failure(violations)


def test_a_scene_answer_that_drops_the_not_official_caveat_is_rejected() -> None:
    results = [run("get_flood_scene_status", region="nepal_koshi_terai")]
    has_obs, has_model, caveats = agents._evidence(results)
    ok, violations = agents.validate_response(
        "The U-Net run on the Nepal scene was blocked by the distribution gate.",
        tools.groundable_values(results),
        has_observation=has_obs,
        has_model=has_model,
        caveats=caveats,
    )
    assert not ok
    assert any(kind == "missing_caveat" for kind, _ in violations)


def test_a_scene_result_counts_as_observation_and_model() -> None:
    """A Sentinel-1 acquisition is observed; the extent is modelled. Both are true."""
    has_obs, has_model, _ = agents._evidence(
        [run("get_flood_inference_summary", region="nepal_koshi_terai")]
    )
    assert has_obs and has_model


# --- the map ----------------------------------------------------------------------


def _show(region: str, layers: list[str]) -> dict[str, Any]:
    return tools._show_on_map({"region": region, "layers": layers}, index.study_areas())


def test_the_flood_layer_is_available_only_where_a_drawable_extent_exists() -> None:
    nepal = _show("nepal_koshi_terai", ["flood"])
    assert "flood" in nepal["layers_activated"]
    bihar = _show("bihar_ganga", ["flood"])
    assert "flood" not in bihar["layers_activated"]
    assert "flood" in bihar["layers_unavailable"]


def test_a_blocked_result_never_makes_its_layer_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalogue = {
        "study_areas": [
            {
                "id": "bihar_ganga",
                "analyses": [
                    {
                        "hazard": "flood",
                        "kind": "model_inference",
                        "displayable": False,
                        "overlays": [{"layer": "flood", "url": "/layers/x.png"}],
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr(flood, "_ANALYSES_CACHE", catalogue)
    assert "flood" not in _show("bihar_ganga", ["flood"])["layers_activated"]


# --- the loop, with a scripted model ----------------------------------------------


class _Script:
    """Plays back Gemini turns: each entry is (text, [(tool, args), ...])."""

    def __init__(self, turns: list[tuple[str, list[tuple[str, dict[str, Any]]]]]) -> None:
        self.turns = list(turns)
        self.requests: list[dict[str, Any]] = []

    async def __call__(
        self, contents: list[dict[str, Any]], **kwargs: Any
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        self.requests.append({"contents": list(contents), **kwargs})
        text, calls = self.turns.pop(0)
        parts = [{"functionCall": {"name": n, "args": a}} for n, a in calls]
        return text, parts, {"model": "gemini-2.5-flash"}


def _ask(monkeypatch: pytest.MonkeyPatch, question: str, script: _Script) -> Any:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(agents, "call_gemini", script)

    async def no_rows(*_a: Any, **_k: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(index, "_query", no_rows)
    return asyncio.run(agents.answer(index.ChatRequest(message=question)))


def test_the_model_calls_the_metrics_tool_and_its_answer_is_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _Script(
        [
            ("", [("get_flood_metrics", {"split": "india_test"})]),
            ("On the held-out India test split (68 chips) the U-Net scores IoU 0.523.", []),
        ]
    )
    response = _ask(monkeypatch, "What are the India test metrics?", script)
    assert response.tools_called == ["get_flood_metrics"]
    assert response.grounded is True
    assert response.degraded is False
    assert "0.523" in response.answer
    assert any("gemini-2.5-flash" in n for n in response.notes)


def test_calling_the_validation_score_the_india_score_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the India split was fetched, so 0.868 is traceable to nothing."""
    script = _Script(
        [
            ("", [("get_flood_metrics", {"split": "india_test"})]),
            ("The model scores IoU 0.868 on India.", []),
            ("The model scores IoU 0.868 on India.", []),  # the rewrite repeats it
        ]
    )
    response = _ask(monkeypatch, "How accurate is it on India?", script)
    assert response.grounded is False
    assert "0.868" not in response.answer
    assert response.answer.startswith(agents.UNGROUNDED)
    assert any(n.startswith("Grounding violation") for n in response.notes)


def test_a_fixable_draft_is_rewritten_once_and_the_rewrite_is_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _Script(
        [
            ("", [("get_flood_metrics", {"split": "india_test"})]),
            ("IoU is 0.61 on India.", []),
            ("On the India test split IoU is 0.523.", []),
        ]
    )
    response = _ask(monkeypatch, "India IoU?", script)
    assert response.grounded is True
    assert response.answer == "On the India test split IoU is 0.523."
    # The rejected draft is still recorded: it is what C1 counts.
    assert any(n.startswith("Grounding violation in first draft") for n in response.notes)
    # The rewrite was asked for without tools.
    assert script.requests[-1]["tools"] is None


def test_the_blocked_extent_described_as_confirmed_flooding_needs_the_caveats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _Script(
        [
            ("", [("get_flood_inference_summary", {"region": "nepal_koshi_terai"})]),
            ("74.8 km2 flooded.", []),
            (
                "The U-Net's 74.8 km2 is raw, unvalidated output: the distribution gate "
                "failed, so it is not confirmed flooding. This is not an official "
                "warning; official sources are IMD and NDMA.",
                [],
            ),
        ]
    )
    response = _ask(monkeypatch, "Explain the 74.8 km2 raw inference", script)
    assert response.grounded is True
    assert "not confirmed flooding" in response.answer


def test_an_invented_value_for_unavailable_data_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _Script(
        [
            ("", [("get_flood_scene_status", {"region": "bihar_ganga"})]),
            ("About 212 km2 of Bihar is under water.", []),
            ("About 212 km2 of Bihar is under water.", []),
        ]
    )
    response = _ask(monkeypatch, "How much of Bihar is flooded?", script)
    assert response.grounded is False
    assert "212" not in response.answer
    assert "No flood scene has been processed for bihar_ganga" in response.answer


def test_a_fabricated_warning_is_not_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _Script(
        [
            ("", [("get_system_scope", {})]),
            ("NDMA has issued an alert, and an official warning has been issued.", []),
        ]
    )
    response = _ask(monkeypatch, "Does SAT-AI give official warnings?", script)
    assert response.grounded is False
    assert response.answer.startswith(agents.OVERREACHED)
    assert len(script.requests) == 2  # no rewrite was requested
    assert any(n.startswith("Hard failure") for n in response.notes)


def test_evacuation_is_refused_before_the_model_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _Script([])
    response = _ask(monkeypatch, "Should we evacuate the Koshi area?", script)
    assert response.route_method == "refusal_rule"
    assert script.requests == []


def test_the_tool_round_budget_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_MAX_TOOL_ROUNDS", "1")
    script = _Script(
        [
            ("", [("get_flood_model_info", {})]),
            ("The model has 7.76 million parameters.", []),
        ]
    )
    response = _ask(monkeypatch, "Explain the flood model", script)
    assert script.requests[0]["tools"] is not None
    assert script.requests[1]["tools"] is None
    assert response.grounded is True


# --- all nine required flood tools and queries ------------------------------


def test_all_nine_required_flood_tools_are_offered() -> None:
    required = {
        "get_flood_model_info",
        "get_flood_metrics",
        "get_ground_truth_info",
        "get_flood_scene_status",
        "get_distribution_gate",
        "get_flood_inference_summary",
        "get_xai_summary",
        "get_risk_summary",
        "get_provenance",
    }
    offered = {t["name"] for t in tools.TOOL_DECLARATIONS}
    assert required <= offered
    assert required <= flood.FLOOD_TOOL_NAMES


def test_get_ground_truth_info_returns_authoritative_facts() -> None:
    gt = run("get_ground_truth_info")
    assert gt["available"] is True
    assert gt["dataset"]["name"] == "Sen1Floods11 v1.1 HandLabeled"
    assert gt["dataset"]["n_chips"] == 446
    assert gt["dataset"]["n_events"] == 11
    assert gt["dataset"]["split"]["test_chips"] == 68
    assert gt["ground_truth"]["name"] == "Sen1Floods11 LabelHand"
    assert gt["ground_truth"]["values"] == {"1": "water", "0": "not water", "-1": "not annotated"}


def test_get_xai_summary_returns_grounded_attributions() -> None:
    xai = run("get_xai_summary")
    assert xai["available"] is True
    assert xai["model"] == "flood_unet"
    assert xai["n_chips"] == 12
    assert xai["attribution_share_percent"] == {
        "vv_db": 29.8,
        "vh_db": 16.8,
        "vv_vh_ratio": 53.4,
    }
    assert any("not physical causation" in c for c in xai["caveats"])


def test_get_risk_summary_returns_formulation_and_c4_sensitivity() -> None:
    risk = run("get_risk_summary", region="nepal_koshi_terai")
    assert risk["available"] is True
    assert "H^alpha * E^beta * V^gamma" in risk["formulation"]
    assert risk["default_exponents"] == {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}
    assert "E = 0 implies R = 0" in risk["boundary_condition"]
    # The worst case over all five correlation settings, from the reports. An
    # earlier version of the tool asserted 0.98 and 28%, which no report holds.
    c4 = risk["c4_sensitivity"]
    assert c4["min_spearman_rank_correlation"] == 0.957
    assert c4["max_band_reassignment_percent"] == 18.7
    assert c4["n_settings"] == 5
    # Nepal has a displayable risk analysis built on the Otsu extent.
    assert risk["has_risk_map"] is True
    assert risk["risk_maps"][0]["vulnerability_included"] is False


def test_get_risk_summary_says_none_where_none_was_run() -> None:
    risk = run("get_risk_summary", region="bihar_ganga")
    assert risk["has_risk_map"] is False
    assert risk["risk_maps"] == []
    assert "none is estimated" in risk["regional_status"]


def test_get_provenance_returns_complete_scene_and_model_trace() -> None:
    prov = run("get_provenance", region="nepal_koshi_terai")
    assert prov["available"] is True
    assert prov["model_provenance"]["model"] == "flood_unet"
    assert prov["model_provenance"]["parameters_millions"] == 7.76
    scene = next(s for s in prov["scenes_provenance"] if s["region"] == "nepal_koshi_terai")
    assert scene["source_scene"] == (
        "S1A_IW_GRDH_1SDV_20240927T001159_20240927T001224_055843_06D317_rtc"
    )
    assert "2024-09-27" in scene["acquisition_date"]
    assert scene["satellite_platform"] == "SENTINEL-1A"
    assert scene["radiometry"] == "gamma0_rtc_linear"
    assert scene["validation_status"] == "BLOCKED"
    assert scene["gate_verdict"] == "fail"
    assert scene["displayable_on_map"] is False


def test_flood_forecasting_tomorrow_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _Script([])
    response = _ask(monkeypatch, "Can you predict whether Mumbai will flood tomorrow?", script)
    assert response.route_method == "refusal_rule"
    assert "SAT-AI does not forecast hazard timing or occurrence" in response.answer
    assert script.requests == []


def test_chatbot_answers_74_8_km2_case_honestly(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _Script(
        [
            ("", [("get_flood_inference_summary", {"region": "nepal_koshi_terai"})]),
            (
                "The model generated a raw 74.8 km² inference, but the distribution gate "
                "rejected the scene because its input distribution differed from the "
                "training distribution. Therefore the result is not treated as a "
                "validated flood extent and is not displayed as confirmed flooding. "
                "This is not an official warning; official sources in Nepal include DHM.",
                [],
            ),
        ]
    )
    response = _ask(monkeypatch, "What happened with the 74.8 km2 result?", script)
    assert response.grounded is True
    assert "74.8" in response.answer
    assert (
        "not confirmed flooding" in response.answer
        or "not displayed as confirmed flooding" in response.answer
    )


# --- the degraded answer ----------------------------------------------------------


@pytest.mark.parametrize(
    "tool_calls",
    [
        [("get_flood_model_info", {}), ("get_flood_ground_truth", {})],
        [("get_flood_metrics", {"split": "all"})],
        [
            ("get_distribution_gate", {"region": "nepal_koshi_terai"}),
            ("get_flood_inference_summary", {"region": "nepal_koshi_terai"}),
        ],
        [("get_flood_xai", {})],
        [("get_risk_summary", {"region": "nepal_koshi_terai"})],
    ],
)
def test_the_degraded_summary_states_only_tool_values(
    tool_calls: list[tuple[str, dict[str, Any]]],
) -> None:
    """The fallback text skips the model, so it must not carry a number of its own.

    An earlier version typed its summary by hand and stated 4,831 patches, 288
    India chips, Otsu IoU 0.4357, rho >= 0.98 and 28 %: none of them true.
    """
    results = []
    for name, arguments in tool_calls:
        result = run(name, **arguments)
        result["_tool"] = name
        results.append(result)
    summary = "\n".join(agents._synthesis(results))
    assert summary
    grounded, ungrounded = agents.validate_grounding(summary, tools.groundable_values(results))
    assert grounded, ungrounded
    for wrong in ("4,831", "288", "0.4357", "0.98", "28%", "202008"):
        assert wrong not in summary
