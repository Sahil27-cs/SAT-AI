"""The model registry must describe what exists, not what was intended.

This file exists because the registry is the record everything downstream
trusts. If it says TRAINED, a reader — a report, a viva panel, the dashboard —
takes that as a claim that a training run happened. So the rules that keep it
honest are enforced here rather than left to whoever edits it next:

* status is only ever one of the declared vocabulary;
* TRAINED requires a training manifest to have produced the entry;
* an entry with no evaluation report carries no metrics at all, rather than
  zeros or optimistically-named nulls;
* the unsupervised baseline is never marked TRAINED, because it is not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "ml" / "registry") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ml" / "registry"))

REGISTRY = REPO_ROOT / "ml" / "registry" / "models.json"

VALID_STATUS = {"TRAINED AND EVALUATED", "TRAINED", "EVALUATED", "NOT TRAINED"}


@pytest.fixture(scope="module")
def registry() -> dict[str, Any]:
    if not REGISTRY.exists():
        pytest.skip("no registry yet. Build it with: python -m ml.registry.build_registry")
    data: dict[str, Any] = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return data


def test_the_registry_declares_its_status_vocabulary(registry: dict[str, Any]) -> None:
    assert set(registry["status_vocabulary"]) == VALID_STATUS


def test_every_model_uses_a_declared_status(registry: dict[str, Any]) -> None:
    for model in registry["models"]:
        assert model["status"] in VALID_STATUS, f"{model['name']}: {model['status']!r}"


def test_a_trained_model_records_when_and_how_long(registry: dict[str, Any]) -> None:
    """TRAINED is a claim about an event. It needs the event's evidence."""
    for model in registry["models"]:
        if "TRAINED" in model["status"]:
            assert model["trained_at"], f"{model['name']} claims TRAINED with no timestamp"
            assert model["epochs_run"] > 0, f"{model['name']} claims TRAINED with 0 epochs"
            assert model["n_parameters"] > 0


def test_an_unevaluated_model_carries_no_metrics(registry: dict[str, Any]) -> None:
    """A null named `iou` reads as a score of zero to anything that formats it."""
    for model in registry["models"]:
        if "EVALUATED" not in model["status"]:
            assert "test_metrics" not in model, f"{model['name']} has metrics without an evaluation"


def test_an_evaluated_model_points_at_its_report(registry: dict[str, Any]) -> None:
    for model in registry["models"]:
        if "EVALUATED" in model["status"]:
            assert model.get("evaluation_report"), model["name"]
            assert (REPO_ROOT / model["evaluation_report"]).is_file(), (
                f"{model['name']} cites a report that does not exist: {model['evaluation_report']}"
            )


def test_the_unsupervised_baseline_is_never_marked_trained(registry: dict[str, Any]) -> None:
    """Otsu derives its threshold per chip. There is no training run to claim."""
    baseline = next((m for m in registry["models"] if m["name"] == "otsu_baseline"), None)
    if baseline is None:
        pytest.skip("baseline not in the registry")
    assert baseline["status"] == "EVALUATED"
    assert baseline["trained_at"] is None


def test_artifact_presence_is_separate_from_status(registry: dict[str, Any]) -> None:
    """Weights are gitignored, so a clone has the record without the file.

    Conflating the two would make every fresh checkout look as though nothing
    had ever been trained.
    """
    for model in registry["models"]:
        assert "artifact_present" in model
        assert isinstance(model["artifact_present"], bool)


def test_every_evaluated_model_states_its_test_region(registry: dict[str, Any]) -> None:
    """A metric without the region it was measured on is not interpretable."""
    for model in registry["models"]:
        if "EVALUATED" in model["status"]:
            assert model["test_regions"], model["name"]
            assert model["region_disjoint"] is True, (
                f"{model['name']} reports a metric from a split that is not region-disjoint"
            )


def test_the_deep_model_beats_the_baseline_it_is_compared_against(
    registry: dict[str, Any],
) -> None:
    """The project's own standard: a deep model that cannot beat Otsu has
    demonstrated nothing. Encoded so a regression in the model is a failing
    test rather than a quietly worse number in a report."""
    models = {(m["name"], m["variant"]): m for m in registry["models"]}
    deep = models.get(("flood_unet", "sar_ratio"))
    baseline = models.get(("otsu_baseline", "vv_no_hand"))
    if deep is None or baseline is None:
        pytest.skip("both models are needed for this comparison")

    assert deep["test_regions"] == baseline["test_regions"], (
        "the comparison is only meaningful on the same held-out region"
    )
    assert deep["test_metrics"]["iou"] > baseline["test_metrics"]["iou"]


def test_recorded_paths_are_posix(registry: dict[str, Any]) -> None:
    """A Windows separator in a recorded path is a broken citation on Linux.

    The registry is written on whatever machine trained the model and read in
    CI, in the serverless functions and in every clone. A backslash-separated
    path resolves to nothing there, and the result reads as a missing artifact
    rather than as a path bug -- which is why this is a test and not a comment.
    """
    for model in registry["models"]:
        for key in ("artifact", "normalizer", "evaluation_report"):
            value = model.get(key)
            if isinstance(value, str):
                assert "\\" not in value, f"{model['name']}.{key} = {value!r}"
        for key, value in (model.get("onnx_export") or {}).items():
            if isinstance(value, str):
                assert "\\" not in value, f"{model['name']}.onnx_export.{key} = {value!r}"


def test_an_export_is_recorded_only_with_its_parity_result(registry: dict[str, Any]) -> None:
    """An export nobody compared against the checkpoint must not look servable.

    The whole risk of an ONNX export is that it loads, runs, and returns
    slightly different numbers from the ones the model was evaluated on.
    """
    for model in registry["models"]:
        export = model.get("onnx_export")
        if export is None:
            continue
        assert export["n_chips_verified"] > 0, f"{model['name']}: export never verified"
        assert isinstance(export["agrees_with_checkpoint"], bool)
        assert (REPO_ROOT / export["parity_report"]).is_file()
