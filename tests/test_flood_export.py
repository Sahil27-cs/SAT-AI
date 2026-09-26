"""The ONNX export must be the same model, not a similar one.

The serving plane cannot carry `torch`, so CPU inference means an exported
graph. That makes the export a place where a defect is invisible: the file
loads, the shapes are right, the map renders, and the numbers are not the ones
the model was evaluated on. Reported IoU would then belong to a model nobody is
running.

So the export is checked the only way that means anything -- by running both and
comparing. These cases use a deliberately small U-Net rather than the trained
checkpoint, so the suite stays fast and passes on a clone with no weights, and
`ml/flood/export.py` runs the same comparison against the real checkpoint (its
parity report is committed alongside the artifact).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.flood.export import (  # noqa: E402
    BLOCKED_NO_RUNTIME,
    PARITY_TOLERANCE,
    compare,
    export_onnx,
)
from ml.flood.model import UNet, UNetSpec  # noqa: E402

onnxruntime = pytest.importorskip(
    "onnxruntime", reason="the export cannot be verified without a runtime"
)

N_BANDS = 3


@pytest.fixture(scope="module")
def model() -> UNet:
    """A small but structurally real U-Net: pooling, skips, transposed convs.

    depth=2 keeps the export fast while still exercising the parts that go
    wrong -- an interpolation or a concatenation that ONNX traces differently.
    """
    torch.manual_seed(0)
    net = UNet(UNetSpec(in_channels=N_BANDS, base_width=4, depth=2, dropout=0.0))
    return net.eval()


@pytest.fixture(scope="module")
def exported(model: UNet, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("onnx") / "tiny.onnx"
    return export_onnx(model, path, n_bands=N_BANDS)


@pytest.fixture(scope="module")
def session(exported: Path) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(str(exported), providers=["CPUExecutionProvider"])


def test_the_export_writes_a_loadable_graph(exported: Path) -> None:
    assert exported.is_file()
    assert exported.stat().st_size > 0


def test_the_exported_graph_agrees_with_the_checkpoint(
    model: UNet, session: onnxruntime.InferenceSession
) -> None:
    rng = np.random.default_rng(0)
    chips = [rng.normal(size=(N_BANDS, 64, 64)).astype(np.float32) for _ in range(3)]

    worst_probability, worst_mask, per_chip = compare(model, session, chips)

    assert len(per_chip) == 3
    assert worst_probability <= PARITY_TOLERANCE
    assert worst_mask == 0.0


def test_parity_is_measured_on_probabilities_not_logits(
    model: UNet, session: onnxruntime.InferenceSession
) -> None:
    """A logit difference of 1e-3 is a probability difference of ~2.5e-4.

    Comparing logits would report a number that is not the one a reader sees,
    and the threshold column below would not follow from it.
    """
    rng = np.random.default_rng(1)
    chips = [rng.normal(size=(N_BANDS, 32, 32)).astype(np.float32)]
    _, _, per_chip = compare(model, session, chips)
    assert per_chip[0]["max_abs_probability_delta"] <= 1.0


def test_height_and_width_are_dynamic(session: onnxruntime.InferenceSession) -> None:
    """Chips are 512x512; an AOI mosaic is not.

    An export pinned to the training size passes every test written against
    Sen1Floods11 and fails on the first real scene.
    """
    for size in (32, 48, 96):
        logits = session.run(
            ["logits"], {"bands": np.zeros((1, N_BANDS, size, size), dtype=np.float32)}
        )[0]
        assert logits.shape == (1, 1, size, size)


def test_batches_larger_than_one_are_accepted(session: onnxruntime.InferenceSession) -> None:
    logits = session.run(["logits"], {"bands": np.zeros((4, N_BANDS, 32, 32), dtype=np.float32)})[0]
    assert logits.shape == (4, 1, 32, 32)


def test_the_graph_emits_logits_and_leaves_the_sigmoid_to_the_caller(
    model: UNet, session: onnxruntime.InferenceSession, exported: Path
) -> None:
    """Whatever serves this must apply the sigmoid and the 0.5 cut itself.

    If the graph ever starts emitting probabilities, a serving path that also
    applies a sigmoid would squash every prediction toward 0.5 -- and would
    still produce a plausible-looking map. Checked on the graph rather than on
    the output range: an untrained network's logits can happen to fall inside
    (0, 1), so the range proves nothing either way.
    """
    onnx = pytest.importorskip("onnx")
    graph = onnx.load(str(exported))
    assert "Sigmoid" not in {node.op_type for node in graph.graph.node}

    features = np.random.default_rng(2).normal(size=(1, N_BANDS, 32, 32)).astype(np.float32)
    logits = session.run(["logits"], {"bands": features})[0]
    with torch.no_grad():
        reference = model(torch.from_numpy(features)).numpy()
    assert np.abs(logits - reference).max() <= 1e-4


def test_the_blocked_message_names_its_dependency() -> None:
    """The command refuses to present an unverified export as a deliverable."""
    assert BLOCKED_NO_RUNTIME.startswith("BLOCKED")
    assert "onnxruntime" in BLOCKED_NO_RUNTIME


def test_the_committed_parity_report_still_passes_its_own_tolerance() -> None:
    """The real checkpoint's report, if this clone has one.

    Skipped rather than failed where weights are absent: they are gitignored,
    so a fresh clone legitimately has neither the model nor its report.
    """
    import json

    report = REPO_ROOT / "models" / "flood" / "flood_unet_loro_india_sar_ratio.parity.json"
    if not report.is_file():
        pytest.skip("no export report in this clone; run python -m ml.flood.export")

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["agrees_with_checkpoint"] is True
    assert payload["max_abs_probability_delta"] <= payload["tolerance"]
    assert payload["n_chips_verified"] > 0
