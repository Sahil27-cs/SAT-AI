"""Export the trained flood model to ONNX, and prove the export still agrees.

    python -m ml.flood.export \
        --checkpoint models/flood/flood_unet_loro_india_sar_ratio_best.pt \
        --chips data/raw/sen1floods11 --region India --limit 4

Why an export at all: the serving plane runs on CPU in a serverless function
with a size budget, and `torch` alone is an order of magnitude larger than that
budget allows. ONNX plus `onnxruntime` is what makes CPU inference deployable
at all. The export also fixes the graph, so the thing served is the thing that
was measured rather than whatever `ml/flood/model.py` happens to say later.

**The export is not the point; the agreement is.** A silently wrong export is
worse than no export, because it serves numbers that look like the evaluated
model's and are not. So this command does not finish on a written file. It runs
both the checkpoint and the exported graph over real chips and reports the
largest disagreement it found, in probability units and in the flood mask that
results. If `onnxruntime` is not installed it says BLOCKED and writes nothing:
an unverified export is not a deliverable.

Dynamic height and width are declared deliberately. Sen1Floods11 chips are
512x512, but an AOI mosaic is not, and an export pinned to 512 would fail on
the first real scene while passing every test here.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from satai.paths import REPO_ROOT, relative_to_repo
except ImportError:  # pragma: no cover - outside an installed package
    _candidates: list[Path] = []
    if os.environ.get("SATAI_REPO_ROOT"):
        _candidates.append(Path(os.environ["SATAI_REPO_ROOT"]).expanduser().resolve())
    with contextlib.suppress(NameError):
        _candidates.append(Path(__file__).resolve().parents[2])
    _cwd = Path.cwd().resolve()
    _candidates += [_cwd, *_cwd.parents, *(p for p in sorted(_cwd.iterdir()) if p.is_dir())]
    REPO_ROOT = next((c for c in _candidates if (c / "satai" / "provenance.py").is_file()), _cwd)
    sys.path.insert(0, str(REPO_ROOT))

    # The root is on the path now, so the real helper is importable. Importing it
    # here rather than reimplementing it keeps one definition of what a recorded
    # path looks like.
    from satai.paths import relative_to_repo

import numpy as np
import torch

from ml.flood.dataset import bands_for, build_reader, stack_with_ratio
from ml.flood.evaluate import load_checkpoint

DEFAULT_MODELS = REPO_ROOT / "models" / "flood"

#: Tolerance on the maximum per-pixel probability difference between the
#: checkpoint and the exported graph. 1e-4 is two orders of magnitude below the
#: 0.01 that could move a pixel across the 0.5 decision boundary in any way a
#: reader would notice, and well above float32 accumulation noise in a 4-level
#: U-Net. A disagreement above this is a broken export, not rounding.
PARITY_TOLERANCE = 1e-4

BLOCKED_NO_RUNTIME = (
    "BLOCKED: onnxruntime is not installed, so the export cannot be verified "
    "against the checkpoint. An unverified export would serve numbers that "
    "look like the evaluated model's without anything having checked that they "
    "are. Install it with: pip install onnxruntime"
)


def export_onnx(
    model: torch.nn.Module, destination: Path, *, n_bands: int, opset: int = 17
) -> Path:
    """Write the graph, with batch, height and width left dynamic."""
    model.eval()
    example = torch.zeros(1, n_bands, 64, 64, dtype=torch.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        str(destination),
        input_names=["bands"],
        output_names=["logits"],
        # `dynamic_axes`, not `dynamic_shapes`: torch >= 2.9 prefers the latter
        # and warns, but it does not exist on the older releases this repository
        # also has to export from. The warning is the portable choice's cost.
        dynamic_axes={
            "bands": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return destination


def _torch_probability(model: torch.nn.Module, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(features)[None])
    return torch.sigmoid(logits)[0, 0].numpy()


def _onnx_probability(session: Any, features: np.ndarray) -> np.ndarray:
    logits = session.run(["logits"], {"bands": features[None].astype(np.float32)})[0]
    probability: np.ndarray = 1.0 / (1.0 + np.exp(-logits[0, 0]))
    return probability


def compare(
    model: torch.nn.Module, session: Any, chips: list[np.ndarray]
) -> tuple[float, float, list[dict[str, Any]]]:
    """Largest probability and mask disagreement over `chips`.

    The mask figure is the one that matters operationally: two probabilities
    either side of 0.5 by a hair produce different polygons, and that is what a
    reader would see.
    """
    per_chip: list[dict[str, Any]] = []
    worst_probability = 0.0
    worst_mask = 0.0

    for index, features in enumerate(chips):
        reference = _torch_probability(model, features)
        exported = _onnx_probability(session, features)

        probability_delta = float(np.abs(reference - exported).max())
        mask_delta = float(((reference >= 0.5) != (exported >= 0.5)).mean())
        worst_probability = max(worst_probability, probability_delta)
        worst_mask = max(worst_mask, mask_delta)

        per_chip.append(
            {
                "chip_index": index,
                "shape": list(reference.shape),
                "max_abs_probability_delta": probability_delta,
                "fraction_of_pixels_crossing_the_threshold": mask_delta,
            }
        )
    return worst_probability, worst_mask, per_chip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chips", type=Path, default=REPO_ROOT / "data/raw/sen1floods11")
    parser.add_argument("--region", default=None)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    try:
        import onnxruntime
    except ImportError:
        print(BLOCKED_NO_RUNTIME)
        return 2

    model, normalizer, payload = load_checkpoint(args.checkpoint, torch.device("cpu"))
    with_ratio = "vv_vh_ratio" in payload["bands"]
    bands = bands_for(with_ratio)
    band_tag = "sar_ratio" if with_ratio else "sar"

    destination = args.out or DEFAULT_MODELS / f"flood_unet_{payload['fold']}_{band_tag}.onnx"
    export_onnx(model, destination, n_bands=len(bands), opset=args.opset)
    size_mb = destination.stat().st_size / 1_048_576
    print(f"exported {relative_to_repo(destination)}  ({size_mb:.1f} MB)")

    reader = build_reader(args.chips)
    available = reader.available(reader.discover())
    if args.region:
        available = [c for c in available if c.region == args.region]
    available = available[: args.limit] if args.limit else available

    prepared: list[np.ndarray] = []
    for chip in available:
        features, _labels, _valid = reader.load(chip)
        stacked = normalizer.transform_stack(stack_with_ratio(features, with_ratio), list(bands))
        prepared.append(np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

    if not prepared:
        print(
            "BLOCKED: no chips available to verify the export against. The file "
            "was written but nothing has checked that it agrees with the "
            "checkpoint, so it must not be served."
        )
        return 2

    session = onnxruntime.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    worst_probability, worst_mask, per_chip = compare(model, session, prepared)
    agrees = worst_probability <= PARITY_TOLERANCE

    report = {
        "experiment": "flood_onnx_export",
        "run_at": datetime.now(UTC).isoformat(),
        "checkpoint": relative_to_repo(args.checkpoint),
        "artifact": relative_to_repo(destination),
        "artifact_mb": round(size_mb, 2),
        "opset": args.opset,
        "onnxruntime_version": onnxruntime.__version__,
        "torch_version": torch.__version__,
        "bands": list(bands),
        "dynamic_axes": ["batch", "height", "width"],
        "n_chips_verified": len(per_chip),
        "region": args.region,
        "tolerance": PARITY_TOLERANCE,
        "max_abs_probability_delta": worst_probability,
        "max_fraction_of_pixels_crossing_the_threshold": worst_mask,
        "agrees_with_checkpoint": agrees,
        "per_chip": per_chip,
        "caveats": [
            "Parity is measured on the chips listed above, not proved for all "
            "inputs. It is evidence that the export is faithful, not a theorem.",
            "The exported graph emits logits. Whatever serves it must apply the "
            "sigmoid and the 0.5 threshold itself, exactly as evaluate.py does.",
            "Normalisation is NOT part of the graph: it needs the fold's own "
            "normalizer, and baking one fold's statistics into the file would "
            "make the artifact silently wrong for every other fold.",
        ],
    }
    report_path = destination.with_suffix(".parity.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"  chips verified          {len(per_chip)}")
    print(f"  max probability delta   {worst_probability:.3e}  (tolerance {PARITY_TOLERANCE:.0e})")
    print(f"  pixels crossing 0.5     {worst_mask:.6%}")
    print(f"  agrees with checkpoint  {agrees}")
    print(f"\nwritten to {relative_to_repo(report_path)}")

    if not agrees:
        print(
            "\nThe export does NOT agree with the checkpoint within tolerance. "
            "It must not be served: it would produce numbers that differ from "
            "the ones the model was evaluated on."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
