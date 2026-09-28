"""Explain the flood model: per-band attribution and occlusion sensitivity.

    python -m ml.flood.explain --checkpoint models/flood/flood_unet_loro_india_sar_ratio_best.pt \
        --chips data/raw/sen1floods11 --region India --limit 6

Two methods, chosen for what a segmentation network actually needs:

``integrated_gradients``
    Attribution of the predicted water probability to each input band, along a
    straight path from a baseline to the real input. Answers "which band moved
    this prediction", which is the question the modality-ablation contribution
    (C2) asks at the dataset level and this asks per chip.

``occlusion``
    Blank a band and measure how far the prediction moves. Slower and cruder
    than gradients, and included precisely because it is *model-agnostic* --
    it needs no gradient and cannot be fooled by a saturated activation, so
    when the two methods disagree that disagreement is information.

**Grad-CAM is deliberately absent.** It localises which spatial region drove a
*classification*. A segmentation network's output is already spatially resolved
-- a per-pixel probability map -- so a Grad-CAM heatmap over it answers a
question the model has already answered, at coarser resolution. Attribution
here is over *bands*, which is the axis the output does not already expose.

What these do not tell you
--------------------------
Attribution describes what the model used, not what caused the flood. A band
with high attribution is one the network leaned on, which may be because it
carries signal or because it correlates with something that does. Every
artifact written here carries that caveat, and the interface must present it as
MODEL EXPLANATION rather than as an explanation of the world.
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

from ml.flood.dataset import BandSelection, build_reader, selection_for_bands, stack_with_ratio
from ml.flood.evaluate import load_checkpoint

DEFAULT_OUT = REPO_ROOT / "ml" / "experiments" / "flood_xai"


def _prepare(
    reader: Any,
    chip: Any,
    normalizer: Any,
    bands: tuple[str, ...],
    with_ratio: bool | BandSelection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, valid = reader.load(chip)
    stacked = stack_with_ratio(features, with_ratio)
    normalised = normalizer.transform_stack(stacked, list(bands))
    normalised = np.nan_to_num(normalised, nan=0.0, posinf=0.0, neginf=0.0)
    return normalised.astype(np.float32), labels, valid


def integrated_gradients(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    valid: torch.Tensor,
    *,
    steps: int = 32,
) -> np.ndarray:
    """Per-band attribution of the mean predicted water probability.

    The baseline is the all-zero normalised input, which after standardisation
    is the per-band *mean* -- a genuinely uninformative reference rather than an
    arbitrary one. Attribution is (input - baseline) times the path-averaged
    gradient, summed over the spatial axes, restricted to annotated pixels.

    Returns one signed value per band: positive means the band pushed the
    prediction toward water.
    """
    baseline = torch.zeros_like(tensor)
    difference = tensor - baseline
    accumulated = torch.zeros_like(tensor)

    for step in range(1, steps + 1):
        # Riemann midpoint rather than left endpoint: with 32 steps it is a
        # strictly better approximation for the same cost, and the completeness
        # property integrated gradients relies on is an integral, not a sum.
        alpha = (step - 0.5) / steps
        point = (baseline + alpha * difference).detach().requires_grad_(True)
        logits = model(point)
        # Scalar target: mean probability over annotated pixels. Summing logits
        # instead would let a confidently-dry region dominate the gradient.
        target = (torch.sigmoid(logits) * valid).sum() / valid.sum().clamp(min=1.0)
        (gradient,) = torch.autograd.grad(target, point)
        accumulated += gradient.detach()

    attribution = (difference * accumulated / steps)[0]
    per_band = (attribution * valid[0]).sum(dim=(1, 2))
    return per_band.cpu().numpy()


@torch.no_grad()
def occlusion(
    model: torch.nn.Module, tensor: torch.Tensor, valid: torch.Tensor
) -> tuple[np.ndarray, float]:
    """Change in mean predicted probability when each band is blanked.

    Blanking means setting the band to zero, which post-standardisation is its
    mean -- so this measures the cost of replacing a band with "no information"
    rather than with noise. Returns one delta per band and the unmodified
    baseline probability.
    """

    def mean_probability(x: torch.Tensor) -> float:
        probability = torch.sigmoid(model(x))
        return float((probability * valid).sum() / valid.sum().clamp(min=1.0))

    reference = mean_probability(tensor)
    deltas = []
    for band in range(tensor.shape[1]):
        occluded = tensor.clone()
        occluded[:, band] = 0.0
        deltas.append(mean_probability(occluded) - reference)
    return np.array(deltas), reference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chips", type=Path, default=REPO_ROOT / "data/raw/sen1floods11")
    parser.add_argument("--region", default=None)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model, normalizer, payload = load_checkpoint(args.checkpoint, device)
    selection = selection_for_bands(payload["bands"])
    with_ratio = selection
    bands = selection.bands

    reader = build_reader(args.chips)
    chips = reader.available(reader.discover())
    if args.region:
        chips = [c for c in chips if c.region == args.region]
    chips = chips[: args.limit] if args.limit else chips
    if not chips:
        raise SystemExit(f"no chips matched region={args.region!r}")

    print(f"explaining {len(chips)} chip(s) with {len(bands)} bands: {', '.join(bands)}")

    ig_rows: list[np.ndarray] = []
    occ_rows: list[np.ndarray] = []
    per_chip: list[dict[str, Any]] = []

    for index, chip in enumerate(chips, start=1):
        features, _labels, valid_mask = _prepare(reader, chip, normalizer, bands, with_ratio)
        if not valid_mask.any():
            continue

        tensor = torch.from_numpy(features)[None].to(device)
        valid = torch.from_numpy(valid_mask.astype(np.float32))[None, None].to(device)

        ig = integrated_gradients(model, tensor, valid, steps=args.steps)
        occ, reference = occlusion(model, tensor, valid)
        ig_rows.append(ig)
        occ_rows.append(occ)

        per_chip.append(
            {
                "chip": chip.key,
                "mean_probability": round(reference, 4),
                "integrated_gradients": {
                    b: round(float(v), 6) for b, v in zip(bands, ig, strict=True)
                },
                "occlusion_delta": {b: round(float(v), 6) for b, v in zip(bands, occ, strict=True)},
            }
        )
        print(f"  {index}/{len(chips)}", end="\r", flush=True)

    if not per_chip:
        raise SystemExit("no chip had annotated pixels; nothing to explain")

    ig_mean = np.mean(ig_rows, axis=0)
    occ_mean = np.mean(occ_rows, axis=0)
    # Normalised so bands are comparable across chips of different sizes; the
    # sign is preserved because direction is the point.
    ig_share = ig_mean / (np.abs(ig_mean).sum() or 1.0)

    report = {
        "experiment": "flood_xai",
        "run_at": datetime.now(UTC).isoformat(),
        "checkpoint": relative_to_repo(args.checkpoint),
        "model": "flood_unet",
        "model_version": f"1.0.0+{payload['fold']}",
        "source_kind": "derived",
        "region": args.region,
        "n_chips": len(per_chip),
        "bands": list(bands),
        "methods": {
            "integrated_gradients": {
                "steps": args.steps,
                "baseline": "all-zero normalised input, i.e. the per-band mean",
                "target": "mean predicted water probability over annotated pixels",
            },
            "occlusion": {
                "operation": "set one band to its mean, measure the shift in mean probability"
            },
        },
        "mean_integrated_gradients": {
            b: round(float(v), 6) for b, v in zip(bands, ig_mean, strict=True)
        },
        "attribution_share": {b: round(float(v), 4) for b, v in zip(bands, ig_share, strict=True)},
        "mean_occlusion_delta": {
            b: round(float(v), 6) for b, v in zip(bands, occ_mean, strict=True)
        },
        "per_chip": per_chip,
        # Attribution and necessity are different questions, and on this model
        # they give different answers. Computed rather than asserted so the
        # cross-reference cannot go stale if either side is re-run.
        "cross_reference_c2": {
            "note": (
                "High attribution does not mean necessary. C2 removed the "
                "highest-attributed band and measured the cost; compare the "
                "share below against the delta C2 reports."
            ),
            "highest_attributed_band": max(
                zip(bands, np.abs(ig_share), strict=True), key=lambda kv: kv[1]
            )[0],
            "c2_report": "ml/experiments/c2_modality_ablation.json",
        },
        "methods_disagree_on": [
            band
            for band, ig_value, occ_value in zip(bands, ig_mean, occ_mean, strict=True)
            if (ig_value > 0) != (occ_value > 0)
        ],
        "caveats": [
            "Attribution describes what the model used, not physical causation. "
            "A band may rank high because it carries signal or because it "
            "correlates with something that does.",
            "This is a MODEL EXPLANATION. It is not an explanation of the flood, "
            "and it is not the natural-language summary the assistant produces.",
            "Integrated gradients and occlusion answer slightly different "
            "questions and can disagree. Where they do, that disagreement is "
            "information about the model, not an error in either method.",
            "Grad-CAM is not used: a segmentation model already outputs a "
            "per-pixel map, so a coarse spatial heatmap over it adds nothing.",
        ],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"xai_{payload['fold']}_{args.region or 'all'}.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n  {'band':<14} {'IG':>12} {'share':>8} {'occlusion':>12}")
    for band, ig_value, share, occ_value in zip(bands, ig_mean, ig_share, occ_mean, strict=True):
        print(f"  {band:<14} {ig_value:>12.4f} {share:>7.1%} {occ_value:>12.4f}")
    print(f"\nwritten to {relative_to_repo(destination)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
