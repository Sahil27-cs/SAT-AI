"""Score a trained flood model on a fold's TEST region.

    python -m ml.flood.evaluate --checkpoint models/flood/flood_unet_loro_india_sar_ratio_best.pt

This produces the number that may be reported. Training prints a *validation*
IoU, and validation is a different held-out region chosen to steer early
stopping -- quoting it as the model's score reports the region the checkpoint
was selected on, which is the mistake this file exists to prevent.

Every guard here is about keeping that separation intact:

* the fold is rebuilt from the checkpoint's recorded fold name, not from a
  command-line argument, so the test region cannot drift from the one the model
  was trained against;
* normalisation is restored from the checkpoint rather than refitted, because
  refitting on test data would erase the domain shift being measured, and
  ``StackNormalizer.assert_fold`` refuses statistics from another fold;
* confusion counts are pooled across chips, never averaged, so a chip with
  three water pixels does not weigh as much as one that is half flooded;
* a threshold sweep is reported, with the operating point fixed at 0.5. Picking
  the threshold that maximises test IoU would be tuning on the test set, so the
  sweep is published as a curve and the headline uses the a-priori 0.5.
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
from torch.utils.data import DataLoader

from ml.flood.dataset import FloodBatch, FloodChips, selection_for_bands
from ml.flood.model import UNet, UNetSpec
from ml.flood.train import select_fold
from satai.ml.metrics import SegmentationMetrics, aggregate, evaluate
from satai.preprocessing.normalize import BandStats, StackNormalizer

DEFAULT_DATA = REPO_ROOT / "data" / "raw" / "sen1floods11"
DEFAULT_OUT = REPO_ROOT / "ml" / "experiments" / "flood_unet"

#: Reported as a curve, never used to pick the headline. See the module docstring.
SWEEP_THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[UNet, StackNormalizer, dict[str, Any]]:
    """Rebuild the exact model and transform the checkpoint was saved with."""
    if not path.is_file():
        raise SystemExit(f"checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)
    spec = UNetSpec(**payload["spec"])
    model = UNet(spec).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    stats = {name: BandStats(**values) for name, values in payload["normalizer"]["bands"].items()}
    normalizer = StackNormalizer(fold=payload["normalizer"]["fold"], stats=stats)
    return model, normalizer, payload


@torch.no_grad()
def score(
    model: UNet, loader: DataLoader[FloodBatch], device: torch.device, thresholds: tuple[float, ...]
) -> tuple[dict[float, SegmentationMetrics], list[dict[str, Any]]]:
    """Pooled metrics at each threshold, plus a per-chip record."""
    per_threshold: dict[float, list[SegmentationMetrics]] = {t: [] for t in thresholds}
    per_chip: list[dict[str, Any]] = []

    for features, target, valid in loader:
        probability = torch.sigmoid(model(features.to(device))).cpu().numpy()
        truth = target.numpy()
        mask = valid.numpy().astype(bool)

        for i in range(probability.shape[0]):
            if not mask[i, 0].any():
                continue
            for threshold in thresholds:
                metrics = evaluate(
                    probability[i, 0], truth[i, 0], valid=mask[i, 0], threshold=threshold
                )
                per_threshold[threshold].append(metrics)
                if threshold == 0.5:
                    per_chip.append(
                        {
                            "iou": round(metrics.iou, 4),
                            "f1": round(metrics.f1, 4),
                            "positive_rate": round(metrics.positive_rate, 4),
                            "n_valid": metrics.n_valid,
                        }
                    )

    return {t: aggregate(m) for t, m in per_threshold.items()}, per_chip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--partition",
        default="test",
        choices=("test", "val"),
        help="test is the reportable number; val is only for diagnosing a run.",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model, normalizer, payload = load_checkpoint(args.checkpoint, device)

    # The fold comes from the checkpoint, not from an argument: an evaluation
    # run pointed at the wrong fold would silently score the model on regions it
    # was trained on.
    fold_name = payload["fold"]
    fold = select_fold(args.data_root, fold_name)
    normalizer.assert_fold(fold_name)

    chips = list(fold.test if args.partition == "test" else fold.val)
    regions = sorted({c.region for c in chips})
    selection = selection_for_bands(payload["bands"])
    with_ratio = selection

    dataset = FloodChips(chips, args.data_root, normalizer, with_ratio=with_ratio)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    print(f"checkpoint : {args.checkpoint.name}")
    print(f"fold       : {fold_name}  ({fold.summary()})")
    print(f"partition  : {args.partition}  regions {', '.join(regions)}  {len(dataset)} chips")
    print(f"bands      : {', '.join(payload['bands'])}")
    print(f"trained to : epoch {payload['epoch']}, best val IoU {payload['best_iou']:.4f}")
    print()

    pooled, per_chip = score(model, loader, device, SWEEP_THRESHOLDS)
    headline = pooled[0.5]

    print("threshold sweep (published as a curve; the headline uses 0.5):")
    for threshold in SWEEP_THRESHOLDS:
        metrics = pooled[threshold]
        marker = "  <- operating point" if threshold == 0.5 else ""
        print(
            f"  {threshold:.1f}  IoU {metrics.iou:.4f}  F1 {metrics.f1:.4f}  "
            f"P {metrics.precision:.4f}  R {metrics.recall:.4f}{marker}"
        )

    ious = [c["iou"] for c in per_chip]
    report = {
        "experiment": "flood_unet_test",
        "run_at": datetime.now(UTC).isoformat(),
        "checkpoint": relative_to_repo(args.checkpoint),
        "fold": fold_name,
        "partition": args.partition,
        "protocol": "leave_one_region_out",
        "region_disjoint": fold.is_region_disjoint,
        "regions": regions,
        "n_chips": len(dataset),
        "bands": payload["bands"],
        "band_config": selection.tag,
        "selection": {
            "checkpoint_selected_on": list(fold.val_regions),
            "note": (
                "The checkpoint was selected by validation IoU on a different "
                "held-out region. The scores below are on the test region, which "
                "influenced neither training nor selection."
            ),
        },
        "headline_threshold": 0.5,
        "headline": {
            "iou": round(headline.iou, 4),
            "dice": round(headline.dice, 4),
            "f1": round(headline.f1, 4),
            "precision": round(headline.precision, 4),
            "recall": round(headline.recall, 4),
            "specificity": round(headline.specificity, 4),
            "accuracy": round(headline.accuracy, 4),
        },
        "confusion": {"tp": headline.tp, "fp": headline.fp, "fn": headline.fn, "tn": headline.tn},
        "per_chip_iou": {
            "median": round(float(np.median(ious)), 4) if ious else None,
            "p25": round(float(np.percentile(ious, 25)), 4) if ious else None,
            "p75": round(float(np.percentile(ious, 75)), 4) if ious else None,
            "min": round(min(ious), 4) if ious else None,
            "max": round(max(ious), 4) if ious else None,
        },
        "threshold_sweep": {
            f"{t:.1f}": {
                "iou": round(pooled[t].iou, 4),
                "f1": round(pooled[t].f1, 4),
                "precision": round(pooled[t].precision, 4),
                "recall": round(pooled[t].recall, 4),
            }
            for t in SWEEP_THRESHOLDS
        },
        "caveats": [
            "Trained on Sen1Floods11, 11 flood events. Generalisation beyond "
            "those regimes is untested.",
            "Known failure mode: urban double-bounce RAISES backscatter over "
            "flooded streets, inverting the signature this model relies on. "
            "Contribution C3 measures that degradation directly.",
            "The threshold sweep is reported as a curve. Selecting the threshold "
            "that maximises test IoU would be tuning on the test set; the "
            "headline uses the a-priori 0.5.",
        ],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    # The band configuration is part of the filename. Without it a SAR-only run
    # silently overwrites the SAR+ratio result for the same fold, which is
    # exactly how a modality ablation loses the arm it was comparing against.
    band_tag = selection.tag
    destination = args.out / f"{args.partition}_{fold_name}_{band_tag}.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(
        f"TEST on {', '.join(regions)} @ 0.5: IoU {headline.iou:.4f}  F1 {headline.f1:.4f}  "
        f"P {headline.precision:.4f}  R {headline.recall:.4f}"
    )
    if ious:
        print(
            f"  per-chip IoU  median {np.median(ious):.3f}  "
            f"IQR [{np.percentile(ious, 25):.3f}, {np.percentile(ious, 75):.3f}]"
        )
    print(f"written to {relative_to_repo(destination)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
