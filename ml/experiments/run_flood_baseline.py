"""Score the classical flood baseline on Sen1Floods11, region by region.

    python ml/experiments/run_flood_baseline.py --chips data/raw/sen1floods11

This is the measurement every deep-model claim in this project has to beat.
Otsu thresholding of SAR backscatter is the standard operational flood-mapping
method: unsupervised, fast, and genuinely strong. A U-Net that cannot beat it
has demonstrated nothing, and most student flood projects never find that out
because they never build the baseline.

No GPU and no training. It needs only the chips, numpy and rasterio, which is
why it is the first real number this project can produce.

**What this run does NOT include.** Sen1Floods11's hand-labelled release ships
no HAND (height above nearest drainage) raster, so the terrain mask that
normally removes Otsu's false positives is absent here. Tarmac, dry sand and
smooth bare soil scatter radar away from the sensor exactly like water does, and
without terrain there is nothing to tell them apart. The scores below are
therefore a **lower bound** on what the method achieves operationally, and they
are reported as Otsu-without-HAND rather than as "the Otsu+HAND baseline".

Results are reported per region under leave-one-region-out, because a pooled
number over all 11 events hides exactly the variation the project cares about.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from satai.paths import REPO_ROOT
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

import numpy as np

from satai.errors import ValidationError
from satai.ml.baseline import OtsuHandBaseline
from satai.ml.metrics import (
    SegmentationMetrics,
    aggregate,
    evaluate,
    valid_mask_from_labels,
)
from satai.preprocessing.splits import parse_chip_name

OUTPUT = REPO_ROOT / "ml" / "experiments" / "flood_baseline"


def load_chip(s1_path: Path, label_path: Path) -> tuple[Any, Any]:
    """Read one chip's backscatter stack and its hand-drawn mask."""
    import rasterio

    with rasterio.open(s1_path) as src:
        features = src.read().astype(np.float64)
    with rasterio.open(label_path) as src:
        labels = src.read(1)
    return features, labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chips", type=Path, default=REPO_ROOT / "data/raw/sen1floods11")
    parser.add_argument("--band", type=int, default=0, help="0 = VV, 1 = VH")
    parser.add_argument(
        "--min-separability",
        type=float,
        default=0.75,
        help="Below this, the histogram is treated as unimodal and no water is predicted.",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    s1_dir, label_dir = args.chips / "S1Hand", args.chips / "LabelHand"
    if not s1_dir.is_dir() or not label_dir.is_dir():
        raise ValidationError(
            f"expected S1Hand/ and LabelHand/ under {args.chips}. "
            f"Run: python scripts/download_sen1floods11.py"
        )

    baseline = OtsuHandBaseline(band_index=args.band, min_separability=args.min_separability)
    per_region: dict[str, list[SegmentationMetrics]] = defaultdict(list)
    unimodal: dict[str, int] = defaultdict(int)
    skipped = 0

    paths = sorted(s1_dir.glob("*_S1Hand.tif"))
    print(f"scoring {len(paths)} chips (Otsu on band {args.band}, no HAND mask)")

    for index, s1_path in enumerate(paths, start=1):
        chip = parse_chip_name(s1_path.name)
        label_path = label_dir / f"{chip.region}_{chip.chip_id}_LabelHand.tif"
        if not label_path.is_file():
            skipped += 1
            continue

        features, labels = load_chip(s1_path, label_path)
        # Sen1Floods11 marks unannotated pixels -1. They are excluded from the
        # confusion counts rather than treated as dry: scoring a pixel nobody
        # labelled is scoring a guess against a guess.
        valid = valid_mask_from_labels(labels).astype(bool)
        if not valid.any():
            skipped += 1
            continue

        # Border no-data in Sen1Floods11 is a large negative sentinel; left in,
        # it drags the histogram down until the water mode disappears.
        finite = np.isfinite(features[args.band]) & (features[args.band] > -50.0)
        probability = baseline.predict(features, valid=finite)

        if baseline.separability_ is not None and (baseline.separability_ < args.min_separability):
            unimodal[chip.region] += 1

        metrics = evaluate(probability, np.maximum(labels, 0), valid=valid, threshold=0.5)
        per_region[chip.region].append(metrics)

        if index % 50 == 0:
            print(f"  {index}/{len(paths)}", end="\r", flush=True)

    report: dict[str, Any] = {
        "experiment": "flood_baseline_otsu",
        "run_at": datetime.now(UTC).isoformat(),
        "method": f"Otsu threshold on band {args.band} (VV), no HAND terrain mask",
        "dataset": "Sen1Floods11 v1.1 HandLabeled",
        "n_chips_scored": sum(len(v) for v in per_region.values()),
        "n_chips_skipped": skipped,
        "min_separability": args.min_separability,
        "caveats": [
            "No HAND raster ships with Sen1Floods11 HandLabeled, so the terrain "
            "mask that removes Otsu's false positives over tarmac, dry sand and "
            "smooth bare soil is absent. These are a LOWER BOUND on the method.",
            "Unannotated pixels (label -1) are excluded from the confusion counts.",
            "Per-region scores are pooled over chips by summing confusion counts, "
            "not by averaging per-chip scores -- averaging lets a chip with three "
            "water pixels weigh as much as one that is half flooded.",
        ],
        "per_region": {},
    }

    for region in sorted(per_region):
        pooled = aggregate(per_region[region])
        report["per_region"][region] = {
            "n_chips": len(per_region[region]),
            "iou": round(pooled.iou, 4),
            "f1": round(pooled.f1, 4),
            "precision": round(pooled.precision, 4),
            "recall": round(pooled.recall, 4),
            "chips_called_unimodal": unimodal.get(region, 0),
            # Raw counts, so a later run can pool regions exactly rather than
            # averaging their scores. Averaging IoUs across regions weights a
            # 15-chip region equally with a 69-chip one, which is a different
            # and worse statistic than the pooled number.
            "tp": pooled.tp,
            "fp": pooled.fp,
            "fn": pooled.fn,
            "tn": pooled.tn,
        }

    everything = [m for group in per_region.values() for m in group]
    overall = aggregate(everything)
    report["pooled"] = {
        "iou": round(overall.iou, 4),
        "f1": round(overall.f1, 4),
        "precision": round(overall.precision, 4),
        "recall": round(overall.recall, 4),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "otsu_baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    header = (
        f"\n{'region':<12} {'chips':>5} {'IoU':>7} {'F1':>7} "
        f"{'prec':>7} {'recall':>7} {'unimod':>7}"
    )
    print(header)
    for region, row in report["per_region"].items():
        print(
            f"{region:<12} {row['n_chips']:>5} {row['iou']:>7.3f} {row['f1']:>7.3f} "
            f"{row['precision']:>7.3f} {row['recall']:>7.3f} {row['chips_called_unimodal']:>7}"
        )
    print(
        f"{'POOLED':<12} {report['n_chips_scored']:>5} {overall.iou:>7.3f} {overall.f1:>7.3f} "
        f"{overall.precision:>7.3f} {overall.recall:>7.3f}"
    )
    print(f"\nwritten to {args.out / 'otsu_baseline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
