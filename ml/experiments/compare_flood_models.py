"""Compare the deep flood model against the classical baseline, honestly.

    python -m ml.experiments.compare_flood_models

The comparison this project exists to make. Otsu thresholding with a
unimodality guard is the standard operational method: unsupervised, fast and
genuinely strong. A U-Net that cannot beat it has demonstrated nothing, and
most student flood projects never find that out because they never build the
baseline.

Both numbers must come from the same held-out region under the same protocol,
or the comparison is decoration. This script refuses to emit a delta unless the
two reports agree on the region, and it reports the validation-to-test gap
beside the headline because that gap is a result in its own right.
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

EXPERIMENTS = REPO_ROOT / "ml" / "experiments"


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default="loro_india")
    parser.add_argument("--region", default="India")
    parser.add_argument("--out", type=Path, default=EXPERIMENTS / "flood_comparison.json")
    args = parser.parse_args(argv)

    unet = load(EXPERIMENTS / "flood_unet" / f"test_{args.fold}.json")
    baseline_all = load(EXPERIMENTS / "flood_baseline" / "sweep_0.66" / "otsu_baseline.json")
    selection = load(EXPERIMENTS / "flood_baseline" / "threshold_selection.json")

    if args.region not in unet["regions"]:
        raise SystemExit(
            f"the U-Net report covers {unet['regions']}, not {args.region!r}; "
            f"comparing across regions would not be a comparison"
        )
    if args.region not in baseline_all["per_region"]:
        raise SystemExit(f"no baseline result for region {args.region!r}")

    baseline = baseline_all["per_region"][args.region]
    deep = unet["headline"]
    delta = deep["iou"] - baseline["iou"]

    manifest_path = REPO_ROOT / "models" / "flood" / f"train_manifest_{args.fold}_sar_ratio.json"
    training = load(manifest_path) if manifest_path.is_file() else {}
    val_iou = training.get("best_val_iou")

    report = {
        "experiment": "flood_model_vs_baseline",
        "run_at": datetime.now(UTC).isoformat(),
        "protocol": "leave-one-region-out; both scored on the same held-out region",
        "test_region": args.region,
        "n_chips": unet["n_chips"],
        "baseline": {
            "method": "Otsu on Sentinel-1 VV, unimodality guard 0.66, no HAND mask",
            "supervised": False,
            "iou": baseline["iou"],
            "f1": baseline["f1"],
            "precision": baseline["precision"],
            "recall": baseline["recall"],
            "guard_selected_by": "leave-one-region-out over 11 regions",
            "guard_optimism": selection["headline"]["optimism_of_naive_protocol"],
        },
        "deep_model": {
            "method": "U-Net, 4 levels, base width 32, trained from scratch",
            "supervised": True,
            "bands": unet["bands"],
            "n_parameters": training.get("n_parameters"),
            "epochs_run": training.get("epochs_run"),
            "iou": deep["iou"],
            "f1": deep["f1"],
            "precision": deep["precision"],
            "recall": deep["recall"],
            "per_chip_iou_median": unet["per_chip_iou"]["median"],
            "per_chip_iou_iqr": [unet["per_chip_iou"]["p25"], unet["per_chip_iou"]["p75"]],
        },
        "delta_iou": round(delta, 4),
        "relative_gain": round(delta / baseline["iou"], 4) if baseline["iou"] else None,
        "verdict": (
            "The deep model beats the classical baseline on the held-out region."
            if delta > 0
            else "The deep model does NOT beat the classical baseline. Reported as a "
            "negative result; a model that cannot beat Otsu has demonstrated nothing."
        ),
        "validation_to_test_gap": {
            "validation_iou": val_iou,
            "validation_region": training.get("val_regions"),
            "test_iou": deep["iou"],
            "gap": round(val_iou - deep["iou"], 4) if val_iou is not None else None,
            "note": (
                "The checkpoint was selected on validation IoU over a different "
                "held-out region. The gap between that and the test score is the "
                "cost of treating any single held-out region as representative -- "
                "it is a property of the data, not a defect of the run, and "
                "quoting the validation number as the model's score would "
                "overstate it by exactly this much."
            ),
        },
        "caveats": [
            "Pooled IoU is dominated by chips with large water bodies. The "
            "per-chip median is far below the pooled figure, which means the "
            "model does well where there is a lot of water and poorly where "
            "there is little. Both numbers are reported; neither alone is honest.",
            "The baseline runs without a HAND terrain mask, because Sen1Floods11 "
            "ships none. With terrain it would score higher, so the margin here "
            "is an upper bound on the deep model's advantage.",
            "Neither model has seen an urban Indian scene. Contribution C3 "
            "measures that transfer, and it is not measured by this comparison.",
        ],
    }

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"held-out region: {args.region}  ({unet['n_chips']} chips, region-disjoint)")
    print()
    print(f"  {'':<22} {'IoU':>7} {'F1':>7} {'prec':>7} {'recall':>7}")
    print(
        f"  {'Otsu baseline':<22} {baseline['iou']:>7.4f} {baseline['f1']:>7.4f} "
        f"{baseline['precision']:>7.4f} {baseline['recall']:>7.4f}"
    )
    print(
        f"  {'U-Net (SAR + ratio)':<22} {deep['iou']:>7.4f} {deep['f1']:>7.4f} "
        f"{deep['precision']:>7.4f} {deep['recall']:>7.4f}"
    )
    print(f"  {'delta IoU':<22} {delta:>+7.4f}")
    print()
    print(f"  {report['verdict']}")
    if val_iou is not None:
        print(
            f"  validation {val_iou:.4f} on {training.get('val_regions')} vs test "
            f"{deep['iou']:.4f} on {args.region}: gap {val_iou - deep['iou']:+.4f}"
        )
    print(
        f"  per-chip IoU median {unet['per_chip_iou']['median']:.3f} "
        f"(pooled is dominated by large-water chips)"
    )
    print(f"\nwritten to {os.path.relpath(args.out, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
