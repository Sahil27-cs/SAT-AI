"""Select the Otsu separability guard leave-one-region-out, and report the curve.

    # once per guard value:
    python ml/experiments/run_flood_baseline.py --min-separability <s> --out sweep_<s>
    # then:
    python ml/experiments/select_baseline_threshold.py

Why this script exists
----------------------
Sweeping the guard over all 446 chips and reporting the best score would be
test-set tuning. The number that comes out of that is not a generalisation
estimate, it is the maximum of a curve fitted to the thing being measured, and
reporting it as "the baseline achieves IoU 0.416" would be exactly the kind of
inflated positive this project sets out not to produce.

So the threshold is chosen the same way the model will be evaluated: for each
region in turn, pick the guard value that maximises pooled IoU over the *other
ten regions*, then score the held-out region with it. The region being reported
never influenced the choice. The gap between that number and the all-data
optimum is itself the interesting quantity -- it is how much the naive protocol
would have overstated the result.

Confusion counts are pooled, never averaged. Averaging per-region IoU weights
Bolivia's 15 chips equally with the USA's 69.
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

BASELINE_DIR = REPO_ROOT / "ml" / "experiments" / "flood_baseline"


def _iou(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    return tp / denominator if denominator else 0.0


def load_sweep(root: Path) -> dict[float, dict[str, Any]]:
    """Every sweep_<s>/otsu_baseline.json under ``root``, keyed by threshold."""
    sweep: dict[float, dict[str, Any]] = {}
    for directory in sorted(root.glob("sweep_*")):
        report_path = directory / "otsu_baseline.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        sweep[float(report["min_separability"])] = report
    # The un-suffixed run is the shipped default; include it so the curve covers
    # the value the code actually ships with.
    default_path = root / "otsu_baseline.json"
    if default_path.is_file():
        report = json.loads(default_path.read_text(encoding="utf-8"))
        sweep.setdefault(float(report["min_separability"]), report)
    return dict(sorted(sweep.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=BASELINE_DIR)
    args = parser.parse_args(argv)

    sweep = load_sweep(args.dir)
    if len(sweep) < 3:
        print(
            f"found {len(sweep)} sweep runs under {args.dir}; run "
            f"run_flood_baseline.py at several --min-separability values first",
            file=sys.stderr,
        )
        return 2

    thresholds = sorted(sweep)
    regions = sorted(sweep[thresholds[0]]["per_region"])

    def counts(threshold: float, region: str) -> tuple[int, int, int]:
        row = sweep[threshold]["per_region"][region]
        return int(row["tp"]), int(row["fp"]), int(row["fn"])

    # --- the all-data optimum, reported only as the thing NOT to quote -------
    all_data: dict[float, float] = {}
    for threshold in thresholds:
        tp = sum(counts(threshold, r)[0] for r in regions)
        fp = sum(counts(threshold, r)[1] for r in regions)
        fn = sum(counts(threshold, r)[2] for r in regions)
        all_data[threshold] = _iou(tp, fp, fn)
    naive_best = max(all_data, key=lambda t: all_data[t])

    # --- leave-one-region-out selection --------------------------------------
    folds: dict[str, dict[str, Any]] = {}
    pooled_tp = pooled_fp = pooled_fn = 0

    for held_out in regions:
        others = [r for r in regions if r != held_out]
        scored = {}
        for threshold in thresholds:
            tp = sum(counts(threshold, r)[0] for r in others)
            fp = sum(counts(threshold, r)[1] for r in others)
            fn = sum(counts(threshold, r)[2] for r in others)
            scored[threshold] = _iou(tp, fp, fn)
        chosen = max(scored, key=lambda t: scored[t])

        tp, fp, fn = counts(chosen, held_out)
        pooled_tp, pooled_fp, pooled_fn = pooled_tp + tp, pooled_fp + fp, pooled_fn + fn
        folds[held_out] = {
            "selected_min_separability": chosen,
            "selection_iou_on_other_regions": round(scored[chosen], 4),
            "held_out_iou": round(_iou(tp, fp, fn), 4),
            "n_chips": sweep[chosen]["per_region"][held_out]["n_chips"],
        }

    honest_iou = _iou(pooled_tp, pooled_fp, pooled_fn)

    report = {
        "experiment": "flood_baseline_threshold_selection",
        "run_at": datetime.now(UTC).isoformat(),
        "method": "Otsu on Sentinel-1 VV, no HAND terrain mask",
        "dataset": "Sen1Floods11 v1.1 HandLabeled, 441 chips scored across 11 regions",
        "protocol": (
            "Leave-one-region-out selection of the separability guard: for each "
            "region, the guard is chosen on the other ten and applied to the "
            "held-out one. Confusion counts pooled, not averaged."
        ),
        "curve_all_data": {str(t): round(v, 4) for t, v in all_data.items()},
        "naive_all_data_best": {
            "min_separability": naive_best,
            "iou": round(all_data[naive_best], 4),
            "warning": (
                "This is the maximum of a curve fitted to the evaluation set. It "
                "is NOT a generalisation estimate and must not be reported as the "
                "baseline's score."
            ),
        },
        "loro_selected": folds,
        "headline": {
            "iou": round(honest_iou, 4),
            "optimism_of_naive_protocol": round(all_data[naive_best] - honest_iou, 4),
        },
        "caveats": [
            "No HAND raster ships with Sen1Floods11 HandLabeled, so the terrain "
            "mask that removes Otsu's false positives is absent. This is a lower "
            "bound on the method as operationally deployed.",
            "The guard is the only tuned quantity; Otsu itself is unsupervised "
            "and derives its threshold per chip.",
        ],
    }

    (args.dir / "threshold_selection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("guard sweep, pooled over all regions (NOT a generalisation estimate):")
    for threshold in thresholds:
        marker = "  <- all-data best" if threshold == naive_best else ""
        print(f"  {threshold:>5}  IoU {all_data[threshold]:.4f}{marker}")

    print("\nleave-one-region-out selection:")
    print(f"  {'region':<12} {'chose':>6} {'held-out IoU':>13} {'chips':>6}")
    for region, fold in folds.items():
        print(
            f"  {region:<12} {fold['selected_min_separability']:>6} "
            f"{fold['held_out_iou']:>13.4f} {fold['n_chips']:>6}"
        )

    print(f"\nHONEST POOLED IoU (region-disjoint selection): {honest_iou:.4f}")
    print(f"naive all-data optimum:                       {all_data[naive_best]:.4f}")
    print(f"optimism the naive protocol would have added: {all_data[naive_best] - honest_iou:+.4f}")
    print(f"\nwritten to {args.dir / 'threshold_selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
