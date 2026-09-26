"""Build the model registry from artifacts on disk.

    python -m ml.registry.build_registry

**Derived, never hand-written.** A registry someone edits by hand is a registry
that says TRAINED for a model nobody trained -- the single most damaging thing
this file could contain, because everything downstream treats it as the record
of what exists. So status is computed from evidence:

    TRAINED AND EVALUATED   a training manifest AND a test report exist
    TRAINED                 a training manifest exists, no test report
    EVALUATED               scored, but not a trained model (the baselines)
    NOT TRAINED             neither

Metrics are copied out of the evaluation reports rather than restated, and a
model with no evaluation report carries no metrics at all -- not zeros, not
nulls with optimistic names. The registry records the checkpoint's existence
separately from its status, because weights are gitignored and a clone will
legitimately have the record without the file.
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

MODELS = REPO_ROOT / "models" / "flood"
EXPERIMENTS = REPO_ROOT / "ml" / "experiments"
OUTPUT = REPO_ROOT / "ml" / "registry" / "models.json"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with contextlib.suppress(ValueError, OSError):
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data
    return None


def _posix(path: str) -> str:
    """Normalise a separator in a path copied out of an older report.

    Reports written on Windows before `satai.paths.relative_to_repo` existed
    record paths with backslashes, which resolve to nothing on Linux -- in CI,
    in the serverless functions, and in any clone. A report is the record of a
    run and is not edited after the fact, so the separator is normalised on the
    way into the registry instead. Anything written from now on is already
    POSIX and passes through unchanged.
    """
    return path.replace("\\", "/")


def _deep_entries() -> list[dict[str, Any]]:
    """One entry per trained flood model, from its training manifest."""
    entries: list[dict[str, Any]] = []

    for manifest_path in sorted(MODELS.glob("train_manifest_*.json")):
        manifest = _load(manifest_path)
        if manifest is None:
            continue

        fold = manifest["fold"]
        band_tag = "sar_ratio" if "vv_vh_ratio" in manifest["bands"] else "sar"
        test = _load(EXPERIMENTS / "flood_unet" / f"test_{fold}_{band_tag}.json")
        checkpoint = MODELS / f"flood_unet_{fold}_{band_tag}_best.pt"

        status = "TRAINED AND EVALUATED" if test is not None else "TRAINED"

        entry: dict[str, Any] = {
            "name": "flood_unet",
            "variant": band_tag,
            "version": f"1.0.0+{fold}",
            "status": status,
            "architecture": "U-Net, 4 levels, base width 32, trained from scratch",
            "framework": "pytorch",
            "dataset": "Sen1Floods11 v1.1 HandLabeled",
            "modalities": manifest["bands"],
            "split_protocol": manifest["protocol"],
            "region_disjoint": manifest["region_disjoint"],
            "test_regions": manifest["test_regions"],
            "val_regions": manifest["val_regions"],
            "trained_at": manifest["finished_at"],
            "epochs_run": manifest["epochs_run"],
            "n_parameters": manifest["n_parameters"],
            "hyperparameters": manifest["hyperparameters"],
            "best_val_iou": manifest["best_val_iou"],
            "artifact": relative_to_repo(checkpoint),
            # Weights are gitignored, so a clone has the record without the
            # file. Stated rather than inferred, so "missing" never reads as
            # "never trained".
            "artifact_present": checkpoint.is_file(),
            "artifact_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
            "normalizer": relative_to_repo(MODELS / f"normalizer_{fold}_{band_tag}.json"),
        }

        # What is servable, and whether anything checked it. An export recorded
        # without its parity result would be an invitation to serve a graph
        # nobody compared against the checkpoint that was evaluated.
        parity = _load(MODELS / f"flood_unet_{fold}_{band_tag}.parity.json")
        if parity is not None:
            entry["onnx_export"] = {
                "artifact": _posix(parity["artifact"]),
                "artifact_present": (REPO_ROOT / parity["artifact"]).is_file(),
                "opset": parity["opset"],
                "agrees_with_checkpoint": parity["agrees_with_checkpoint"],
                "max_abs_probability_delta": parity["max_abs_probability_delta"],
                "n_chips_verified": parity["n_chips_verified"],
                "parity_report": relative_to_repo(
                    MODELS / f"flood_unet_{fold}_{band_tag}.parity.json"
                ),
            }

        if test is not None:
            entry["test_metrics"] = test["headline"]
            entry["test_region_chips"] = test["n_chips"]
            entry["evaluation_report"] = relative_to_repo(
                EXPERIMENTS / "flood_unet" / f"test_{fold}_{band_tag}.json"
            )
            entry["caveats"] = [
                f"Validation IoU was {manifest['best_val_iou']:.4f} on "
                f"{', '.join(manifest['val_regions'])}; the test score on "
                f"{', '.join(manifest['test_regions'])} is {test['headline']['iou']:.4f}. "
                f"The validation number is not this model's score.",
                "Pooled IoU is dominated by chips with large water bodies; the "
                f"per-chip median is {test['per_chip_iou']['median']}.",
            ]
        else:
            entry["caveats"] = [
                "No evaluation report for this variant. It has been trained but "
                "not scored on a held-out region, so it has no reportable metric."
            ]

        entries.append(entry)

    return entries


def _baseline_entry() -> dict[str, Any] | None:
    """The classical baseline, from its own selection report."""
    selection = _load(EXPERIMENTS / "flood_baseline" / "threshold_selection.json")
    scored = _load(EXPERIMENTS / "flood_baseline" / "sweep_0.66" / "otsu_baseline.json")
    if selection is None or scored is None:
        return None

    india = scored["per_region"].get("India", {})
    return {
        "name": "otsu_baseline",
        "variant": "vv_no_hand",
        "version": "1.1.0",
        # Never TRAINED: it is unsupervised and derives its threshold per chip.
        # Marking it trained would misrepresent what it is.
        "status": "EVALUATED",
        "architecture": "Otsu threshold on Sentinel-1 VV with a unimodality guard",
        "framework": "numpy",
        "dataset": "Sen1Floods11 v1.1 HandLabeled",
        "modalities": ["vv_db"],
        "split_protocol": "leave-one-region-out selection of the guard, 11 regions",
        "region_disjoint": True,
        "test_regions": ["India"],
        "trained_at": None,
        "artifact": None,
        "artifact_present": False,
        "test_metrics": {
            "iou": india.get("iou"),
            "f1": india.get("f1"),
            "precision": india.get("precision"),
            "recall": india.get("recall"),
        },
        "pooled_all_regions_iou": selection["headline"]["iou"],
        "hyperparameters": {"min_separability": 0.66, "band": "vv_db"},
        "evaluation_report": "ml/experiments/flood_baseline/threshold_selection.json",
        "caveats": [
            "Unsupervised: no training run, hence EVALUATED rather than TRAINED.",
            "No HAND terrain mask ships with Sen1Floods11, so this is a lower "
            "bound on the method as operationally deployed.",
            "This is the bar the deep model must clear.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    entries = _deep_entries()
    baseline = _baseline_entry()
    if baseline is not None:
        entries.append(baseline)

    registry = {
        "_comment": (
            "GENERATED from artifacts on disk by ml/registry/build_registry.py. "
            "Do not edit: a hand-edited registry is how a model comes to be "
            "recorded as trained without a training run."
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "n_models": len(entries),
        "status_vocabulary": {
            "TRAINED AND EVALUATED": "a training manifest and a held-out test report both exist",
            "TRAINED": "a training manifest exists; no test report, so no reportable metric",
            "EVALUATED": "scored on held-out data, but not a trained model",
            "NOT TRAINED": "neither",
        },
        "models": entries,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"{len(entries)} model(s)\n")
    print(f"  {'name':<16} {'variant':<12} {'status':<22} {'IoU':>7}  artifact")
    for entry in entries:
        iou = (entry.get("test_metrics") or {}).get("iou")
        present = "present" if entry["artifact_present"] else "gitignored/absent"
        print(
            f"  {entry['name']:<16} {entry['variant']:<12} {entry['status']:<22} "
            f"{iou if iou is not None else '—':>7}  {present}"
        )
    print(f"\nwritten to {relative_to_repo(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
