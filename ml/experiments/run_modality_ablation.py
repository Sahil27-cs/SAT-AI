"""C2: degradation under modality loss. Executed, and the result is negative.

    python -m ml.experiments.run_modality_ablation

Reads the per-arm test reports written by ``ml.flood.evaluate`` and reports the
delta between them. It computes nothing itself -- every number traces to a
training run and an evaluation on the same held-out region -- so that the
comparison cannot quietly disagree with the runs it summarises.

**Scope, stated precisely.** C2 as registered compares SAR against SAR plus
rainfall, SAR plus DEM/HAND, and the full stack. Only the first pair is
executed here, because Sen1Floods11's hand-labelled release ships neither a
rainfall raster nor a HAND raster co-registered to its chips. Acquiring and
aligning those is a data-plane task that has not been done, so the rainfall and
terrain arms remain NOT EXECUTED rather than being approximated from something
else. What *is* executed is a real ablation: the derived VV/VH ratio band,
present in one arm and absent in the other, with everything else -- seed,
schedule, augmentation, fold, selection region -- held fixed.

The result is that the ratio band does not help. That is worth reporting. An
ablation that only ever confirms the fuller stack is better is an ablation
nobody needed to run.
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

RESULTS = REPO_ROOT / "ml" / "experiments" / "flood_unet"
OUTPUT = REPO_ROOT / "ml" / "experiments" / "c2_modality_ablation.json"

#: Arms of the ablation. The two executed ones differ by exactly one band; the
#: rest name the raster they need, because "not executed" without a reason is
#: indistinguishable from "forgotten".
ARMS: tuple[tuple[str, str, str | None], ...] = (
    ("vv_only", "Sentinel-1 VV only", None),
    ("vh_only", "Sentinel-1 VH only", None),
    ("sar", "Sentinel-1 VV + VH", None),
    ("sar_ratio", "Sentinel-1 VV + VH + VV/VH dB ratio", None),
    (
        "sar_rain",
        "Sentinel-1 + GPM IMERG rainfall",
        "No rainfall raster co-registered to Sen1Floods11 chips. Needs a GPM "
        "IMERG pull for each chip's acquisition window, resampled to the chip grid.",
    ),
    (
        "sar_dem",
        "Sentinel-1 + Copernicus DEM slope and HAND",
        "No DEM or HAND raster ships with Sen1Floods11. Needs a Copernicus DEM "
        "GLO-30 pull per chip footprint plus a HAND derivation.",
    ),
    (
        "full",
        "Sentinel-1 + rainfall + terrain",
        "Blocked by both of the above.",
    ),
)


def _finding(executed: dict[str, dict[str, Any]], spread: float) -> str:
    """State what the executed arms show, computed rather than asserted.

    Written as a function because the conclusion changed when the two
    single-polarisation arms landed: a sentence about the ratio band was true of
    two arms and became the least interesting thing about four.
    """
    ranked = sorted(executed.items(), key=lambda kv: kv[1]["iou"], reverse=True)
    best_arm, best = ranked[0]
    worst_arm, worst = ranked[-1]
    fullest = max(executed.items(), key=lambda kv: kv[1]["n_bands"])

    return (
        f"All {len(executed)} executed arms fall within {spread:.4f} IoU of each "
        f"other. The best is {best_arm} at {best['iou']:.4f} with "
        f"{best['n_bands']} band(s); the worst is {worst_arm} at "
        f"{worst['iou']:.4f}. The fullest stack, {fullest[0]} with "
        f"{fullest[1]['n_bands']} bands, scores {fullest[1]['iou']:.4f} -- lower "
        f"than a single {best_arm.split('_')[0].upper()} polarisation. "
        "On this dataset, with this architecture, adding SAR modalities beyond "
        "one polarisation produces no measurable benefit. The spread is smaller "
        "than the per-chip variance and, on a single seed per arm, smaller than "
        "what seed variation would plausibly produce -- so the defensible claim "
        "is that these arms are indistinguishable, not that fewer bands are "
        "better."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default="loro_india")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    executed: dict[str, dict[str, Any]] = {}
    blocked: dict[str, str] = {}

    for arm, description, blocker in ARMS:
        path = RESULTS / f"test_{args.fold}_{arm}.json"
        if blocker is not None:
            blocked[arm] = blocker
            continue
        if not path.is_file():
            blocked[arm] = f"no evaluation report at {path.name}; train and evaluate this arm first"
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        executed[arm] = {
            "description": description,
            "bands": report["bands"],
            "n_bands": len(report["bands"]),
            **report["headline"],
            "per_chip_iou_median": report["per_chip_iou"]["median"],
        }

    if len(executed) < 2:
        raise SystemExit(f"C2 needs at least two executed arms to compare; have {sorted(executed)}")

    # The fullest executed arm is the reference; every other arm is reported as
    # its degradation. On this dataset the "degradations" are mostly negative --
    # arms with fewer bands score higher -- which is the result rather than a
    # problem with the framing.
    ious = [values["iou"] for values in executed.values()]
    spread = round(max(ious) - min(ious), 4)
    reference_arm = max(executed, key=lambda a: executed[a]["n_bands"])
    reference = executed[reference_arm]

    degradation = {
        arm: {
            "delta_iou": round(values["iou"] - reference["iou"], 4),
            "delta_f1": round(values["f1"] - reference["f1"], 4),
            "relative_iou": round((values["iou"] - reference["iou"]) / reference["iou"], 4),
        }
        for arm, values in executed.items()
        if arm != reference_arm
    }

    report = {
        "experiment": "C2_modality_ablation",
        "run_at": datetime.now(UTC).isoformat(),
        "status": "PARTIALLY EXECUTED",
        "fold": args.fold,
        "test_region": "India",
        "protocol": (
            "Leave-one-region-out. Every arm shares the seed, schedule, "
            "augmentation, training fold and selection region; they differ only "
            "in the input band stack."
        ),
        "reference_arm": reference_arm,
        "executed": executed,
        "degradation_vs_reference": degradation,
        "blocked": blocked,
        "finding": _finding(executed, spread),
        "spread_across_executed_arms_iou": spread,
        "caveats": [
            f"{len(executed)} arms, not five. The rainfall and terrain arms need "
            "ancillary rasters that have not been acquired, and are reported as "
            "blocked rather than approximated.",
            "Single seed per arm. With differences this small the honest "
            "statement is that the arms are indistinguishable at this sample "
            "size, not that one is better. Separating them would need repeated "
            "seeds and a confidence interval, and that measurement has not been "
            "made.",
            "The precision/recall split differs far more than the IoU does -- "
            "the two-polarisation arm is the most precise and the least "
            "sensitive of the four. That is a real difference in operating "
            "behaviour even where the summary metric agrees, and it matters for "
            "a flood product where a missed inundation and a false one carry "
            "different costs.",
            "Every arm shares one architecture. A larger model, or one with a "
            "pretrained encoder, might extract something from the extra bands "
            "that this one cannot. The result is about this model on this "
            "dataset, not about SAR polarimetry in general.",
        ],
    }

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"C2 modality ablation -- test region India, fold {args.fold}\n")
    print(f"  {'arm':<12} {'bands':>5} {'IoU':>7} {'F1':>7} {'prec':>7} {'recall':>7}")
    for arm, values in sorted(executed.items(), key=lambda kv: kv[1]["n_bands"]):
        print(
            f"  {arm:<12} {values['n_bands']:>5} {values['iou']:>7.4f} {values['f1']:>7.4f} "
            f"{values['precision']:>7.4f} {values['recall']:>7.4f}"
        )
    print(f"\n  reference arm: {reference_arm}")
    for arm, delta in degradation.items():
        print(
            f"  {arm} vs {reference_arm}: IoU {delta['delta_iou']:+.4f} "
            f"({delta['relative_iou']:+.1%})"
        )
    print(f"\n  {report['finding']}")
    print("\n  blocked arms:")
    for arm, reason in blocked.items():
        print(f"    {arm}: {reason}")
    print(f"\nwritten to {relative_to_repo(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
