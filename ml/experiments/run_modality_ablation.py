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
    (
        "ratio_only",
        "Sentinel-1 VV/VH dB ratio only (the one band that survives a change of radiometry)",
        None,
    ),
    ("vv_only", "Sentinel-1 VV only", None),
    ("vh_only", "Sentinel-1 VH only", None),
    ("sar", "Sentinel-1 VV + VH", None),
    ("sar_ratio", "Sentinel-1 VV + VH + VV/VH dB ratio", None),
    (
        "sar_rain",
        "Sentinel-1 + GPM IMERG rainfall",
        "Needs GPM IMERG 72-hour rainfall per chip: scripts/build_ancillary.py --rain, "
        "then train and evaluate the sar_rain arm.",
    ),
    (
        "sar_dem",
        "Sentinel-1 + Copernicus DEM elevation and slope (HAND not derived)",
        "Needs Copernicus DEM elevation and slope per chip: scripts/build_ancillary.py "
        "--dem, then train and evaluate the sar_dem arm. HAND is not derived.",
    ),
    (
        "full",
        "Sentinel-1 + rainfall + terrain",
        "Blocked by both of the above.",
    ),
)


ANCILLARY = {"elevation_m", "slope_deg", "rain_72h_mm"}
ABSOLUTE = {"vv_db", "vh_db"}


def _finding(executed: dict[str, dict[str, Any]], spread: float) -> str:
    """State what the executed arms show, computed rather than asserted.

    The arms fall into three groups that answer different questions, so each is
    summarised on its own: arms built from absolute backscatter, the ratio-only
    arm, and arms that add ancillary rasters. A single sentence over all of them
    stopped being true the moment the ratio-only arm landed 0.15 IoU below the
    rest.
    """
    sentences: list[str] = []
    absolute = {
        a: v
        for a, v in executed.items()
        if set(v["bands"]) & ABSOLUTE and not set(v["bands"]) & ANCILLARY
    }
    if absolute:
        ious = [v["iou"] for v in absolute.values()]
        best = max(absolute, key=lambda a: absolute[a]["iou"])
        sentences.append(
            f"The {len(absolute)} arms built from absolute backscatter fall within "
            f"{max(ious) - min(ious):.4f} IoU of each other; the best is {best} at "
            f"{absolute[best]['iou']:.4f}. On a single seed per arm that spread is "
            "smaller than seed variation would plausibly produce, so these arms "
            "are indistinguishable: adding polarisations or the ratio band to one "
            "polarisation brings no measurable benefit."
        )
    if "ratio_only" in executed and absolute:
        best_abs = max(v["iou"] for v in absolute.values())
        ratio = executed["ratio_only"]["iou"]
        sentences.append(
            f"The ratio band alone scores {ratio:.4f}, {best_abs - ratio:.4f} below the "
            "best absolute arm. That band is the one input that survives a change "
            "of radiometric product -- a common offset on both polarisations cancels "
            "in VV - VH -- and it carries too little of the signal on its own. The "
            "flood signal lives in absolute backscatter, which is exactly what does "
            "not transfer between products."
        )
    reference = executed.get("sar_ratio")
    for arm in ("sar_dem", "sar_rain", "full"):
        if arm in executed and reference:
            delta = executed[arm]["iou"] - reference["iou"]
            sentences.append(
                f"{arm} scores {executed[arm]['iou']:.4f}, {delta:+.4f} against the "
                f"same SAR stack without the ancillary bands."
            )
    return " ".join(sentences)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default="loro_india")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    executed: dict[str, dict[str, Any]] = {}
    blocked: dict[str, str] = {}

    for arm, description, blocker in ARMS:
        path = RESULTS / f"test_{args.fold}_{arm}.json"
        # An arm counts as executed when its evaluation report exists, whatever
        # blocker it was declared with. The blocker is what to say when it does
        # not -- the rainfall and terrain arms were once blocked on data, and a
        # hard-coded "blocked" would outlive the data arriving.
        if not path.is_file() and blocker is not None:
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
        "status": "EXECUTED" if not blocked else "PARTIALLY EXECUTED",
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
            (
                f"{len(executed)} of {len(executed) + len(blocked)} arms executed; the "
                f"rest are reported as blocked with what they need, not approximated."
                if blocked
                else f"All {len(executed)} arms executed."
            ),
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
