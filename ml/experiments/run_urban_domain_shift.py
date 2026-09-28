"""C3's precondition: how far urban Indian SAR sits from the training domain.

    python -m ml.experiments.run_urban_domain_shift --aoi mumbai_mmr \
        --start 2024-07-01 --end 2024-07-31

C3 asks for the **size of the performance gap** when a flood model trained on
rural Indian chips is evaluated on urban Indian imagery. That number cannot be
produced here, and this script does not pretend otherwise: it needs labelled
urban Indian flood imagery, and none exists that this project can obtain. The
blocker is recorded in the artifact, with what was checked.

What *is* measurable is the precondition. `satai/ml/distribution_gate.py` exists
to answer "are these two domains close enough that a model trained on one means
anything on the other", and answering it needs no labels at all — only the two
distributions. So this measures the shift between the rural Indian chips the
model was trained on and a real Sentinel-1 acquisition over Mumbai.

Why the direction is known in advance, and why measuring it still matters
------------------------------------------------------------------------
Urban flooding inverts the SAR signature. Open water is smooth and returns
little energy to the sensor, which is what makes flood mapping work at all;
but water standing between building walls produces a double-bounce that returns
*more* energy than dry ground. A model that has only seen rural flooding has
learned "dark means water", and in a city that rule is wrong in the most
damaging possible way -- it will miss precisely the flooded streets that matter.

Knowing the direction is not the same as knowing the magnitude, which is why C3
was worth proposing. This script produces neither the magnitude of the
*performance* gap nor a substitute for it. It produces the measured distance
between the two input distributions, and says exactly that.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path

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

    from satai.paths import relative_to_repo

import numpy as np

from ml.flood.dataset import bands_for, build_reader, stack_with_ratio
from ml.flood.infer_scene import read_window, to_db
from satai.geo.aoi import load_aoi_registry
from satai.ml.distribution_gate import run_gate
from satai.providers import PlanetaryComputerProvider, SearchQuery

OUTPUT = REPO_ROOT / "ml" / "experiments" / "c3_urban_domain_shift.json"

#: What was checked for a labelled urban Indian target, and what it holds. Kept
#: in the artifact so "blocked" is a finding with evidence rather than an
#: assertion someone has to take on trust.
TARGETS_CHECKED: tuple[dict[str, str], ...] = (
    {
        "dataset": "Sen1Floods11 v1.1 HandLabeled",
        "status": "obtained",
        "urban_indian_scenes": "none",
        "note": (
            "The 68 India chips are the training domain for this project and are "
            "rural floodplain. They are the source domain, not a target."
        ),
    },
    {
        "dataset": "UrbanSARFloods (processed release, Zenodo, open access)",
        "status": "available but unsuitable",
        "urban_indian_scenes": "none",
        "note": (
            "Held-out urban scenes are Weihui (China), Nova Kakhovka (Ukraine) "
            "and two Jubba (Somalia) image-label pairs. No Indian city."
        ),
    },
    {
        "dataset": "Copernicus EMS rapid mapping",
        "status": "not obtained",
        "urban_indian_scenes": "unknown",
        "note": (
            "Publishes vector delineations for activated events. Activations "
            "over Indian cities would need to be identified case by case and "
            "the products are vector, not a labelled raster benchmark."
        ),
    },
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", default="mumbai_mmr")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--chips", type=Path, default=REPO_ROOT / "data/raw/sen1floods11")
    parser.add_argument("--source-region", default="India")
    parser.add_argument("--limit-chips", type=int, default=24)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    registry = load_aoi_registry(REPO_ROOT / "configs" / "aoi.yaml")
    aoi = next((a for a in registry.aois if a.id == args.aoi), None)
    if aoi is None:
        raise SystemExit(f"{args.aoi!r} is not a configured study area")

    # --- the source domain: the rural Indian chips the model was trained on ---
    bands = bands_for(True)
    reader = build_reader(args.chips)
    chips = [c for c in reader.available(reader.discover()) if c.region == args.source_region]
    chips = chips[: args.limit_chips]
    if not chips:
        raise SystemExit(f"no {args.source_region} chips under {args.chips}")

    stacks = []
    for chip in chips:
        features, _labels, valid = reader.load(chip)
        stack = stack_with_ratio(features, True)
        stacks.append(np.where(valid[None], stack, np.nan))
    source = np.concatenate(stacks, axis=2)
    print(f"source  {len(chips)} {args.source_region} chips (rural floodplain)")

    # --- the target domain: a real urban acquisition --------------------------
    provider = PlanetaryComputerProvider()
    result = provider.search(
        SearchQuery(
            bbox=aoi.bbox,
            start=args.start,
            end=args.end,
            collection="sentinel-1-rtc",
            limit=50,
        )
    )
    if not result.scenes:
        print(
            f"BLOCKED: no Sentinel-1 RTC scene over {aoi.id} "
            f"between {args.start} and {args.end}."
        )
        return 2
    scene = result.scenes[0]
    print(f"target  {scene.scene_id}")
    print(f"        {scene.acquired_at}  orbit {scene.relative_orbit}")

    target_bands = []
    for band in ("vv", "vh"):
        linear, _t, _p = read_window(provider.asset_href(scene, band), aoi.bbox)
        target_bands.append(to_db(linear))
    target = stack_with_ratio(np.stack(target_bands), True)
    observed = np.isfinite(target).all(axis=0)

    gate = run_gate(
        {band: source[i] for i, band in enumerate(bands)},
        {band: np.where(observed, target[i], np.nan) for i, band in enumerate(bands)},
        track_a_source=(
            f"Sen1Floods11 {args.source_region} chips ({len(chips)}), rural, sigma0 dB"
        ),
        track_b_source=f"{scene.scene_id} over {aoi.name}, urban, gamma0 RTC dB",
    )

    print(f"\ngate    {gate.verdict}")
    for comparison in gate.bands:
        print(f"        {comparison.summary()}")

    report = {
        "experiment": "c3_urban_domain_shift",
        "contribution": "C3",
        "status": "PRECONDITION MEASURED; CONTRIBUTION BLOCKED",
        "run_at": datetime.now(UTC).isoformat(),
        "what_this_is": (
            "The measured distribution shift between the rural Indian chips the "
            "flood model was trained on and a real Sentinel-1 acquisition over "
            "an Indian city."
        ),
        "what_this_is_not": (
            "NOT the rural-to-urban performance gap C3 proposes. That is an IoU "
            "difference and it requires labelled urban Indian flood imagery, "
            "which does not exist in any source this project can obtain. No "
            "performance number is produced here, and none should be inferred "
            "from the shift below."
        ),
        "source_domain": {
            "dataset": "Sen1Floods11 v1.1 HandLabeled",
            "region": args.source_region,
            "n_chips": len(chips),
            "character": "rural floodplain",
            "radiometry": "sigma0 dB (Earth Engine COPERNICUS/S1_GRD)",
        },
        "target_domain": {
            "aoi": aoi.id,
            "aoi_name": aoi.name,
            "scene_id": scene.scene_id,
            "acquired_at": scene.acquired_at.isoformat() if scene.acquired_at else None,
            "relative_orbit": scene.relative_orbit,
            "orbit_direction": scene.orbit_direction,
            "character": "urban and peri-urban",
            "radiometry": scene.properties["radiometry"],
            "provider": "planetary_computer",
            "labelled": False,
        },
        "distribution_gate": gate.to_dict(),
        "expected_direction": (
            "Urban flooding inverts the signature the model learned. Open water "
            "is smooth and returns little energy; water standing between "
            "building walls double-bounces and returns more than dry ground. A "
            "model taught 'dark means water' will therefore miss flooded "
            "streets, which is the failure mode that matters most in a city."
        ),
        "blocker": {
            "missing": "labelled urban Indian flood imagery",
            "targets_checked": list(TARGETS_CHECKED),
            "what_would_unblock_it": (
                "Either a labelled urban Indian flood benchmark, or a Copernicus "
                "EMS activation over an Indian city whose vector delineation is "
                "rasterised to the chip grid and validated as a label set."
            ),
        },
        "caveats": [
            "The shift measured here mixes two things: the rural-to-urban "
            "difference C3 is about, and the sigma0-to-gamma0 radiometric "
            "difference between the two products. They are not separated, and "
            "the second is not a domain gap at all.",
            "One target scene, one date. A single acquisition is not the urban "
            "domain; it is one sample of it.",
            "A distribution gate says whether a transfer is defensible, not how "
            "well it would perform. Those are different questions and only the "
            "first is answered here.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {relative_to_repo(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
