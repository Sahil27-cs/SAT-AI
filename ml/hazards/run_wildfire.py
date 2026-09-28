"""Active-fire observations over the configured study areas.

    python -m ml.hazards.run_wildfire

`satai.hazards.wildfire` can compute burn severity and a fire-danger index; what
it lacked was any real input. This runner supplies the one wildfire quantity
that is obtainable without an account: NASA FIRMS thermal anomalies, from the
rolling regional archive NASA publishes as plain CSV.

What this produces, stated precisely
------------------------------------
**Active-fire detections.** A thermal anomaly at a satellite overpass. That is
an observation, and it is the weakest of the three wildfire products this
project can describe:

* it is *not* burned area, which needs a pre/post optical pair and the dNBR
  computation in `satai.hazards.wildfire`;
* it is *not* burn severity, for the same reason;
* it is *not* a fire-danger forecast, which needs the weather and fuel-state
  inputs that require Earth Engine credentials;
* and it is emphatically not ignition prediction, which this project does not
  do and has no model for.

**Zero detections is not zero fire.** The sensor sees a given pixel twice a day
at best, cloud blocks the thermal band, and a fire below the detection limit is
invisible. An empty result for a study area means exactly "nothing was detected
in this window", and the artifact says so in those words rather than reporting
a reassuring zero.

**A detection is not necessarily a wildfire.** VIIRS records hot pixels. Over an
urban or industrial area those include flares, kilns and furnaces, which is why
the per-area result carries the confidence breakdown rather than a bare count.
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

    from satai.paths import relative_to_repo

from satai.geo.aoi import load_aoi_registry
from satai.providers.firms import fetch_open_detections

DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "wildfire"

#: Days the rolling archive covers, matching the file requested below. Recorded
#: in the artifact because "3 detections" means nothing without the window.
WINDOW_DAYS = {"24h": 1, "48h": 2, "7d": 7}


def _summarise(detections: list[Any]) -> dict[str, Any]:
    """Counts and radiative power, with the confidence split kept visible."""
    confidences = [str(d.confidence or "").lower() for d in detections]
    frp = [d.frp_mw for d in detections if d.frp_mw is not None]
    return {
        "detection_count": len(detections),
        "confidence_breakdown": {
            "high": sum(1 for c in confidences if c in {"h", "high"}),
            "nominal": sum(1 for c in confidences if c in {"n", "nominal"}),
            "low": sum(1 for c in confidences if c in {"l", "low"}),
        },
        # Averaged over the records that report it, not over all records:
        # dividing by the full count deflates the mean in proportion to how many
        # are missing it.
        "mean_frp_mw": round(sum(frp) / len(frp), 3) if frp else None,
        "max_frp_mw": round(max(frp), 3) if frp else None,
        "frp_reported_count": len(frp),
        "first_detection": min((d.acquired_at for d in detections), default=None),
        "last_detection": max((d.acquired_at for d in detections), default=None),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", default=None, help="one study area; default is all of them")
    parser.add_argument("--product", default="noaa-20-viirs-c2")
    parser.add_argument("--window", default="7d", choices=sorted(WINDOW_DAYS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    registry = load_aoi_registry(REPO_ROOT / "configs" / "aoi.yaml")
    areas = [a for a in registry.aois if args.aoi in (None, a.id)]
    if not areas:
        raise SystemExit(f"{args.aoi!r} is not a configured study area")

    per_area: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    print(f"  {'study area':<24} {'detections':>10}  {'high conf':>9}  {'max FRP':>8}")

    for aoi in areas:
        detections, provenance = fetch_open_detections(
            aoi.bbox, product=args.product, window=args.window
        )
        summary = _summarise(detections)
        per_area.append(
            {
                "aoi": aoi.id,
                "aoi_name": aoi.name,
                "country": aoi.country,
                "bbox": list(aoi.bbox),
                **summary,
                "first_detection": summary["first_detection"].isoformat()
                if summary["first_detection"]
                else None,
                "last_detection": summary["last_detection"].isoformat()
                if summary["last_detection"]
                else None,
                "detections": [
                    {
                        "lat": d.latitude,
                        "lon": d.longitude,
                        "observed_at": d.acquired_at.isoformat(),
                        "frp_mw": d.frp_mw,
                        "brightness_k": d.brightness_k,
                        "confidence": d.confidence,
                        "satellite": d.satellite,
                    }
                    for d in detections
                ],
            }
        )
        high = summary["confidence_breakdown"]["high"]
        peak = summary["max_frp_mw"]
        print(
            f"  {aoi.id:<24} {summary['detection_count']:>10}  {high:>9}  "
            f"{peak if peak is not None else '-':>8}"
        )

    payload = {
        "analysis": "active_fire_detections",
        "hazard": "wildfire",
        "source_kind": "observation",
        "not_a_prediction": (
            "Thermal anomalies observed at satellite overpass. Not burned area, "
            "not burn severity, not a fire-danger forecast, and not ignition "
            "prediction."
        ),
        "run_at": datetime.now(UTC).isoformat(),
        "window": args.window,
        "window_days": WINDOW_DAYS[args.window],
        "provenance": provenance,
        "study_areas": per_area,
        "totals": {
            "areas_examined": len(per_area),
            "areas_with_detections": sum(1 for a in per_area if a["detection_count"]),
            "detections": sum(a["detection_count"] for a in per_area),
        },
        "caveats": [
            "Zero detections means nothing was detected in this window. It does "
            "NOT mean nothing burned: the sensor sees a given pixel twice a day "
            "at best, cloud blocks the thermal band, and a fire below the "
            "detection limit is invisible.",
            "A detection is a hot pixel. Over urban and industrial areas these "
            "include flares, kilns and furnaces, which is why the confidence "
            "breakdown is reported rather than a bare count.",
            "The keyless archive covers only a rolling recent window. A named "
            "historical fire needs the FIRMS archive API and a MAP_KEY, which "
            "this deployment does not have.",
            "Burned area and burn severity are implemented in satai/hazards/ "
            "but have no input here: they need a pre/post optical pair that no "
            "acquisition in this repository provides.",
        ],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = args.out / f"active_fire_{args.window}_{stamp}.json"
    destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    latest = args.out / "active_fire_latest.json"
    latest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    totals = payload["totals"]
    print(
        f"\n  {totals['detections']} detection(s) across "
        f"{totals['areas_with_detections']} of {totals['areas_examined']} study areas"
    )
    print(f"  source file held {provenance.get('rows_in_file', 0):,} rows for South Asia")
    print(f"\nwritten to {relative_to_repo(destination)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
