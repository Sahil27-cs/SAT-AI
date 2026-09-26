#!/usr/bin/env python3
"""Verify what is actually inside Sen1Floods11, from the file listing.

Two questions, both of which the project had answered from memory rather than
from evidence:

**1. Does an India event exist?** ADR-007 assumed it did. Verified 2026-09-22:
yes, 68 hand-labelled chips.

**2. Are the official splits region-disjoint?** The project assumed they were.
**They are not.** Every event region appears in train, validation *and* test at
roughly 60/20/20 -- these are chip-level stratified random splits, so chips from
the same flood event, often the same acquisition date and adjacent ground, sit
on both sides of the train/test boundary. That is spatial autocorrelation
leakage, and it inflates reported IoU.

Bolivia is the exception: it is held out of all three files entirely and
distributed as a separate region hold-out set. That is the one genuinely
region-disjoint evaluation the dataset ships with, and it is 15 chips.

The consequence for SAT-AI is in ADR-009: build leave-one-region-out splits
rather than using the official ones, and report both numbers so the gap between
them is visible.

**3. Where are the India chips?** ``--footprints`` reads each chip's
georeferencing from its TIFF header over HTTP range requests -- a few kilobytes
per file instead of the whole raster -- and prints the union bounding box. That
is what tells us whether ``configs/aoi.yaml``'s invented ``bihar_ganga`` bbox
actually matches the labelled data.

Usage
-----
    python scripts/verify_sen1floods11.py
    python scripts/verify_sen1floods11.py --footprints
    python scripts/verify_sen1floods11.py --footprints --region India --json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satai.errors import ValidationError  # noqa: E402
from satai.geo.tiff_header import parse_geotiff_header, union_bbox  # noqa: E402

BUCKET = "sen1floods11"
LIST_API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
OBJECT_URL = f"https://storage.googleapis.com/{BUCKET}/"

# Hand-labelled chips only. The weakly-labelled split is produced by
# thresholding, so it inherits the errors of the classical method this project
# benchmarks against and must never be used as test data.
HAND_S1_PREFIX = "v1.1/data/flood_events/HandLabeled/S1Hand/"
HAND_LABEL_PREFIX = "v1.1/data/flood_events/HandLabeled/LabelHand/"
SPLIT_PREFIX = "v1.1/splits/flood_handlabeled/"

SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",  # the one true region hold-out
}

# Bytes to fetch per chip. A GDAL-written TIFF puts its image file directory and
# georeferencing tags at the front, so this is comfortably enough.
HEADER_BYTES = 65_536

REFERENCE = (
    "Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). "
    "Sen1Floods11: a georeferenced dataset to train and test deep learning "
    "flood algorithms for Sentinel-1. CVPRW 2020."
)


def _get(url: str, *, range_bytes: int | None = None, timeout: int = 90) -> bytes:
    headers = {"User-Agent": "SAT-AI/0.1"}
    if range_bytes is not None:
        headers["Range"] = f"bytes=0-{range_bytes - 1}"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        data: bytes = response.read()
    return data


def list_objects(prefix: str, *, max_pages: int = 60) -> list[dict[str, Any]]:
    """List objects under a prefix in the public bucket, following pagination."""
    objects: list[dict[str, Any]] = []
    page_token: str | None = None

    for _ in range(max_pages):
        params: dict[str, str] = {"prefix": prefix, "maxResults": "1000"}
        if page_token:
            params["pageToken"] = page_token
        payload = json.loads(_get(f"{LIST_API}?{urllib.parse.urlencode(params)}").decode())
        objects.extend(payload.get("items", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return objects


def event_from_name(name: str) -> str | None:
    """Extract the event region from a chip filename.

    Chips are named ``<Region>_<id>_<Type>.tif`` -- ``India_902184_S1Hand.tif``.
    """
    base = name.rsplit("/", 1)[-1]
    if not base.endswith(".tif"):
        return None
    return base.split("_", 1)[0] or None


def fetch_split(filename: str) -> list[str]:
    text = _get(f"{OBJECT_URL}{urllib.parse.quote(SPLIT_PREFIX + filename)}").decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def classify_splits(split_events: dict[str, Counter[str]]) -> tuple[str, list[str]]:
    """Decide whether the splits are region-disjoint or chip-level random.

    Returns ``(verdict, shared_regions)``. A region appearing in both train and
    test means chips from one flood event straddle the boundary -- the splits
    are stratified *within* region, not disjoint *across* it. That distinction
    is the difference between a spatial-generalisation result and an inflated
    one, so it is decided from the data rather than assumed.
    """
    train = set(split_events.get("train", Counter()))
    test = set(split_events.get("test", Counter()))
    shared = sorted(train & test)
    return ("chip_level_random" if shared else "region_disjoint"), shared


def fetch_footprints(
    region: str, chip_names: list[str], *, limit: int | None = None
) -> tuple[list[tuple[float, float, float, float]], set[int | None], list[str]]:
    """Read each chip's bbox from its TIFF header via range requests."""
    boxes: list[tuple[float, float, float, float]] = []
    crs_codes: set[int | None] = set()
    problems: list[str] = []

    selected = chip_names[:limit] if limit else chip_names
    for i, name in enumerate(selected, start=1):
        url = f"{OBJECT_URL}{urllib.parse.quote(name)}"
        try:
            header = parse_geotiff_header(_get(url, range_bytes=HEADER_BYTES))
            boxes.append(header.bbox)
            crs_codes.add(header.epsg)
        except (ValidationError, urllib.error.URLError, TimeoutError) as exc:
            problems.append(f"{name.rsplit('/', 1)[-1]}: {exc}")

        if i % 10 == 0 or i == len(selected):
            print(f"    {region}: read {i}/{len(selected)} chip headers", flush=True)

    return boxes, crs_codes, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a report.")
    parser.add_argument(
        "--footprints",
        action="store_true",
        help="Read chip georeferencing over range requests and print the union bbox.",
    )
    parser.add_argument(
        "--region", default="India", help="Region to read footprints for (default: India)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap chips read per region.")
    args = parser.parse_args()

    print("\nListing the public Sen1Floods11 bucket (metadata only)...\n")

    try:
        s1_objects = list_objects(HAND_S1_PREFIX)
        label_objects = list_objects(HAND_LABEL_PREFIX)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Could not list the bucket: {exc}\n", file=sys.stderr)
        print(
            "Fallback: https://github.com/cloudtostreet/Sen1Floods11 (gs://sen1floods11)\n"
            f"Reference: {REFERENCE}",
            file=sys.stderr,
        )
        return 1

    s1_names = [o["name"] for o in s1_objects if event_from_name(o.get("name", ""))]
    s1_events = Counter(e for n in s1_names if (e := event_from_name(n)))
    label_events = Counter(e for o in label_objects if (e := event_from_name(o.get("name", ""))))

    split_events: dict[str, Counter[str]] = {}
    split_sizes: dict[str, int] = {}
    for split_name, filename in SPLIT_FILES.items():
        try:
            rows = fetch_split(filename)
            split_sizes[split_name] = len(rows)
            split_events[split_name] = Counter(
                e for row in rows if (e := event_from_name(row.split(",")[0]))
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  (could not fetch {filename}: {exc})", file=sys.stderr)

    verdict, shared_regions = classify_splits(split_events)
    in_any_split = set().union(*(set(c) for c in split_events.values())) if split_events else set()
    excluded = sorted(set(s1_events) - in_any_split)

    footprint_result: dict[str, Any] = {}
    if args.footprints:
        region_chips = [n for n in s1_names if event_from_name(n) == args.region]
        if not region_chips:
            print(f"No chips found for region {args.region!r}", file=sys.stderr)
        else:
            print(
                f"Reading {len(region_chips)} chip headers for {args.region} "
                f"(~{HEADER_BYTES // 1024} KB each, range requests)...\n"
            )
            boxes, crs_codes, problems = fetch_footprints(
                args.region, region_chips, limit=args.limit
            )
            if boxes:
                footprint_result = {
                    "region": args.region,
                    "chips_read": len(boxes),
                    "chips_failed": len(problems),
                    "crs": sorted(str(c) for c in crs_codes),
                    "union_bbox": list(union_bbox(boxes)),
                    "problems": problems[:5],
                }

    result: dict[str, Any] = {
        "bucket": BUCKET,
        "s1_hand_chips": sum(s1_events.values()),
        "label_hand_chips": sum(label_events.values()),
        "events": dict(sorted(s1_events.items())),
        "split_sizes": split_sizes,
        "split_structure": verdict,
        "regions_in_both_train_and_test": shared_regions,
        "regions_excluded_from_main_splits": excluded,
        "footprints": footprint_result,
        "reference": REFERENCE,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    # --- contents -------------------------------------------------------
    print("=" * 78)
    print("  Sen1Floods11 — verified contents")
    print("=" * 78)
    print(f"\nHand-labelled S1 chips:    {sum(s1_events.values()):,}")
    print(f"Hand-labelled label masks: {sum(label_events.values()):,}")
    print(f"Distinct event regions:    {len(s1_events)}\n")

    cols = [s for s in ("train", "valid", "test", "bolivia") if s in split_events]
    head = f"{'Event region':<16} {'chips':>6}  " + "".join(f"{c:>9}" for c in cols)
    print(head)
    print("-" * len(head))
    for event, count in sorted(s1_events.items()):
        row = "".join(f"{split_events[c].get(event, 0):>9}" for c in cols)
        print(f"{event:<16} {count:>6}  {row}")

    # --- question 1 -----------------------------------------------------
    print("\n" + "=" * 78)
    india = [e for e in s1_events if "india" in e.lower()]
    if india:
        total = sum(s1_events[e] for e in india)
        print(f"  Q1  INDIA EVENT: CONFIRMED — {', '.join(india)}, {total} chips")
        print(f"      ~{total * 26:,} km² of hand-labelled Sentinel-1 water masks.")
    else:
        print("  Q1  INDIA EVENT: NOT FOUND — ADR-007's assumption is wrong.")
        print("      bihar_ganga has no hand-labelled training data. Switch the")
        print("      primary AOI to assam_brahmaputra (Copernicus EMS) or treat")
        print("      every Indian AOI as a transfer target.")

    # --- question 2 -----------------------------------------------------
    print("\n" + "-" * 78)
    if verdict == "chip_level_random":
        print("  Q2  SPLIT STRUCTURE: **NOT REGION-DISJOINT**")
        print(f"\n      {len(shared_regions)} regions appear in BOTH train and test:")
        print(f"      {', '.join(shared_regions)}")
        print("\n      These are chip-level stratified random splits. Chips from the")
        print("      same flood event — often the same acquisition, on adjacent")
        print("      ground — sit on both sides of the train/test boundary. Spatial")
        print("      autocorrelation then leaks across it and inflates reported IoU.")
        if excluded:
            print(f"\n      Held out of the main splits entirely: {', '.join(excluded)}")
            print("      That is the dataset's one genuinely region-disjoint test set.")
        print("\n      => SAT-AI must build its own leave-one-region-out splits.")
        print("         See docs/adr/ADR-009-evaluation-splits.md.")
        print("         Published Sen1Floods11 IoU figures use THESE splits and are")
        print("         therefore not comparable to a region-disjoint number.")
    else:
        print("  Q2  SPLIT STRUCTURE: region-disjoint — no region spans train and test.")

    # --- question 3 -----------------------------------------------------
    if footprint_result:
        print("\n" + "-" * 78)
        bbox = footprint_result["union_bbox"]
        crs = ", ".join(footprint_result["crs"])
        print(
            f"  Q3  {footprint_result['region'].upper()} FOOTPRINT "
            f"({footprint_result['chips_read']} chips read, CRS {crs})"
        )
        print(f"\n      union bbox: [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")
        print(f"      spans {bbox[2] - bbox[0]:.2f}° lon x {bbox[3] - bbox[1]:.2f}° lat")
        print("\n      Compare against configs/aoi.yaml. The bihar_ganga bbox there")
        print("      was chosen before this was known — if it does not contain this")
        print("      box, replace it with this one and record the change in ADR-007.")
        if footprint_result["chips_failed"]:
            print(f"\n      {footprint_result['chips_failed']} chips failed to parse:")
            for problem in footprint_result["problems"]:
                print(f"        {problem}")
    elif not args.footprints:
        print("\n  Q3  FOOTPRINTS: not checked. Run with --footprints to locate the")
        print("      labelled chips and confirm the AOI bbox matches them.")

    print("\n" + "=" * 78)
    print(f"  Reference: {REFERENCE}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
