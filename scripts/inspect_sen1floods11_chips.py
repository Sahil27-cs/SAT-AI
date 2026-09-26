#!/usr/bin/env python3
"""Establish Sen1Floods11's chip structure from the files themselves.

ADR-010 makes the inference pipeline's specification *derived from* the training
data rather than chosen independently, because a model trained on one
radiometric convention and run on another produces confident, plausible, wrong
flood masks. That derivation needs facts, and this script collects them:

* band count, data type and compression per product type
* CRS and pixel size -- geographic or projected, and at what resolution
* nodata convention
* chip dimensions, and the footprint the chips actually cover

Everything comes from TIFF headers read over HTTP range requests, so the whole
survey costs a few megabytes rather than the ~200 MB of rasters.

**Why this is a script and not an assumption.** Phase 2 already caught one
claim this project had carried from recollection rather than evidence: that
Sen1Floods11's splits were region-disjoint. They are not. The band layout and
units are the next thing it would be easy to assume, and the next thing that
would be expensive to get wrong -- normalisation, the dB-domain band math in
``satai.preprocessing.indices``, and the whole Track B specification all depend
on it.

Usage
-----
    python scripts/inspect_sen1floods11_chips.py
    python scripts/inspect_sen1floods11_chips.py --region India --sample 20
    python scripts/inspect_sen1floods11_chips.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satai.errors import ValidationError  # noqa: E402
from satai.geo.tiff_header import GeoTIFFHeader, parse_geotiff_header, union_bbox  # noqa: E402

BUCKET = "sen1floods11"
LIST_API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
OBJECT_URL = f"https://storage.googleapis.com/{BUCKET}/"

#: The three hand-labelled products. S1Hand is the model input, LabelHand the
#: target; S2Hand is the optional optical modality for the fusion ablation.
PRODUCTS = {
    "S1Hand": "v1.1/data/flood_events/HandLabeled/S1Hand/",
    "S2Hand": "v1.1/data/flood_events/HandLabeled/S2Hand/",
    "LabelHand": "v1.1/data/flood_events/HandLabeled/LabelHand/",
}

HEADER_BYTES = 65_536


def _get(url: str, *, range_bytes: int | None = None, timeout: int = 90) -> bytes:
    headers = {"User-Agent": "SAT-AI/0.1"}
    if range_bytes is not None:
        headers["Range"] = f"bytes=0-{range_bytes - 1}"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        data: bytes = response.read()
    return data


def list_objects(prefix: str, *, max_pages: int = 60) -> list[str]:
    names: list[str] = []
    page_token: str | None = None
    for _ in range(max_pages):
        params: dict[str, str] = {"prefix": prefix, "maxResults": "1000"}
        if page_token:
            params["pageToken"] = page_token
        payload = json.loads(_get(f"{LIST_API}?{urllib.parse.urlencode(params)}").decode())
        names.extend(
            item["name"] for item in payload.get("items", []) if item["name"].endswith(".tif")
        )
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return names


def region_of(name: str) -> str:
    return name.rsplit("/", 1)[-1].split("_", 1)[0]


def survey_product(
    product: str, names: list[str], *, sample: int, region: str | None
) -> dict[str, Any]:
    """Read headers for a sample of one product's chips and summarise them."""
    selected = [n for n in names if region is None or region_of(n) == region]
    selected = sorted(selected)[:sample]

    headers: list[GeoTIFFHeader] = []
    problems: list[str] = []

    for i, name in enumerate(selected, start=1):
        try:
            headers.append(
                parse_geotiff_header(
                    _get(f"{OBJECT_URL}{urllib.parse.quote(name)}", range_bytes=HEADER_BYTES)
                )
            )
        except (ValidationError, urllib.error.URLError, TimeoutError) as exc:
            problems.append(f"{name.rsplit('/', 1)[-1]}: {exc}")
        if i % 10 == 0 or i == len(selected):
            print(f"    {product}: {i}/{len(selected)}", flush=True)

    if not headers:
        return {"product": product, "chips_read": 0, "problems": problems[:5]}

    boxes = [h.bbox for h in headers]
    return {
        "product": product,
        "chips_available": len(names),
        "chips_read": len(headers),
        "chips_failed": len(problems),
        "bands": sorted({h.bands for h in headers}),
        "dtype": sorted({h.dtype or "unknown" for h in headers}),
        "bits_per_sample": sorted({h.bits_per_sample or 0 for h in headers}),
        "compression": sorted({h.compression or "unknown" for h in headers}),
        "nodata": sorted({h.nodata or "(not set)" for h in headers}),
        "size_px": sorted({(h.width, h.height) for h in headers}),
        "crs": sorted({f"EPSG:{h.epsg}" if h.epsg else "unknown" for h in headers}),
        "pixel_size": sorted({round(h.pixel_size_x, 8) for h in headers}),
        "union_bbox": list(union_bbox(boxes)),
        "problems": problems[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=None, help="Limit to one event region.")
    parser.add_argument("--sample", type=int, default=12, help="Chips per product (default 12).")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print("\nListing hand-labelled products...\n")
    try:
        listings = {p: list_objects(prefix) for p, prefix in PRODUCTS.items()}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Could not list the bucket: {exc}", file=sys.stderr)
        return 1

    print(f"Reading headers ({HEADER_BYTES // 1024} KB per chip, range requests)...\n")
    surveys = {
        product: survey_product(product, names, sample=args.sample, region=args.region)
        for product, names in listings.items()
        if names
    }

    regions = Counter(region_of(n) for n in listings.get("S1Hand", []))
    per_region_bbox: dict[str, Any] = defaultdict(dict)

    result: dict[str, Any] = {
        "bucket": BUCKET,
        "region_filter": args.region,
        "sample_per_product": args.sample,
        "regions": dict(sorted(regions.items())),
        "products": surveys,
        "per_region": dict(per_region_bbox),
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("=" * 78)
    print("  Sen1Floods11 — chip structure (measured, not assumed)")
    if args.region:
        print(f"  Region filter: {args.region}")
    print("=" * 78)

    for product, survey in surveys.items():
        if not survey.get("chips_read"):
            print(f"\n{product}: no chips read. {survey.get('problems')}")
            continue
        print(f"\n{product}  ({survey['chips_read']} of {survey['chips_available']} read)")
        print("-" * 78)
        for field in ("bands", "dtype", "compression", "nodata", "size_px", "crs", "pixel_size"):
            values = survey[field]
            flag = "" if len(values) == 1 else "   <-- NOT UNIFORM"
            print(f"  {field:<16} {', '.join(str(v) for v in values)}{flag}")
        bbox = survey["union_bbox"]
        print(f"  {'union_bbox':<16} [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")
        if survey["chips_failed"]:
            print(f"  {'failed':<16} {survey['chips_failed']}: {survey['problems'][0]}")

    # --- what the numbers mean for the pipeline -------------------------
    print("\n" + "=" * 78)
    print("  Implications")
    print("=" * 78)

    s1 = surveys.get("S1Hand", {})
    label = surveys.get("LabelHand", {})

    if s1.get("bands"):
        n_bands = s1["bands"][0] if len(s1["bands"]) == 1 else None
        if n_bands == 2:
            print("  • S1Hand has 2 bands — VV and VH, as expected. sar_ratio_db()")
            print("    applies directly.")
        elif n_bands:
            print(f"  • S1Hand has {n_bands} bands. Confirm the band order before")
            print("    assuming band 0 is VV — the dB band math depends on it.")
        else:
            print("  • S1Hand band count is NOT uniform across chips. Investigate")
            print("    before building a loader that assumes a fixed layout.")

    if s1.get("dtype"):
        print(f"  • S1Hand dtype {', '.join(s1['dtype'])}. Float means the values are")
        print("    already calibrated (dB), not raw DN — so no scaling factor is")
        print("    needed, and the dB-domain guards in indices.py apply.")

    if s1.get("crs"):
        crs = s1["crs"][0] if len(s1["crs"]) == 1 else None
        if crs == "EPSG:4326":
            print("  • Chips are in EPSG:4326 (lon/lat). Track B must therefore")
            print("    deliver its AOI stack in the same CRS, or reproject to match.")
        elif crs:
            print(f"  • Chips are in {crs} (projected). Note the AOI grid in ADR-007")
            print("    assumes a UTM zone — confirm they agree.")

    if label.get("dtype"):
        print(
            f"  • LabelHand dtype {', '.join(label['dtype'])}, nodata "
            f"{', '.join(label['nodata'])}. Sen1Floods11 labels use -1 for"
        )
        print("    'not labelled', which must be EXCLUDED from the loss, not")
        print("    treated as a negative class. Confirm the value above.")

    print("\n  Record these in ADR-010 as the Track A specification, then build")
    print("  Track B to match them. Do not infer any of this from a paper.")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
