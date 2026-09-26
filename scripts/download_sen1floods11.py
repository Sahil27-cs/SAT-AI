"""Download the Sen1Floods11 hand-labelled subset.

    python scripts/download_sen1floods11.py --out data/raw/sen1floods11

446 chips over 11 flood events, roughly 700 MB for the two products this
project needs:

    S1Hand     Sentinel-1 VV/VH backscatter -- the model input
    LabelHand  hand-drawn water masks -- the ground truth

The bucket is public and needs no credentials. Downloads are resumable: a file
already present at the expected byte size is skipped, so an interrupted run can
simply be repeated.

Why a script rather than a one-off command
------------------------------------------
Every number this project reports has to be reproducible from a stated
procedure. "I downloaded the data" is not a procedure; this file is. It also
records what was fetched into a manifest, so a later run can tell whether the
inputs behind a metric have changed.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BUCKET = "sen1floods11"
LIST_API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
DOWNLOAD_BASE = f"https://storage.googleapis.com/{BUCKET}"

#: The two products needed for supervised flood segmentation. S2Hand,
#: JRCWaterHand and the weak-labelled splits exist in the same bucket and are
#: deliberately not fetched -- they are a further 10+ GB this project has no
#: use for yet, and downloading data you will not use is how a student disk
#: fills up two days before a deadline.
PREFIXES: tuple[str, ...] = (
    "v1.1/data/flood_events/HandLabeled/S1Hand/",
    "v1.1/data/flood_events/HandLabeled/LabelHand/",
)


def list_objects(prefix: str) -> list[dict[str, Any]]:
    """Every object under ``prefix``, following pagination."""
    items: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        query = {"prefix": prefix, "maxResults": "1000"}
        if token:
            query["pageToken"] = token
        with urllib.request.urlopen(  # noqa: S310 - fixed https host
            f"{LIST_API}?{urllib.parse.urlencode(query)}", timeout=60
        ) as response:
            page = json.loads(response.read())
        items.extend(page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            return items


def fetch(obj: dict[str, Any], out_root: Path) -> tuple[str, bool]:
    """Download one object. Returns (name, downloaded) -- False means skipped."""
    name = obj["name"]
    expected = int(obj["size"])
    destination = out_root / Path(name).name

    # Size is the resume check. It is not a checksum and does not pretend to be
    # one; the bucket publishes md5Hash and the manifest records it, so a real
    # integrity check is possible later without re-downloading everything now.
    if destination.exists() and destination.stat().st_size == expected:
        return name, False

    url = f"{DOWNLOAD_BASE}/{urllib.parse.quote(name)}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310
        temporary.write_bytes(response.read())
    # Written to .part and renamed, so an interrupted run never leaves a
    # truncated file that the size check would then accept as complete.
    temporary.replace(destination)
    return name, True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/sen1floods11"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args(argv)

    manifest: dict[str, Any] = {
        "source": f"gs://{BUCKET}",
        "prefixes": list(PREFIXES),
        "fetched_at": datetime.now(UTC).isoformat(),
        "products": {},
    }
    total_bytes = 0

    for prefix in PREFIXES:
        product = prefix.rstrip("/").split("/")[-1]
        objects = list_objects(prefix)
        size = sum(int(o["size"]) for o in objects)
        total_bytes += size
        print(f"{product:12} {len(objects):4} files  {size / 1_048_576:8.1f} MB")

        manifest["products"][product] = {
            "count": len(objects),
            "bytes": size,
            # Recorded per file so integrity can be verified later without
            # another full download.
            "md5": {Path(o["name"]).name: o.get("md5Hash") for o in objects},
        }
        if args.list_only:
            continue

        out_root = args.out / product
        out_root.mkdir(parents=True, exist_ok=True)
        downloaded = skipped = failed = 0

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch, o, out_root): o for o in objects}
            for future in as_completed(futures):
                try:
                    _, did_download = future.result()
                except (urllib.error.URLError, OSError) as exc:
                    failed += 1
                    print(f"  FAILED {futures[future]['name']}: {exc}", file=sys.stderr)
                    continue
                if did_download:
                    downloaded += 1
                else:
                    skipped += 1
                done = downloaded + skipped + failed
                if done % 50 == 0 or done == len(objects):
                    print(f"  {done}/{len(objects)}", end="\r", flush=True)

        print(f"  {downloaded} downloaded, {skipped} already present, {failed} failed")
        if failed:
            print(f"  re-run to retry the {failed} that failed", file=sys.stderr)

    print(f"total {total_bytes / 1_048_576:.1f} MB")
    if not args.list_only:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"manifest written to {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
