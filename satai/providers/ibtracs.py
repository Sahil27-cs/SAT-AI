"""IBTrACS best tracks: observed cyclone positions, never predicted ones.

`satai.hazards.extreme_weather.track_exposure` has always required a track to be
*supplied* — its first line refuses to run without one, because SAT-AI does not
forecast cyclones and a parametric wind field built from an invented track would
look exactly like one built from an observed track.

This module supplies them, from NOAA's International Best Track Archive for
Climate Stewardship: the WMO-endorsed consolidated record of historical
cyclones, open, no account, no key.

Best track, and what that means
-------------------------------
A best track is a *post-season reanalysis*, not the operational advisory issued
at the time. It is the most accurate record of where a storm actually went, and
it is not what was known while the storm was happening. Anything derived from it
is a hindcast, and every artifact here says so — "historical analysis", never
"prediction".

The wind column, and why it is converted
----------------------------------------
IBTrACS reports `WMO_WIND` in knots, as a 10-minute or 1-minute sustained wind
depending on the reporting agency. `TrackPoint` wants metres per second. The
conversion happens here, once, rather than at each call site where a missed
factor of 0.514 would silently halve every wind field.
"""

from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from satai.errors import ProviderError, ValidationError
from satai.hazards.extreme_weather import TrackPoint
from satai.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "IBTRACS_CITATION",
    "IBTRACS_URL",
    "KNOTS_TO_MS",
    "CycloneTrack",
    "fetch_basin",
    "load_tracks",
]

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.{basin}.list.v04r01.csv"
)

IBTRACS_CITATION = (
    "Knapp, K. R., M. C. Kruk, D. H. Levinson, H. J. Diamond, and C. J. Neumann "
    "(2010): The International Best Track Archive for Climate Stewardship "
    "(IBTrACS). Bull. Amer. Meteor. Soc., 91, 363-376. NOAA NCEI, v04r01."
)

#: One knot in metres per second, exactly. Applied once, here.
KNOTS_TO_MS = 0.514444

#: Nominal radius of maximum wind where IBTrACS does not report one. Stated as
#: an assumption in the artifact rather than buried: a real RMW varies from
#: ~15 km in an intense compact storm to over 100 km in a broad weak one, and
#: this single value is the largest approximation in the exposure field.
DEFAULT_RMW_KM = 50.0


@dataclass(frozen=True)
class CycloneTrack:
    """One storm's observed track."""

    sid: str
    name: str
    season: int
    basin: str
    points: list[TrackPoint]
    times: list[datetime]

    @property
    def peak_wind_ms(self) -> float:
        return max((p.max_wind_ms for p in self.points), default=0.0)

    @property
    def landfall_window(self) -> tuple[datetime, datetime] | None:
        if not self.times:
            return None
        return min(self.times), max(self.times)

    def provenance(self) -> dict[str, Any]:
        window = self.landfall_window
        return {
            "dataset": "IBTrACS v04r01",
            "source_kind": "observation",
            "provider": "noaa_ncei",
            "storm_id": self.sid,
            "storm_name": self.name,
            "season": self.season,
            "basin": self.basin,
            "n_fixes": len(self.points),
            "first_fix": window[0].isoformat() if window else None,
            "last_fix": window[1].isoformat() if window else None,
            "peak_wind_ms": round(self.peak_wind_ms, 2),
            "citation": IBTRACS_CITATION,
            "caveats": [
                "Best track: a post-season reanalysis, not the advisory issued "
                "at the time. Anything derived from it is a hindcast.",
                f"Radius of maximum wind defaults to {DEFAULT_RMW_KM:.0f} km "
                "where IBTrACS reports none. This is the largest approximation "
                "in any wind field built from this track.",
                "Wind converted from knots at 0.514444 m/s per knot. Agencies "
                "report 1-minute or 10-minute sustained winds; IBTrACS does "
                "not harmonise them, so the two are mixed across a basin.",
            ],
        }


def fetch_basin(basin: str, destination: Path, *, timeout: float = 300.0) -> Path:
    """Download one basin's full track archive, once.

    `basin` follows IBTrACS: NI = North Indian, which covers the Bay of Bengal
    and the Arabian Sea, and therefore every cyclone that reaches India.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    url = IBTRACS_URL.format(basin=basin)
    request = urllib.request.Request(url, headers={"User-Agent": "SAT-AI/0.5 (research)"})  # noqa: S310 - fixed https endpoint, not user input
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            partial.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise ProviderError(f"could not fetch {url}: {exc}", provider="ibtracs") from exc

    partial.replace(destination)
    log.info(
        "ibtracs basin downloaded",
        extra={"basin": basin, "bytes": destination.stat().st_size},
    )
    return destination


def _parse_float(value: str) -> float | None:
    """IBTrACS writes missing values as an empty field or a space."""
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_tracks(
    path: Path,
    *,
    name: str | None = None,
    season: int | None = None,
    min_wind_ms: float = 0.0,
) -> list[CycloneTrack]:
    """Read storms out of a basin file, filtered.

    Fixes with no reported wind are dropped rather than zero-filled: a zero
    would place a real position in the wind field with no wind at it, which
    reads as a storm that was there and harmless.
    """
    if not path.is_file():
        raise ValidationError(f"no IBTrACS file at {path}; fetch_basin() first")

    text = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    # IBTrACS carries a second header row of units directly under the names.
    rows = [row for row in reader if (row.get("SID") or "").strip() not in ("", "SID")]

    storms: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = row["SID"].strip()
        if name and (row.get("NAME") or "").strip().upper() != name.upper():
            continue
        if season is not None and (row.get("SEASON") or "").strip() != str(season):
            continue

        lat, lon = _parse_float(row.get("LAT", "")), _parse_float(row.get("LON", ""))
        wind_kt = _parse_float(row.get("WMO_WIND", ""))
        if lat is None or lon is None or wind_kt is None:
            continue

        wind_ms = wind_kt * KNOTS_TO_MS
        if wind_ms < min_wind_ms:
            continue

        try:
            when = datetime.fromisoformat(row["ISO_TIME"].strip()).replace(tzinfo=UTC)
        except (ValueError, KeyError):
            continue

        record = storms.setdefault(
            sid,
            {
                "name": (row.get("NAME") or "UNNAMED").strip(),
                "season": int((row.get("SEASON") or "0").strip() or 0),
                "basin": (row.get("BASIN") or "").strip(),
                "points": [],
                "times": [],
            },
        )
        rmw_nm = _parse_float(row.get("USA_RMW", ""))
        record["points"].append(
            TrackPoint(
                lon=lon,
                lat=lat,
                max_wind_ms=wind_ms,
                # USA_RMW is in nautical miles where present.
                radius_max_wind_km=(rmw_nm * 1.852) if rmw_nm and rmw_nm > 0 else DEFAULT_RMW_KM,
            )
        )
        record["times"].append(when)

    return [
        CycloneTrack(
            sid=sid,
            name=r["name"],
            season=r["season"],
            basin=r["basin"],
            points=r["points"],
            times=r["times"],
        )
        for sid, r in sorted(storms.items())
    ]
