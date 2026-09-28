"""WorldPop population rasters: the exposure term, from an open source.

`R = H^a * E^b * V^g` needs E, and this repository had none. The risk map was
blocked on it, and `ml/flood/to_risk.py` refuses to substitute a constant,
because a flat exposure field turns the risk map into a rescaling of the hazard
map while presenting itself as a multi-factor result.

WorldPop publishes global population-count rasters under CC BY 4.0 with no
account and no key, which makes it the one exposure source this project can
actually use today.

Which product, and why it matters
---------------------------------
The **constrained** 2020 estimates are used, not the unconstrained ones. The
unconstrained product distributes population across every cell the model thinks
is habitable; the constrained product only puts people where built settlement
was actually detected. For flood exposure over a river plain that difference is
the whole question — unconstrained estimates spread population across the
floodplain itself, which would inflate exposure exactly where the hazard is
highest.

What this is not
----------------
A census. It is a modelled surface, dasymetrically redistributed from census
units, and its per-pixel value is an estimate with error that is largest where
settlement is sparse. Every artifact derived from it says so.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from satai.errors import ProviderError, ValidationError
from satai.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "WORLDPOP_CITATION",
    "WORLDPOP_LICENCE",
    "WorldPopProduct",
    "fetch_population",
    "product_for",
]

BASE = "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM"

WORLDPOP_LICENCE = "CC BY 4.0"
WORLDPOP_CITATION = (
    "WorldPop (www.worldpop.org), School of Geography and Environmental Science, "
    "University of Southampton. Global High Resolution Population Denominators "
    "Project (OPP1134076). Constrained individual countries 2020, 100 m."
)

#: ISO3 codes for the countries this project's study areas fall in. Kept
#: explicit rather than derived from a lookup table, so adding a country is a
#: deliberate act that comes with checking the product exists.
COUNTRY_ISO3: dict[str, str] = {"India": "IND", "Nepal": "NPL"}


@dataclass(frozen=True)
class WorldPopProduct:
    """One country-year population raster, and where it came from."""

    iso3: str
    year: int
    url: str
    licence: str = WORLDPOP_LICENCE
    citation: str = WORLDPOP_CITATION

    @property
    def filename(self) -> str:
        return f"{self.iso3.lower()}_ppp_{self.year}_constrained.tif"

    def provenance(self, local_path: Path | None = None) -> dict[str, Any]:
        return {
            "dataset": "WorldPop Global High Resolution Population Denominators",
            "product": f"{self.iso3} {self.year} constrained, 100 m",
            "source_kind": "model",
            "provider": "worldpop",
            "url": self.url,
            "licence": self.licence,
            "citation": self.citation,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "local_path": str(local_path) if local_path else None,
            "caveats": [
                "Modelled population surface, not a census. Values are "
                "dasymetrically redistributed from census units; per-pixel "
                "error is largest where settlement is sparse.",
                "Constrained product: population is placed only where built "
                "settlement was detected. The unconstrained variant would "
                "spread population across the floodplain itself.",
                "2020 estimate. Population has changed since; this is the most "
                "recent year of this product.",
            ],
        }


def product_for(country: str, year: int = 2020) -> WorldPopProduct:
    """The population product covering a study area's country."""
    iso3 = COUNTRY_ISO3.get(country)
    if iso3 is None:
        raise ValidationError(
            f"no WorldPop product configured for {country!r}; known: "
            f"{', '.join(sorted(COUNTRY_ISO3))}"
        )
    return WorldPopProduct(
        iso3=iso3,
        year=year,
        url=f"{BASE}/{iso3}/{iso3.lower()}_ppp_{year}_constrained.tif",
    )


def fetch_population(
    product: WorldPopProduct, destination: Path, *, timeout: float = 300.0
) -> Path:
    """Download the country raster, once.

    Downloaded whole rather than read as a remote window: WorldPop's server
    does not support range requests, so `/vsicurl` cannot open it at all. The
    files are a few megabytes, so this costs little and makes every later run
    offline and reproducible.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        log.info("worldpop raster already present", extra={"path": str(destination)})
        return destination

    request = urllib.request.Request(  # noqa: S310 - fixed https endpoint, not user input
        product.url, headers={"User-Agent": "SAT-AI/0.5 (research; open data)"}
    )
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            partial.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise ProviderError(f"could not fetch {product.url}: {exc}", provider="worldpop") from exc

    # Renamed only once the body is complete, so an interrupted download can
    # never be mistaken for a valid raster on the next run.
    partial.replace(destination)
    log.info(
        "worldpop raster downloaded",
        extra={"path": str(destination), "bytes": destination.stat().st_size},
    )
    return destination
