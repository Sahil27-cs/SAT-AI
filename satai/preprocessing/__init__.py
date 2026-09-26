"""Preprocessing shared by the ML plane and the inference path.

Two tracks, per ADR-010:

**Track A — training.** Sen1Floods11 ships chips its authors already calibrated
and terrain-corrected, so training needs no SAR preprocessing chain: chip
loading, splitting, normalisation and band math. All of it is AOI-independent,
which is why it is built before the study area is chosen.

**Track B — inference.** Running over a live AOI does start from raw GRD and
needs the full chain. It is built after the AOI is selected, and its
specification is *derived from* Track A's measured characteristics rather than
chosen independently — because a model trained on one radiometric convention
and run on another produces confident, plausible, wrong flood masks.

What lives here is what both tracks must agree on. Putting it in ``satai``
rather than ``ml`` is the same reasoning as the provenance contract: defined
once, so it cannot drift between training and serving.
"""

from __future__ import annotations

from satai.preprocessing.indices import (
    db_to_linear,
    dnbr,
    linear_to_db,
    log_ratio_db,
    mndwi,
    nbr,
    ndvi,
    ndwi,
    normalized_difference,
    sar_ratio_db,
)
from satai.preprocessing.normalize import BandStats, StackNormalizer, fit_band_stats
from satai.preprocessing.splits import (
    BOLIVIA,
    ChipRef,
    Fold,
    SplitProtocol,
    leave_one_region_out,
    official_fold,
    parse_chip_name,
)

__all__ = [
    "BOLIVIA",
    "BandStats",
    "ChipRef",
    "Fold",
    "SplitProtocol",
    "StackNormalizer",
    "db_to_linear",
    "dnbr",
    "fit_band_stats",
    "leave_one_region_out",
    "linear_to_db",
    "log_ratio_db",
    "mndwi",
    "nbr",
    "ndvi",
    "ndwi",
    "normalized_difference",
    "official_fold",
    "parse_chip_name",
    "sar_ratio_db",
]
