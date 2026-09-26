"""Track A: Sen1Floods11 chip loading (ADR-010).

Sen1Floods11 ships chips its authors already calibrated and terrain-corrected,
so this module does **no SAR preprocessing**. Applying an orbit correction or a
speckle filter here would be processing already-processed data, which is worse
than doing nothing: it would shift the training distribution away from what the
dataset's own labels were drawn against.

What it does do is handle the two things that silently corrupt flood metrics:

1. **The ignore label.** Unannotated pixels carry a sentinel rather than a
   class. They must be excluded from both the loss and the metrics. Counting
   them as "not water" inflates every number, because most of them are not.
2. **Band identity.** Band order is read from configuration, never assumed.
   Swapping VV and VH produces a model that trains happily and means nothing.

``rasterio`` is imported lazily so this module can be imported, inspected and
unit-tested without the geospatial stack installed -- which is what keeps CI
light (ADR-004).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from satai.errors import DataQualityError, ValidationError
from satai.logging import get_logger
from satai.preprocessing.splits import ChipRef

log = get_logger(__name__)

__all__ = ["ChipSpec", "Sen1Floods11Dataset", "load_chip_array"]

FloatArray = npt.NDArray[np.floating]

#: Sentinel for "not annotated". Confirmed against the data by
#: `scripts/inspect_sen1floods11_chips.py`, not assumed from the paper.
IGNORE_LABEL = -1


@dataclass(frozen=True)
class ChipSpec:
    """The measured structure of the dataset's chips.

    Every field here is something `inspect_sen1floods11_chips.py` reports from
    the files. The defaults encode the expected Sen1Floods11 layout, and
    :meth:`validate_against` fails loudly if the data on disk disagrees --
    because a silent band-order or unit mismatch is exactly the class of error
    that produces a model which trains cleanly and predicts nonsense.
    """

    s1_bands: tuple[str, ...] = ("VV", "VH")
    s1_units: str = "dB"
    s2_bands: tuple[str, ...] = (
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
        "B1",
    )
    label_values: tuple[int, ...] = (0, 1)
    ignore_label: int = IGNORE_LABEL
    chip_size: int = 512
    expected_crs: str | None = None  # measured; None means "do not enforce"

    def validate_against(self, measured: dict[str, Any]) -> list[str]:
        """Compare the declared spec with a measured survey. Returns problems."""
        problems: list[str] = []
        bands = measured.get("bands", [])
        if bands and len(bands) == 1 and bands[0] != len(self.s1_bands):
            problems.append(
                f"spec declares {len(self.s1_bands)} S1 bands {self.s1_bands} but the "
                f"data has {bands[0]}. Confirm band order before training -- swapping "
                f"VV and VH trains cleanly and means nothing."
            )
        sizes = measured.get("size_px", [])
        if sizes and len(sizes) == 1 and tuple(sizes[0]) != (self.chip_size, self.chip_size):
            problems.append(f"spec declares {self.chip_size}px chips; data has {sizes[0]}")
        return problems


def load_chip_array(path: Path) -> FloatArray:
    """Read a GeoTIFF as ``(band, y, x)`` float64. Requires rasterio."""
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ValidationError(
            "rasterio is required to read chips. Install the conda environment: "
            "conda env create -f environment.yml (see ADR-004)"
        ) from exc

    if not path.is_file():
        raise DataQualityError(f"chip not found: {path}", path=str(path))

    with rasterio.open(path) as src:
        return np.asarray(src.read(), dtype=np.float64)


@dataclass
class Sen1Floods11Dataset:
    """Chips on disk, addressed by :class:`ChipRef`.

    Deliberately not a ``torch.utils.data.Dataset``: keeping it framework-free
    means the baselines (scikit-learn, thresholding) and the deep model consume
    exactly the same loading path, so a difference in their results cannot be a
    difference in how the data reached them.
    """

    root: Path
    spec: ChipSpec = field(default_factory=ChipSpec)
    s1_subdir: str = "v1.1/data/flood_events/HandLabeled/S1Hand"
    s2_subdir: str = "v1.1/data/flood_events/HandLabeled/S2Hand"
    label_subdir: str = "v1.1/data/flood_events/HandLabeled/LabelHand"

    def s1_path(self, chip: ChipRef) -> Path:
        return self.root / self.s1_subdir / chip.filename("S1Hand")

    def s2_path(self, chip: ChipRef) -> Path:
        return self.root / self.s2_subdir / chip.filename("S2Hand")

    def label_path(self, chip: ChipRef) -> Path:
        return self.root / self.label_subdir / chip.filename("LabelHand")

    def available(self, chips: Sequence[ChipRef]) -> list[ChipRef]:
        """Chips whose S1 image and label are both present."""
        return [c for c in chips if self.s1_path(c).is_file() and self.label_path(c).is_file()]

    def discover(self) -> list[ChipRef]:
        """Enumerate chips from the filesystem."""
        from satai.preprocessing.splits import parse_chip_name

        s1_dir = self.root / self.s1_subdir
        if not s1_dir.is_dir():
            raise DataQualityError(
                f"Sen1Floods11 S1 directory not found: {s1_dir}. Download the dataset "
                f"first (scripts/download_sen1floods11.py).",
                path=str(s1_dir),
            )
        return sorted(parse_chip_name(p.name) for p in s1_dir.glob("*.tif"))

    def load(
        self, chip: ChipRef, *, with_optical: bool = False
    ) -> tuple[FloatArray, npt.NDArray[np.int_], npt.NDArray[np.bool_]]:
        """Return ``(features, labels, valid_mask)`` for one chip.

        ``valid_mask`` is False wherever the label is the ignore sentinel or
        any feature band is non-finite. Both must be excluded: a NaN feature
        propagates into the loss and a NaN loss destroys the whole batch.
        """
        s1 = load_chip_array(self.s1_path(chip))
        if s1.shape[0] != len(self.spec.s1_bands):
            raise DataQualityError(
                f"{chip.key}: expected {len(self.spec.s1_bands)} S1 bands "
                f"{self.spec.s1_bands}, found {s1.shape[0]}",
                chip=chip.key,
            )

        features = s1
        if with_optical:
            s2 = load_chip_array(self.s2_path(chip))
            if s2.shape[1:] != s1.shape[1:]:
                raise DataQualityError(
                    f"{chip.key}: optical grid {s2.shape[1:]} does not match SAR {s1.shape[1:]}",
                    chip=chip.key,
                )
            features = np.concatenate([s1, s2], axis=0)

        labels = load_chip_array(self.label_path(chip))[0].astype(np.int64)
        valid = (labels != self.spec.ignore_label) & np.all(np.isfinite(features), axis=0)

        if not valid.any():
            log.warning("chip has no valid pixels", extra={"chip": chip.key})

        return features, labels, valid

    def iter_chips(
        self, chips: Sequence[ChipRef], *, with_optical: bool = False
    ) -> Iterator[tuple[ChipRef, FloatArray, npt.NDArray[np.int_], npt.NDArray[np.bool_]]]:
        """Yield loaded chips, skipping any that fail with a logged reason."""
        for chip in chips:
            try:
                features, labels, valid = self.load(chip, with_optical=with_optical)
            except DataQualityError as exc:
                log.warning("skipping chip", extra={"chip": chip.key, "reason": str(exc)})
                continue
            yield chip, features, labels, valid

    def sample_band_values(
        self,
        chips: Sequence[ChipRef],
        *,
        band_index: int,
        max_chips: int = 200,
        per_chip: int = 5_000,
        seed: int = 0,
    ) -> FloatArray:
        """Sample one band's values across chips, for fitting normalisation.

        Subsamples per chip rather than loading everything: fitting percentiles
        needs a representative sample, not every pixel, and the training
        partition of a LORO fold is ~330 chips of 262,144 pixels each.
        """
        rng = np.random.default_rng(seed)
        collected: list[FloatArray] = []

        for chip, features, _labels, valid in self.iter_chips(list(chips)[:max_chips]):
            values = features[band_index][valid]
            values = values[np.isfinite(values)]
            if values.size == 0:
                log.debug("no finite values in band", extra={"chip": chip.key})
                continue
            take = min(per_chip, values.size)
            collected.append(rng.choice(values, take, replace=False))

        if not collected:
            raise DataQualityError("no band values could be sampled from any chip")
        return np.concatenate(collected)
