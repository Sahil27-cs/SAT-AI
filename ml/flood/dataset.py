"""Torch dataset over Sen1Floods11 chips.

Wraps :class:`satai.preprocessing.chips.Sen1Floods11Dataset`, which is
framework-free on purpose so the classical baselines and the deep model read
the data through exactly the same path -- a difference in their scores then
cannot be a difference in how the bytes reached them.

Three things this layer adds, each a place where leakage or a silent NaN would
otherwise enter:

*Normalisation fitted on the training partition only.* Statistics come from
:class:`satai.preprocessing.normalize.StackNormalizer`, which refuses to fit on
anything but ``train`` and refuses to apply one fold's statistics to another.

*A validity mask that survives to the loss.* Unannotated pixels (-1) and
non-finite feature pixels are masked. The mask is returned as a tensor rather
than applied and forgotten, because the loss needs it too -- roughly a third of
every chip is unannotated, and the sentinel is -1, not NaN, so a loss that does
not mask consumes it silently.

*Augmentation that respects SAR geometry.* Flips and 90-degree rotations only.
No brightness or contrast jitter: those are photometric operations, and
backscatter in decibels is a calibrated physical measurement, not an image
channel. Shifting it by a random offset would teach the model that -12 dB and
-6 dB are interchangeable, which is precisely the distinction it must learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from satai.preprocessing.chips import ChipSpec, Sen1Floods11Dataset
from satai.preprocessing.normalize import StackNormalizer
from satai.preprocessing.splits import ChipRef

#: Bands in stacking order. The ratio is derived rather than read: VV minus VH
#: in decibels separates rough water from smooth land better than either alone.
SAR_BANDS: tuple[str, ...] = ("vv_db", "vh_db")
DERIVED_RATIO = "vv_vh_ratio"


def bands_for(with_ratio: bool) -> tuple[str, ...]:
    return (*SAR_BANDS, DERIVED_RATIO) if with_ratio else SAR_BANDS


def build_reader(root: Path) -> Sen1Floods11Dataset:
    """Reader for the flat layout scripts/download_sen1floods11.py produces.

    The upstream archive nests products under v1.1/data/flood_events/HandLabeled/;
    the download script flattens that to S1Hand/ and LabelHand/ because nothing
    else in this project needs the other directories. Subdirectories are passed
    explicitly so the two layouts cannot be silently confused.
    """
    return Sen1Floods11Dataset(
        root=root,
        spec=ChipSpec(),
        s1_subdir="S1Hand",
        s2_subdir="S2Hand",
        label_subdir="LabelHand",
    )


def stack_with_ratio(
    features: npt.NDArray[np.floating], with_ratio: bool
) -> npt.NDArray[np.floating]:
    """Append the VV/VH dB ratio band, if requested.

    Computed before normalisation so the ratio is a physical quantity that then
    gets its own fitted statistics. Taking the difference of two already
    standardised bands would produce a number with no units and no meaning.
    """
    if not with_ratio:
        return features
    ratio = features[0] - features[1]
    return np.concatenate([features, ratio[None, ...]], axis=0)


class FloodChips(Dataset):
    """One Sen1Floods11 partition, as tensors.

    Yields ``(features, target, valid)``:

        features  (C, H, W) float32, normalised
        target    (1, H, W) float32 in {0, 1}
        valid     (1, H, W) float32 in {0, 1}
    """

    def __init__(
        self,
        chips: Sequence[ChipRef],
        root: Path,
        normalizer: StackNormalizer,
        *,
        with_ratio: bool = True,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.reader = build_reader(root)
        # Chips missing either product are dropped here rather than raising
        # mid-epoch: a partially downloaded dataset should shrink the run, not
        # kill it at step 300.
        self.chips = self.reader.available(list(chips))
        self.normalizer = normalizer
        self.with_ratio = with_ratio
        self.augment = augment
        self._rng = np.random.default_rng(seed)

        if not self.chips:
            raise ValueError(
                f"no usable chips under {root}: every entry is missing its S1 image "
                f"or its label. Run scripts/download_sen1floods11.py"
            )

    @property
    def bands(self) -> tuple[str, ...]:
        return bands_for(self.with_ratio)

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chip = self.chips[index]
        features, labels, valid = self.reader.load(chip)

        features = stack_with_ratio(features, self.with_ratio)
        features = self.normalizer.transform_stack(features, list(self.bands))

        # Normalisation leaves NaN as NaN; the network cannot. Zero is the
        # post-standardisation mean, so a masked pixel contributes the band
        # average, and the mask keeps it out of the loss regardless.
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        stack = features.astype(np.float32)
        target = (labels == 1).astype(np.float32)[None, ...]
        valid_mask = valid.astype(np.float32)[None, ...]

        if self.augment:
            stack, target, valid_mask = self._augment(stack, target, valid_mask)

        return (
            torch.from_numpy(np.ascontiguousarray(stack)),
            torch.from_numpy(np.ascontiguousarray(target)),
            torch.from_numpy(np.ascontiguousarray(valid_mask)),
        )

    def _augment(
        self,
        features: npt.NDArray[np.floating],
        target: npt.NDArray[np.floating],
        valid: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Flips and quarter turns, applied identically to all three arrays.

        Geometric only -- see the module docstring on why photometric jitter is
        wrong for calibrated backscatter.
        """
        if self._rng.random() < 0.5:
            features, target, valid = (np.flip(a, axis=-1) for a in (features, target, valid))
        if self._rng.random() < 0.5:
            features, target, valid = (np.flip(a, axis=-2) for a in (features, target, valid))
        turns = int(self._rng.integers(0, 4))
        if turns:
            features, target, valid = (
                np.rot90(a, turns, axes=(-2, -1)) for a in (features, target, valid)
            )
        return features, target, valid


def fit_normalizer(
    chips: Sequence[ChipRef],
    root: Path,
    fold: str,
    *,
    with_ratio: bool = True,
    max_chips: int = 120,
    seed: int = 42,
) -> StackNormalizer:
    """Fit per-band statistics on the TRAINING chips of one fold.

    Sampled rather than exhaustive: 120 chips is roughly 31 million pixels per
    band, well past the point where another chip moves a percentile. The cap
    keeps fitting to seconds, and the sample is seeded so the statistics are
    reproducible.

    Passing validation or test chips here is refused downstream by ``BandStats``,
    which records the partition it was fitted on and rejects anything but train.
    """
    reader = build_reader(root)
    usable = reader.available(list(chips))
    if not usable:
        raise ValueError(f"no usable training chips under {root} for fold {fold!r}")

    rng = np.random.default_rng(seed)
    if len(usable) > max_chips:
        picked = sorted(rng.choice(len(usable), max_chips, replace=False).tolist())
        usable = [usable[i] for i in picked]

    bands = bands_for(with_ratio)
    samples: dict[str, list[npt.NDArray[np.floating]]] = {b: [] for b in bands}

    for chip in usable:
        features, _, valid = reader.load(chip)
        stack = stack_with_ratio(features, with_ratio)
        for i, band in enumerate(bands):
            # Only pixels that are both finite and annotated. Fitting on the
            # border no-data sentinel would drag every percentile downward.
            samples[band].append(stack[i][valid])

    normalizer = StackNormalizer(fold=fold)
    for band in bands:
        normalizer.fit_band(band, np.concatenate(samples[band]), partition="train")
    return normalizer
