"""Evaluation splits for Sen1Floods11.

Implements ADR-009. Two protocols, and the whole point is to run both:

**Leave-one-region-out (primary).** Train on 10 event regions, test on the
eleventh. Repeat for every region. This measures generalisation to an unseen
flood, which is the thing an operational system actually has to do.

**Official (secondary).** Sen1Floods11's shipped splits, used so SAT-AI's
numbers can be placed next to published figures — and so the gap between the two
protocols can be reported.

That gap is the reason this module exists in the form it does. The official
splits are chip-level random splits stratified *within* region: verified
2026-09-22, every region except Bolivia appears in train, validation and test at
roughly 58/21/21. Chips are ~5 km across and chips from one flood event share an
acquisition, an orbit and the weather, so training on some and testing on the
rest measures interpolation within an event. ``IoU_official - IoU_LORO``
quantifies what that is worth, and nothing in this project's literature corpus
reports it.

Everything here is pure bookkeeping over chip identifiers -- no raster I/O, no
numpy -- so the splits can be built, inspected and tested before any data is
downloaded.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from satai.errors import ValidationError
from satai.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "BOLIVIA",
    "ChipRef",
    "Fold",
    "SplitProtocol",
    "leave_one_region_out",
    "official_fold",
    "parse_chip_name",
]

#: Held out of Sen1Floods11's three main split files entirely and distributed
#: separately. It is the dataset's one genuinely region-disjoint evaluation, and
#: SAT-AI reserves it as a final untouched check (ADR-009).
BOLIVIA = "Bolivia"


@dataclass(frozen=True, order=True)
class ChipRef:
    """One Sen1Floods11 chip, identified by region and id.

    Ordered so that folds are deterministic: the same inputs always produce the
    same split, which is a precondition for a reproducible result.
    """

    region: str
    chip_id: str

    @property
    def key(self) -> str:
        return f"{self.region}_{self.chip_id}"

    def filename(self, kind: str = "S1Hand") -> str:
        """Filename for a given product, e.g. ``India_902184_S1Hand.tif``."""
        return f"{self.region}_{self.chip_id}_{kind}.tif"


def parse_chip_name(name: str) -> ChipRef:
    """Parse ``<Region>_<id>_<Type>.tif`` (or a bare ``<Region>_<id>``).

    Raises
    ------
    ValidationError
        The name does not carry a region and a chip id.
    """
    base = name.rsplit("/", 1)[-1].removesuffix(".tif")
    parts = base.split("_")
    if len(parts) < 2:
        raise ValidationError(f"cannot parse chip name {name!r}: expected <Region>_<id>[_<Type>]")
    return ChipRef(region=parts[0], chip_id=parts[1])


@dataclass(frozen=True)
class Fold:
    """One train/validation/test partition."""

    name: str
    protocol: str
    train: tuple[ChipRef, ...]
    val: tuple[ChipRef, ...]
    test: tuple[ChipRef, ...]
    test_regions: tuple[str, ...] = ()
    val_regions: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.assert_disjoint()

    def assert_disjoint(self) -> None:
        """Fail loudly if any chip appears in more than one partition.

        Cheap, and it catches the class of mistake that silently inflates every
        downstream number. A leaked chip does not raise anything by itself -- it
        just makes the model look better than it is.
        """
        train, val, test = set(self.train), set(self.val), set(self.test)
        for a_name, a, b_name, b in (
            ("train", train, "val", val),
            ("train", train, "test", test),
            ("val", val, "test", test),
        ):
            overlap = a & b
            if overlap:
                sample = ", ".join(sorted(c.key for c in list(overlap)[:5]))
                raise ValidationError(
                    f"fold {self.name!r}: {len(overlap)} chips appear in both "
                    f"{a_name} and {b_name} ({sample}...)"
                )

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    @property
    def is_region_disjoint(self) -> bool:
        """True when no region spans the train and test partitions."""
        train_regions = {c.region for c in self.train}
        test_regions = {c.region for c in self.test}
        return not (train_regions & test_regions)

    def summary(self) -> str:
        sizes = self.sizes
        disjoint = "region-disjoint" if self.is_region_disjoint else "NOT region-disjoint"
        return (
            f"{self.name} [{self.protocol}, {disjoint}] "
            f"train={sizes['train']} val={sizes['val']} test={sizes['test']}"
        )


class SplitProtocol:
    """Protocol identifiers, recorded in every result so numbers are comparable."""

    LORO = "leave_one_region_out"
    OFFICIAL = "official_sen1floods11"


def leave_one_region_out(
    chips: Sequence[ChipRef],
    *,
    exclude_regions: Sequence[str] = (BOLIVIA,),
) -> Iterator[Fold]:
    """Yield one fold per region: test on it, train on the rest.

    The validation region is chosen **deterministically** as the next region in
    sorted order after the test region, wrapping around. Two reasons: a random
    validation split would make folds irreproducible, and drawing validation
    chips from the training regions at random would put chips from a training
    event into validation, so early stopping would be tuned on data
    statistically identical to the training set. Holding out a whole region for
    validation keeps the model-selection signal honest too.

    Parameters
    ----------
    chips:
        Every chip available.
    exclude_regions:
        Regions withheld from all folds. Defaults to Bolivia, reserved as the
        final untouched hold-out (ADR-009).

    Raises
    ------
    ValidationError
        Fewer than three usable regions -- LORO needs test, validation and at
        least one training region.
    """
    usable = [c for c in chips if c.region not in set(exclude_regions)]
    regions = sorted({c.region for c in usable})

    if len(regions) < 3:
        raise ValidationError(
            f"leave-one-region-out needs at least 3 regions (test, val, train); "
            f"got {len(regions)}: {', '.join(regions) or '(none)'}"
        )

    by_region: dict[str, list[ChipRef]] = {r: [] for r in regions}
    for chip in sorted(usable):
        by_region[chip.region].append(chip)

    for i, test_region in enumerate(regions):
        val_region = regions[(i + 1) % len(regions)]
        train_regions = [r for r in regions if r not in (test_region, val_region)]

        notes = [
            f"test region: {test_region}",
            f"validation region: {val_region} (deterministic: next in sorted order)",
            f"training regions: {', '.join(train_regions)}",
        ]
        if exclude_regions:
            notes.append(f"withheld from all folds: {', '.join(exclude_regions)}")

        yield Fold(
            name=f"loro_{test_region.lower()}",
            protocol=SplitProtocol.LORO,
            train=tuple(c for r in train_regions for c in by_region[r]),
            val=tuple(by_region[val_region]),
            test=tuple(by_region[test_region]),
            test_regions=(test_region,),
            val_regions=(val_region,),
            notes=tuple(notes),
        )


def official_fold(
    train_names: Sequence[str],
    val_names: Sequence[str],
    test_names: Sequence[str],
) -> Fold:
    """Build a fold from Sen1Floods11's shipped split files.

    Used only as the **secondary** protocol, so that SAT-AI's numbers can be
    compared with published figures. The returned fold will report
    ``is_region_disjoint == False``, which is the finding, not a defect -- see
    ADR-009. A warning is logged so the distinction cannot be lost between
    producing a number and reading it.
    """
    fold = Fold(
        name="official",
        protocol=SplitProtocol.OFFICIAL,
        train=tuple(sorted(parse_chip_name(n) for n in train_names)),
        val=tuple(sorted(parse_chip_name(n) for n in val_names)),
        test=tuple(sorted(parse_chip_name(n) for n in test_names)),
        notes=(
            "Sen1Floods11 official splits: chip-level random, stratified within "
            "region. Not region-disjoint -- see ADR-009.",
        ),
    )

    if not fold.is_region_disjoint:
        shared = sorted({c.region for c in fold.train} & {c.region for c in fold.test})
        log.warning(
            "official splits are not region-disjoint",
            extra={"shared_regions": len(shared), "regions": ",".join(shared)},
        )

    return fold


def load_split_file(path: Path) -> list[str]:
    """Read chip names from one of Sen1Floods11's split CSVs.

    Rows pair an image with its label; only the first column is needed here.
    """
    if not path.is_file():
        raise ValidationError(f"split file not found: {path}")
    return [
        line.split(",")[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
