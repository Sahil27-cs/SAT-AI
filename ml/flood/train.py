"""Train the flood segmentation U-Net.

    python -m ml.flood.train --data-root data/raw/sen1floods11 --fold loro_india

Protocol is leave-one-region-out (ADR-009), not Sen1Floods11's shipped splits.
The shipped splits are chip-level random *within* region, so chips from one
flood event straddle train and test and the resulting IoU measures
interpolation within an event rather than generalisation to a new one.

Three guards, each against a specific way a segmentation number gets inflated:

*The validation region is held out too.* Early stopping tuned on chips drawn
from the training regions is tuned on data statistically identical to what the
model saw, so the "best" checkpoint is selected on a signal that does not
generalise. Here validation is a whole separate region.

*Normalisation is fitted on the training partition only*, and the statistics
travel with the checkpoint. Recomputing them at inference would erase the very
domain shift the evaluation exists to measure.

*Unannotated pixels are masked everywhere* -- loss, metrics, and the reported
positive rate. A third of every chip carries label -1.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
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

    # The root is on the path now, so the real helper is importable. Importing it
    # here rather than reimplementing it keeps one definition of what a recorded
    # path looks like.
    from satai.paths import relative_to_repo

import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from ml.flood.dataset import FloodBatch, FloodChips, bands_for, build_reader, fit_normalizer
from ml.flood.losses import FloodLoss
from ml.flood.model import UNet, UNetSpec
from satai.ml.metrics import SegmentationMetrics, aggregate, evaluate
from satai.preprocessing.splits import Fold, leave_one_region_out

DEFAULT_DATA = REPO_ROOT / "data" / "raw" / "sen1floods11"
DEFAULT_OUT = REPO_ROOT / "models" / "flood"


@dataclass
class History:
    """Per-epoch record, written beside the checkpoint."""

    epoch: list[int]
    train_loss: list[float]
    val_loss: list[float]
    val_iou: list[float]
    val_f1: list[float]
    lr: list[float]
    seconds: list[float]

    @classmethod
    def empty(cls) -> History:
        return cls([], [], [], [], [], [], [])

    def append(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            getattr(self, key).append(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fold", default="loro_india", help="LORO fold, e.g. loro_india")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-ratio", action="store_true", help="SAR bands only, no VV/VH ratio")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Mixed precision. Needs CUDA.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping, 0 disables")
    parser.add_argument("--limit-chips", type=int, default=0, help="Smoke test on N chips")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def select_fold(data_root: Path, fold_name: str) -> Fold:
    """Build the requested leave-one-region-out fold from what is on disk."""
    reader = build_reader(data_root)
    chips = reader.available(reader.discover())
    if not chips:
        raise SystemExit(
            f"no chips under {data_root}. Run: python scripts/download_sen1floods11.py"
        )
    for fold in leave_one_region_out(chips):
        if fold.name == fold_name:
            return fold
    available = ", ".join(f.name for f in leave_one_region_out(chips))
    raise SystemExit(f"no fold named {fold_name!r}. Available: {available}")


@torch.no_grad()
def run_validation(
    model: UNet, loader: DataLoader[FloodBatch], loss_fn: FloodLoss, device: torch.device
) -> tuple[float, SegmentationMetrics]:
    """Loss and pooled segmentation metrics over a whole partition."""
    model.eval()
    total, batches = 0.0, 0
    per_chip: list[SegmentationMetrics] = []

    for features, target, valid in loader:
        features, target, valid = features.to(device), target.to(device), valid.to(device)
        logits = model(features)
        loss, _ = loss_fn(logits, target, valid)
        total += float(loss)
        batches += 1

        probability = torch.sigmoid(logits).cpu().numpy()
        truth = target.cpu().numpy()
        mask = valid.cpu().numpy().astype(bool)
        for i in range(probability.shape[0]):
            if mask[i, 0].any():
                per_chip.append(
                    evaluate(probability[i, 0], truth[i, 0], valid=mask[i, 0], threshold=0.5)
                )

    # Confusion counts are pooled, never averaged: averaging per-chip IoU lets a
    # chip with three water pixels weigh as much as one that is half flooded.
    return total / max(1, batches), aggregate(per_chip)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fold = select_fold(args.data_root, args.fold)
    fold.assert_disjoint()

    train_chips = list(fold.train)
    val_chips = list(fold.val)
    test_chips = list(fold.test)
    if args.limit_chips:
        train_chips = train_chips[: args.limit_chips]
        val_chips = val_chips[: max(2, args.limit_chips // 4)]

    with_ratio = not args.no_ratio
    bands = bands_for(with_ratio)
    device = torch.device(args.device)

    print(f"fold {fold.name}: {fold.summary()}")
    print(f"  region-disjoint: {fold.is_region_disjoint}")
    print(f"  bands: {', '.join(bands)}")
    print(f"  device: {device}")

    if args.dry_run:
        print("dry run: fold validated, nothing trained")
        return 0

    print("fitting normalisation on the training partition only...")
    normalizer = fit_normalizer(train_chips, args.data_root, fold.name, with_ratio=with_ratio)
    normalizer.assert_fold(fold.name)
    for band in normalizer.bands:
        stats = normalizer[band]
        print(
            f"  {band:<14} clip [{stats.lower:7.2f}, {stats.upper:7.2f}] "
            f"mean {stats.mean:7.2f} std {stats.std:6.2f}"
        )

    train_set = FloodChips(
        train_chips,
        args.data_root,
        normalizer,
        with_ratio=with_ratio,
        augment=not args.no_augment,
        seed=args.seed,
    )
    val_set = FloodChips(val_chips, args.data_root, normalizer, with_ratio=with_ratio)
    print(f"  train {len(train_set)} chips, val {len(val_set)} chips")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=len(train_set) > args.batch_size,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)

    spec = UNetSpec(in_channels=len(bands), base_width=args.base_width, depth=args.depth)
    model = UNet(spec).to(device)
    print(f"  model: {model.n_parameters():,} trainable parameters")

    loss_fn = FloodLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch, best_iou, history = 0, -1.0, History.empty()
    if args.resume and args.resume.is_file():
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        start_epoch = checkpoint["epoch"] + 1
        best_iou = checkpoint.get("best_iou", -1.0)
        print(f"  resumed from {args.resume} at epoch {start_epoch}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{fold.name}_{'sar_ratio' if with_ratio else 'sar'}"
    best_path = args.output_dir / f"flood_unet_{tag}_best.pt"
    epochs_without_gain = 0

    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()
        model.train()
        running, batches = 0.0, 0

        for features, target, valid in train_loader:
            features, target, valid = features.to(device), target.to(device), valid.to(device)
            optimiser.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=scaler.is_enabled()):
                loss, _ = loss_fn(model(features), target, valid)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running += float(loss)
            batches += 1

        train_loss = running / max(1, batches)
        val_loss, val_metrics = run_validation(model, val_loader, loss_fn, device)
        scheduler.step()
        elapsed = time.perf_counter() - started

        history.append(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_iou=val_metrics.iou,
            val_f1=val_metrics.f1,
            lr=scheduler.get_last_lr()[0],
            seconds=elapsed,
        )
        print(
            f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"IoU {val_metrics.iou:.4f}  F1 {val_metrics.f1:.4f}  {elapsed:5.1f}s"
        )

        # Selected on IoU over a HELD-OUT REGION, not on loss over chips drawn
        # from the training regions.
        if val_metrics.iou > best_iou:
            best_iou = val_metrics.iou
            epochs_without_gain = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "spec": asdict(spec),
                    "bands": list(bands),
                    "fold": fold.name,
                    "epoch": epoch,
                    "best_iou": best_iou,
                    "normalizer": normalizer.to_dict(),
                    "seed": args.seed,
                },
                best_path,
            )
        else:
            epochs_without_gain += 1
            if args.patience and epochs_without_gain >= args.patience:
                print(f"  early stop: {args.patience} epochs without a validation gain")
                break

    normalizer.save(args.output_dir / f"normalizer_{tag}.json")
    manifest = {
        "model": "flood_unet",
        "fold": fold.name,
        "protocol": "leave_one_region_out",
        "region_disjoint": fold.is_region_disjoint,
        "test_regions": list(fold.test_regions),
        "val_regions": list(fold.val_regions),
        "bands": list(bands),
        "spec": asdict(spec),
        "n_parameters": model.n_parameters(),
        "epochs_run": len(history.epoch),
        "best_val_iou": best_iou,
        "sizes": {"train": len(train_set), "val": len(val_set), "test": len(test_chips)},
        "hyperparameters": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "augment": not args.no_augment,
            "amp": args.amp,
            "bce_weight": loss_fn.bce_weight,
            "dice_weight": loss_fn.dice_weight,
        },
        "device": str(device),
        "finished_at": datetime.now(UTC).isoformat(),
        "history": asdict(history),
        # relative_to_repo, not Path.relative_to: the output directory is often
        # outside the checkout (a scratch dir, a mounted drive, Kaggle working),
        # where relative_to raises instead of falling back -- and the recorded
        # path has to be readable on Linux whatever platform trained the model.
        "checkpoint": (relative_to_repo(best_path) if best_path.exists() else None),
        "caveats": [
            "Validation IoU is on a held-out REGION, not a random chip split.",
            "This is a validation score, not a test score. Run ml/flood/evaluate.py "
            "on the fold's test region for the number that should be reported.",
        ],
    }
    (args.output_dir / f"train_manifest_{tag}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\nbest validation IoU {best_iou:.4f} -> {best_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
