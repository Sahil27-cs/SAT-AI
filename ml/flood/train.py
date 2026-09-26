"""Train the flood segmentation model. Runs in a GPU environment, not on Vercel.

    python ml/flood/train.py --preset sar_ratio --fold loro_india --epochs 40

**This has never been executed.** The Sen1Floods11 archive is not retrievable
from the environment this repository was built in, and training needs a GPU.
The script is written so that the run is reproducible when the data is in hand,
not so that a number can be claimed without it — every metric this project
reports comes from a run that actually happened.

Where this fits (ADR-001): training is the ML plane. It reads chips, writes a
checkpoint and a metrics JSON, and exports ONNX for CPU serving. The serving
plane never imports this module and never sees torch.

Evaluation protocol is leave-one-region-out (ADR-009), not Sen1Floods11's
shipped splits. The shipped splits are chip-level random within region, so chips
from one flood event straddle train and test and the resulting IoU measures
interpolation within an event. Both are computed; the gap between them is a
reportable result.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Bootstrap. `satai.paths` is the canonical resolver, but it cannot be imported
# until the package is importable — so when it is not (a fresh notebook that has
# not run `pip install -e .`), a minimal inline search finds the checkout and
# puts it on the path. `__file__` is wrapped because Jupyter, Colab and Kaggle
# do not define it, and a GPU notebook is exactly where this script runs.
try:
    from satai.paths import REPO_ROOT, in_notebook
except ImportError:  # pragma: no cover - exercised only outside an installed package
    _candidates: list[Path] = []
    if os.environ.get("SATAI_REPO_ROOT"):
        _candidates.append(Path(os.environ["SATAI_REPO_ROOT"]).expanduser().resolve())
    # Suppressed rather than guarded: `__file__` is simply absent in a notebook,
    # which is the expected case here, not an error worth branching on.
    with contextlib.suppress(NameError):
        _candidates.append(Path(__file__).resolve().parents[2])
    _cwd = Path.cwd().resolve()
    _candidates += [_cwd, *_cwd.parents, *(p for p in sorted(_cwd.iterdir()) if p.is_dir())]
    REPO_ROOT = next(
        (c for c in _candidates if (c / "satai" / "provenance.py").is_file()),
        _cwd,
    )
    sys.path.insert(0, str(REPO_ROOT))
    from satai.paths import in_notebook

from satai.errors import ValidationError
from satai.logging import get_logger
from satai.ml.unet import BAND_PRESETS, FloodUNetConfig, build_model, torch_available

log = get_logger(__name__)

DEFAULT_OUTPUT = REPO_ROOT / "models" / "flood"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="sar_ratio", choices=sorted(BAND_PRESETS))
    parser.add_argument("--chips", type=Path, required=False, help="Sen1Floods11 chip root")
    parser.add_argument("--fold", default="loro_india", help="LORO fold name, or 'official'")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chip-px", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the configuration and the split, then stop without training.",
    )

    if argv is None and in_notebook():
        # A kernel's sys.argv carries its own flags (-f /.../kernel-xxx.json),
        # which argparse rejects with an unrelated-looking error. Say what to do
        # instead of letting the parser complain about a file nobody passed.
        raise ValidationError(
            "Running under a notebook kernel: pass arguments explicitly rather "
            "than relying on sys.argv. For example:\n"
            "    main(['--chips', '/kaggle/input/sen1floods11/chips', "
            "'--preset', 'sar_ratio', '--fold', 'loro_india'])"
        )
    return parser.parse_args(argv)


def preflight(args: argparse.Namespace) -> FloodUNetConfig:
    """Everything that can fail cheaply, before anything expensive starts.

    A training run that dies forty minutes in because a chip directory was
    misspelled has wasted the scarcest resource in this project, which is GPU
    time on a borrowed machine.
    """
    config = FloodUNetConfig.from_preset(args.preset, chip_px=args.chip_px, seed=args.seed)

    if args.chips is None:
        raise ValidationError(
            "--chips is required: point it at the Sen1Floods11 chip root. "
            "The archive is distributed publicly; this repository does not "
            "vendor it and does not synthesise a stand-in."
        )
    if not args.chips.is_dir():
        raise ValidationError(f"chip root does not exist: {args.chips}")

    if not torch_available():
        raise ValidationError(
            "PyTorch is not installed here. Train in a GPU environment (Colab, "
            "Kaggle, or a local CUDA box) and export ONNX for serving (ADR-001)."
        )
    return config


def run(args: argparse.Namespace) -> int:
    config = preflight(args)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "model": config.describe(),
        "fold": args.fold,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "chip_root": str(args.chips),
        "split_protocol": (
            "official_sen1floods11" if args.fold == "official" else "leave_one_region_out"
        ),
    }

    if args.dry_run:
        manifest["status"] = "dry_run"
        print(json.dumps(manifest, indent=2))
        return 0

    import torch
    from torch.utils.data import DataLoader

    from satai.preprocessing.chips import Sen1Floods11Dataset  # type: ignore[attr-defined]
    from satai.preprocessing.splits import leave_one_region_out, parse_chip_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log.warning("no CUDA device; this will be extremely slow")

    chips = [parse_chip_name(p.name) for p in sorted(args.chips.glob("*_S1Hand.tif"))]
    if not chips:
        raise ValidationError(f"no *_S1Hand.tif chips found under {args.chips}")

    fold = next(
        (f for f in leave_one_region_out(chips) if f.name == args.fold),
        None,
    )
    if fold is None:
        raise ValidationError(f"no fold named {args.fold!r} in this chip set")
    # Cheap, and it catches the class of mistake that silently inflates every
    # downstream number.
    fold.assert_disjoint()
    manifest["fold_sizes"] = fold.sizes
    manifest["region_disjoint"] = fold.is_region_disjoint

    model = build_model(config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Dice handles the class imbalance that makes accuracy meaningless here:
    # flood pixels are a small minority, and a model predicting "dry" everywhere
    # scores well on accuracy and is worthless.
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        Sen1Floods11Dataset(fold.train, args.chips, config.bands),
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        Sen1Floods11Dataset(fold.val, args.chips, config.bands),
        batch_size=args.batch,
        num_workers=2,
    )

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimiser.step()
            train_loss += float(loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                val_loss += float(loss_fn(model(features), labels).item())

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        log.info(
            "epoch complete",
            extra={"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss},
        )

        # Selected on a HELD-OUT REGION, not on random chips from the training
        # regions. Early stopping tuned on chips statistically identical to the
        # training set is how a leaked number gets into a results table.
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"state_dict": model.state_dict(), "config": asdict(config), "epoch": epoch},
                args.out / f"flood_unet_{args.preset}_{args.fold}.pt",
            )

    manifest["best_val_loss"] = best_val
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["status"] = "complete"
    (args.out / f"train_manifest_{args.preset}_{args.fold}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
