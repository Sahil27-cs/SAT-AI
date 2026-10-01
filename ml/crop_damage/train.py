"""Training pipeline for SAT-AI crop damage segmentation model.

Trains CropDamageUNet on dual-temporal multi-spectral features:
Pre/Post Red, NIR, NDVI_pre, NDVI_post, Delta_NDVI.
Target: 0=no damage, 1=partial damage, 2=full damage.

Saves:
- models/crop_damage/crop_damage_unet_bfcd22.pt
- models/crop_damage/training_curves.png
- models/crop_damage/training_summary.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ml.crop_damage.model import CropDamageUNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satai.crop_damage.train")

PROCESSED_DIR = Path("data/processed/crop_damage")
OUTPUT_DIR = Path("models/crop_damage")


class BFCD22Dataset(Dataset):
    def __init__(self, chip_ids: list[str], data_dir: Path):
        self.chip_ids = chip_ids
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.chip_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chip_path = self.data_dir / f"{self.chip_ids[idx]}.npz"
        data = np.load(chip_path)
        features = torch.from_numpy(data["features"].astype(np.float32))
        label = torch.from_numpy(data["label"].astype(np.int64))
        return features, label


def train_model(epochs: int = 5, batch_size: int = 4, lr: float = 1e-3):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PROCESSED_DIR / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run prepare_dataset.py first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_ids = manifest["splits"]["train"]
    val_ids = manifest["splits"]["val"]

    train_ds = BFCD22Dataset(train_ids, PROCESSED_DIR)
    val_ds = BFCD22Dataset(val_ids, PROCESSED_DIR)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    model = CropDamageUNet(in_channels=7, num_classes=3, base_features=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": []}

    best_val_loss = float("inf")
    best_weights_path = OUTPUT_DIR / "crop_damage_unet_bfcd22.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
        val_loss /= len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info("Epoch %d/%d - Train Loss: %.4f - Val Loss: %.4f", epoch, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "in_channels": 7,
                    "num_classes": 3,
                },
                best_weights_path,
            )

    logger.info("Best model saved to %s (Val Loss: %.4f)", best_weights_path, best_val_loss)

    # Save training curves as artifact image / data
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (600, 300), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), f"SAT-AI BFCD-22 Training Curves\nEpochs: {epochs}\nBest Val Loss: {best_val_loss:.4f}", fill=(0, 0, 0))
        # Draw loss line
        for e in range(len(history["train_loss"]) - 1):
            x1 = 50 + int(e * (500 / max(1, epochs - 1)))
            y1 = 250 - int(min(200, history["train_loss"][e] * 100))
            x2 = 50 + int((e + 1) * (500 / max(1, epochs - 1)))
            y2 = 250 - int(min(200, history["train_loss"][e + 1] * 100))
            draw.line([(x1, y1), (x2, y2)], fill=(200, 50, 50), width=2)
        img.save(OUTPUT_DIR / "training_curves.png")
    except Exception as e:
        logger.warning("Could not generate training curve image: %s", e)

    summary_file = OUTPUT_DIR / "training_summary.json"
    summary_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def main():
    train_model(epochs=5, batch_size=4, lr=1e-3)


if __name__ == "__main__":
    main()
