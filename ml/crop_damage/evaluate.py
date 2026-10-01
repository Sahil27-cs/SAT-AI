"""Evaluation pipeline for SAT-AI crop damage segmentation model.

Evaluates CropDamageUNet on the held-out test split:
- Computes per-class IoU (no damage, partial damage, full damage)
- Computes macro F1, weighted F1, precision, recall
- Generates 3x3 confusion matrix and confusion_matrix.png
- Generates metrics.json, model_card.json, and sample_predictions/
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from ml.crop_damage.model import CropDamageUNet
from ml.crop_damage.train import BFCD22Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satai.crop_damage.evaluate")

PROCESSED_DIR = Path("data/processed/crop_damage")
OUTPUT_DIR = Path("models/crop_damage")


def evaluate():
    checkpoint_path = OUTPUT_DIR / "crop_damage_unet_bfcd22.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    manifest_path = PROCESSED_DIR / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_ids = manifest["splits"]["test"]

    test_ds = BFCD22Dataset(test_ids, PROCESSED_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CropDamageUNet(in_channels=7, num_classes=3, base_features=32).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []

    samples_dir = OUTPUT_DIR / "sample_predictions"
    samples_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx in range(len(test_ds)):
            features, label = test_ds[idx]
            features_t = features.unsqueeze(0).to(device)
            logits = model(features_t)
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            target = label.numpy()

            all_preds.append(preds.flatten())
            all_targets.append(target.flatten())

            if idx < 3:
                # Save sample prediction
                sample_file = samples_dir / f"sample_{test_ids[idx]}.npz"
                np.savez_compressed(
                    sample_file,
                    prediction=preds,
                    ground_truth=target,
                    ndvi_post=features[5].numpy(),
                )

    flat_preds = np.concatenate(all_preds)
    flat_targets = np.concatenate(all_targets)

    # Confusion matrix
    cm = confusion_matrix(flat_targets, flat_preds, labels=[0, 1, 2])
    logger.info("Confusion Matrix:\n%s", cm)

    # Per-class IoU
    per_class_iou = {}
    class_names = ["no_damage", "partial_damage", "full_damage"]
    for c, name in enumerate(class_names):
        intersection = np.logical_and(flat_targets == c, flat_preds == c).sum()
        union = np.logical_or(flat_targets == c, flat_preds == c).sum()
        iou = float(intersection / union) if union > 0 else 0.0
        per_class_iou[name] = round(iou, 4)

    mean_iou = round(float(np.mean(list(per_class_iou.values()))), 4)

    macro_f1 = round(float(f1_score(flat_targets, flat_preds, average="macro")), 4)
    weighted_f1 = round(float(f1_score(flat_targets, flat_preds, average="weighted")), 4)

    report = classification_report(
        flat_targets,
        flat_preds,
        labels=[0, 1, 2],
        target_names=class_names,
        output_dict=True,
    )

    metrics = {
        "dataset": "BFCD-22 (Muzaffarpur, Bihar benchmark split)",
        "task": "crop_damage_segmentation",
        "num_classes": 3,
        "classes": {
            "0": "no_damage",
            "1": "partial_damage",
            "2": "full_damage",
        },
        "mean_iou": mean_iou,
        "per_class_iou": per_class_iou,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "evaluation_split": "spatially_separated_test",
        "num_test_chips": len(test_ids),
        "total_test_pixels": int(len(flat_targets)),
    }

    metrics_file = OUTPUT_DIR / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Metrics written to %s", metrics_file)

    # Model card
    model_card = {
        "model_name": "CropDamageUNet (SAT-AI Crop Damage Assessment)",
        "version": "1.0.0",
        "architecture": "U-Net with Dual-Temporal Difference Attention (PyTorch)",
        "input_features": [
            "pre_flood_red (Sentinel-2 B4)",
            "pre_flood_nir (Sentinel-2 B8)",
            "post_flood_red (Sentinel-2 B4)",
            "post_flood_nir (Sentinel-2 B8)",
            "ndvi_pre",
            "ndvi_post",
            "delta_ndvi",
        ],
        "output_classes": {
            "0": "No damage",
            "1": "Partial damage",
            "2": "Full damage",
        },
        "intended_use": "Research assessment of flood-induced agriculture damage across localized floodplains in Bihar (e.g. Muzaffarpur).",
        "limitations": [
            "Trained and evaluated on Muzaffarpur flood conditions; requires domain verification for South Bihar or other soil/crop regimes.",
            "Optical imagery is weather/cloud dependent during active monsoons.",
            "Not an official government agricultural loss compensation survey.",
        ],
        "performance": {
            "mean_iou": mean_iou,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "full_damage_iou": per_class_iou["full_damage"],
            "partial_damage_iou": per_class_iou["partial_damage"],
            "no_damage_iou": per_class_iou["no_damage"],
        },
    }
    model_card_file = OUTPUT_DIR / "model_card.json"
    model_card_file.write_text(json.dumps(model_card, indent=2), encoding="utf-8")

    # Generate confusion matrix PNG
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (450, 450), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "Confusion Matrix (Test Split)", fill=(0, 0, 0))
        draw.text((20, 45), f"Macro F1: {macro_f1}  |  Mean IoU: {mean_iou}", fill=(50, 50, 50))

        start_x, start_y = 60, 100
        cell_size = 90
        for r in range(3):
            draw.text((start_x - 45, start_y + r * cell_size + 35), f"T{r}", fill=(0, 0, 0))
            for c in range(3):
                if r == 0:
                    draw.text((start_x + c * cell_size + 35, start_y - 25), f"P{c}", fill=(0, 0, 0))
                val = cm[r, c]
                color = (220, 235, 250) if r == c else (245, 245, 245)
                draw.rectangle(
                    [
                        (start_x + c * cell_size, start_y + r * cell_size),
                        (start_x + (c + 1) * cell_size, start_y + (r + 1) * cell_size),
                    ],
                    fill=color,
                    outline=(180, 180, 180),
                )
                draw.text(
                    (start_x + c * cell_size + 20, start_y + r * cell_size + 35),
                    str(val),
                    fill=(0, 0, 0),
                )
        img.save(OUTPUT_DIR / "confusion_matrix.png")
    except Exception as e:
        logger.warning("Could not generate confusion matrix PNG: %s", e)

    logger.info("Evaluation completed successfully.")
    return metrics


if __name__ == "__main__":
    evaluate()
