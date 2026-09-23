"""Inference ordinale : memes ROI et normalisation que l'entrainement."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .fusion_model import TubeFusionModel

LEVEL_NAMES = ["Normal", "Quantité minimale d'air", "10 L d'air", "18 L d'air", "Absence de vide"]
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class FusionPredictor:
    def __init__(self, checkpoint: Path, config_path: Path, device: str):
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.mean = float(self.config["delta_t_mean"])
        self.std = float(self.config["delta_t_std"])
        if not math.isfinite(self.mean) or not math.isfinite(self.std) or self.std <= 0:
            raise ValueError("Statistiques de normalisation invalides")
        expected_hash = self.config.get("checkpoint_sha256")
        if expected_hash and hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Le checkpoint ne correspond pas à configs/fusion.json")
        self.device = device
        self.model = TubeFusionModel(pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model.to(device).eval()

    def predict(self, image: np.ndarray, detections: list[dict], t_ref: float, t_test: float):
        delta_t = temperature_delta(t_ref, t_test)
        inputs = {}
        for role in ("tube_ref", "tube_test"):
            candidates = [d for d in detections if d.get("class_name") == role and float(d.get("confidence", 0)) >= 0.5]
            if not candidates:
                return {"status": "insufficient_tubes", "reason": "Deux tubes détectés avec une confiance >= 0,5 sont nécessaires."}
            detection = max(candidates, key=lambda d: float(d["confidence"]))
            crop = crop_box(image, detection.get("bbox"))
            if crop is None:
                return {"status": "invalid_crop", "reason": "Zone de tube vide ou invalide."}
            inputs[role] = TRANSFORM(Image.fromarray(crop).convert("RGB")).unsqueeze(0).to(self.device)
        delta = torch.tensor([[(delta_t - self.mean) / self.std]], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            logits = self.model(inputs["tube_test"], inputs["tube_ref"], delta)
            cumulative = torch.sigmoid(logits)[0].cpu().tolist()
        level = sum(p > 0.5 for p in cumulative)
        return {"status": "ok", "level": level, "label": LEVEL_NAMES[level],
                "cumulative_probabilities": cumulative, "delta_t": delta_t,
                "t_ref": t_ref, "t_test": t_test}


def temperature_delta(t_ref, t_test):
    if t_ref is None or t_test is None or not math.isfinite(t_ref) or not math.isfinite(t_test):
        raise ValueError("Deux températures finies en °C sont nécessaires.")
    return t_ref - t_test


def crop_box(image, bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        if not all(math.isfinite(float(v)) for v in bbox):
            return None
        x1, y1, x2, y2 = (int(v) for v in bbox)
    except (ValueError, TypeError, OverflowError):
        return None
    height, width = image.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2, :3].copy()
