"""
Siamese-based anomaly support for tube_ref / tube_test.

This module learns a pairwise comparison directly from (tube_ref, tube_test)
pairs. It is designed as a complementary signal on top of the current thermal
comparison and autoencoder pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class SiameseConfig:
    input_size: int = 128
    embedding_dim: int = 128
    warning_threshold: float = 0.45
    anomaly_threshold: float = 0.6


DEFAULT_CONFIG = SiameseConfig()


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TubePairSiamese(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            _EncoderBlock(3, 32),
            _EncoderBlock(32, 64),
            _EncoderBlock(64, 128),
            _EncoderBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(embedding_dim, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(x))

    def forward(self, ref: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ref_emb = self.encode(ref)
        test_emb = self.encode(test)
        abs_diff = torch.abs(test_emb - ref_emb)
        mul = test_emb * ref_emb
        logits = self.classifier(torch.cat([abs_diff, mul], dim=1)).squeeze(1)
        return logits, ref_emb, test_emb


def load_siamese_model(model_path: Path, device: str) -> tuple[nn.Module, SiameseConfig]:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported siamese checkpoint format")

    config_data = checkpoint.get("config") or {}
    config = SiameseConfig(
        input_size=int(config_data.get("input_size", DEFAULT_CONFIG.input_size)),
        embedding_dim=int(config_data.get("embedding_dim", DEFAULT_CONFIG.embedding_dim)),
        warning_threshold=float(config_data.get("warning_threshold", DEFAULT_CONFIG.warning_threshold)),
        anomaly_threshold=float(config_data.get("anomaly_threshold", DEFAULT_CONFIG.anomaly_threshold)),
    )

    state = checkpoint.get("model_state")
    if state is None:
        raise ValueError("Missing model_state in siamese checkpoint")

    model = TubePairSiamese(embedding_dim=config.embedding_dim)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, config


def analyze_ref_test_siamese(
    image_np: np.ndarray,
    detections: list[dict[str, Any]],
    model: nn.Module | None,
    device: str,
    config: SiameseConfig | None = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = config or DEFAULT_CONFIG

    if model is None:
        return {
            "status": "disabled",
            "enabled": False,
            "reason": "siamese model not loaded",
        }

    height, width = image_np.shape[:2]
    ref_det = _pick_detection(detections, "tube_ref")
    test_det = _pick_detection(detections, "tube_test")
    if ref_det is None or test_det is None:
        return {
            "status": "insufficient_tubes",
            "enabled": True,
            "source": "tube_ref_vs_tube_test_siamese",
            "reason": "tube_ref and tube_test are required",
        }

    ref_mask = _detection_to_mask(ref_det, height, width)
    test_mask = _detection_to_mask(test_det, height, width)
    if ref_mask is None or test_mask is None:
        return {
            "status": "missing_masks",
            "enabled": True,
            "source": "tube_ref_vs_tube_test_siamese",
            "reason": "mask_polygon or bbox is required for both tubes",
        }

    ref_input = _prepare_crop(image_np, ref_mask, config.input_size)
    test_input = _prepare_crop(image_np, test_mask, config.input_size)
    if ref_input is None or test_input is None:
        return {
            "status": "insufficient_pixels",
            "enabled": True,
            "source": "tube_ref_vs_tube_test_siamese",
            "reason": "not enough pixels for siamese crop",
        }

    probability, distance = _pair_probability(model, ref_input, test_input, device)
    decision = _decision(probability, config)
    return {
        "status": "ok",
        "enabled": True,
        "source": "tube_ref_vs_tube_test_siamese",
        "pair_probability": round(float(probability), 4),
        "embedding_distance": round(float(distance), 4),
        "support_score": round(float(probability), 4),
        "decision": decision,
        "confidence": _confidence(decision, probability),
        "input_size": config.input_size,
    }


def build_combined_with_siamese(
    combined_anomaly: dict[str, Any],
    siamese_anomaly: dict[str, Any],
) -> dict[str, Any]:
    siamese_support = float(siamese_anomaly.get("support_score", 0.0) or 0.0)
    siamese_decision = str(siamese_anomaly.get("decision", "unknown"))
    score = float(combined_anomaly.get("anomaly_score", 0.0) or 0.0)
    boosted = min(1.0, score + 0.12 * siamese_support)

    decision = str(combined_anomaly.get("decision", "unknown"))
    if decision == "warning" and siamese_decision == "anomaly":
        decision = "anomaly"
    elif decision == "normal" and siamese_decision in {"warning", "anomaly"}:
        decision = "warning"

    if decision == "anomaly":
        boosted = max(boosted, 0.38)
    elif decision == "warning":
        boosted = max(boosted, 0.24)

    merged = dict(combined_anomaly)
    merged.update(
        {
            "anomaly_score": round(float(boosted), 4),
            "decision": decision,
            "siamese_support": round(float(siamese_support), 4),
            "siamese_status": str(siamese_anomaly.get("status", "unknown")),
            "note": "Thermal, autoencoder, and siamese signals are fused conservatively.",
        }
    )
    return merged


def _pick_detection(detections: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    candidates = [d for d in detections if str(d.get("class_name", "")).lower() == label]
    if not candidates:
        return None
    return max(candidates, key=_detection_score)


def _detection_score(det: dict[str, Any]) -> float:
    confidence = float(det.get("confidence", 0.0) or 0.0)
    area = float(det.get("mask_area", 0.0) or 0.0)
    if area <= 0:
        bbox = det.get("bbox") or [0, 0, 0, 0]
        area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return confidence * max(area, 1.0)


def _detection_to_mask(det: dict[str, Any], height: int, width: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = det.get("mask_polygon")
    if isinstance(polygon, list) and len(polygon) >= 3:
        pts = np.array(polygon, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[1] == 2:
            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            if int(mask.sum()) > 0:
                return mask.astype(bool)
    bbox = det.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
        y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
            return mask.astype(bool)
    return None


def _prepare_crop(image_np: np.ndarray, mask: np.ndarray, input_size: int) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if xs.size < 20 or ys.size < 20:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad_x = max(4, int(0.05 * (x2 - x1 + 1)))
    pad_y = max(4, int(0.05 * (y2 - y1 + 1)))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image_np.shape[1] - 1, x2 + pad_x)
    y2 = min(image_np.shape[0] - 1, y2 + pad_y)

    crop = image_np[y1 : y2 + 1, x1 : x2 + 1, :3].copy()
    crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0 or int(crop_mask.sum()) == 0:
        return None

    crop[~crop_mask] = 0
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    y_offset = (side - h) // 2
    x_offset = (side - w) // 2
    canvas[y_offset : y_offset + h, x_offset : x_offset + w] = crop
    resized = cv2.resize(canvas, (input_size, input_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _pair_probability(model: nn.Module, ref_input: np.ndarray, test_input: np.ndarray, device: str) -> tuple[float, float]:
    ref_tensor = torch.from_numpy(ref_input.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)
    test_tensor = torch.from_numpy(test_input.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        logits, ref_emb, test_emb = model(ref_tensor, test_tensor)
        probability = torch.sigmoid(logits)[0]
        distance = torch.norm(test_emb - ref_emb, p=2, dim=1)[0]
    return float(probability.item()), float(distance.item())


def _decision(probability: float, config: SiameseConfig) -> str:
    if probability >= config.anomaly_threshold:
        return "anomaly"
    if probability >= config.warning_threshold:
        return "warning"
    return "normal"


def _confidence(decision: str, probability: float) -> str:
    if decision == "anomaly":
        return "high" if probability >= 0.8 else "medium"
    if decision == "warning":
        return "medium" if probability >= 0.55 else "low"
    return "high"
