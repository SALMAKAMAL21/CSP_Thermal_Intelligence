"""
Autoencoder-based anomaly support for tube_ref / tube_test.

This module treats the autoencoder as a complementary signal. It compares the
reconstruction error of tube_ref and tube_test crops and converts the relative
gap into a lightweight warning/anomaly support score.
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
class AutoencoderConfig:
    input_size: int = 128
    latent_dim: int = 256
    warning_ratio: float = 1.10
    anomaly_ratio: float = 1.25
    score_eps: float = 1e-6


DEFAULT_CONFIG = AutoencoderConfig()


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TubeAutoencoder(nn.Module):
    """Architecture reconstructed from the shipped checkpoint."""

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                _ConvBlock(3, 32),
                _ConvBlock(32, 32),
                _ConvBlock(32, 64),
                _ConvBlock(64, 64),
                _ConvBlock(64, 128),
                _ConvBlock(128, 128),
                _ConvBlock(128, 256),
                _ConvBlock(256, 256),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.fc_enc = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 16 * 16)
        self.decoder = nn.ModuleList(
            [
                _ConvBlock(256, 256),
                _UpBlock(256, 128),
                _ConvBlock(128, 128),
                _UpBlock(128, 64),
                _ConvBlock(64, 64),
                _UpBlock(64, 32),
                _ConvBlock(32, 32),
                nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, block in enumerate(self.encoder):
            x = block(x)
            if idx in {1, 3, 5}:
                x = self.pool(x)

        x = x.flatten(1)
        x = self.fc_enc(x)
        x = self.fc_dec(x)
        x = x.view(-1, 256, 16, 16)

        for idx, block in enumerate(self.decoder):
            x = block(x)
            if idx == len(self.decoder) - 1:
                x = self.output_activation(x)
        return x


def load_autoencoder_model(model_path: Path, device: str, config: AutoencoderConfig = DEFAULT_CONFIG) -> nn.Module:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model = TubeAutoencoder(latent_dim=config.latent_dim)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def analyze_ref_test_autoencoder(
    image_np: np.ndarray,
    detections: list[dict[str, Any]],
    model: nn.Module | None,
    device: str,
    config: AutoencoderConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    if model is None:
        return {
            "status": "disabled",
            "enabled": False,
            "reason": "autoencoder model not loaded",
        }

    height, width = image_np.shape[:2]
    ref_det = _pick_detection(detections, "tube_ref")
    test_det = _pick_detection(detections, "tube_test")
    if ref_det is None or test_det is None:
        return {
            "status": "insufficient_tubes",
            "enabled": True,
            "source": "tube_ref_vs_tube_test_reconstruction",
            "reason": "tube_ref and tube_test are required",
        }

    ref_mask = _detection_to_mask(ref_det, height, width)
    test_mask = _detection_to_mask(test_det, height, width)
    if ref_mask is None or test_mask is None:
        return {
            "status": "missing_masks",
            "enabled": True,
            "source": "tube_ref_vs_tube_test_reconstruction",
            "reason": "mask_polygon or bbox is required for both tubes",
        }

    ref_input = _prepare_crop(image_np, ref_mask, ref_det, config.input_size)
    test_input = _prepare_crop(image_np, test_mask, test_det, config.input_size)
    if ref_input is None or test_input is None:
        return {
            "status": "insufficient_pixels",
            "enabled": True,
            "source": "tube_ref_vs_tube_test_reconstruction",
            "reason": "not enough pixels for autoencoder crop",
        }

    ref_score = _reconstruction_error(model, ref_input, device)
    test_score = _reconstruction_error(model, test_input, device)
    delta = test_score - ref_score
    ratio = test_score / max(ref_score, config.score_eps)
    decision = _decision(delta, ratio, config)
    support_score = _support_score(delta, ratio, config)

    return {
        "status": "ok",
        "enabled": True,
        "source": "tube_ref_vs_tube_test_reconstruction",
        "tube_ref_score": round(float(ref_score), 4),
        "tube_test_score": round(float(test_score), 4),
        "score_delta": round(float(delta), 4),
        "score_ratio": round(float(ratio), 4),
        "support_score": round(float(support_score), 4),
        "decision": decision,
        "confidence": _confidence(decision, ratio),
        "input_size": config.input_size,
    }


def build_combined_anomaly(
    thermal_anomaly: dict[str, Any],
    autoencoder_anomaly: dict[str, Any],
) -> dict[str, Any]:
    thermal_score = float(thermal_anomaly.get("anomaly_score", 0.0) or 0.0)
    ae_support = float(autoencoder_anomaly.get("support_score", 0.0) or 0.0)
    ae_decision = str(autoencoder_anomaly.get("decision", "unknown"))
    thermal_status = str(thermal_anomaly.get("status", "unknown"))
    ae_status = str(autoencoder_anomaly.get("status", "unknown"))

    raw_score = min(1.0, thermal_score + 0.18 * ae_support)
    if thermal_score >= 0.38:
        decision = "anomaly"
    elif thermal_score >= 0.24 and ae_decision in {"warning", "anomaly"}:
        decision = "anomaly"
    elif thermal_score >= 0.24 or ae_decision in {"warning", "anomaly"}:
        decision = "warning"
    elif thermal_status == "ok" or ae_status == "ok":
        decision = "normal"
    else:
        decision = "unknown"

    if decision == "anomaly":
        score = max(raw_score, 0.38)
    elif decision == "warning":
        score = max(raw_score, 0.24)
    else:
        score = raw_score

    return {
        "status": "ok" if thermal_status == "ok" or ae_status == "ok" else "unknown",
        "enabled": thermal_anomaly.get("enabled", False) or autoencoder_anomaly.get("enabled", False),
        "source": "thermal_plus_autoencoder",
        "anomaly_score": round(float(score), 4),
        "decision": decision,
        "confidence": _combined_confidence(decision, thermal_score, ae_support),
        "thermal_score": round(float(thermal_score), 4),
        "autoencoder_support": round(float(ae_support), 4),
        "thermal_status": thermal_status,
        "autoencoder_status": ae_status,
        "note": "Autoencoder is used as a complementary signal and does not replace thermal comparison.",
    }


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


def _prepare_crop(
    image_np: np.ndarray,
    mask: np.ndarray,
    detection: dict[str, Any],
    input_size: int,
) -> np.ndarray | None:
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


def _reconstruction_error(model: nn.Module, image_np: np.ndarray, device: str) -> float:
    tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        recon = model(tensor)
        error = torch.mean((recon - tensor) ** 2)
    return float(error.item())


def _decision(delta: float, ratio: float, config: AutoencoderConfig) -> str:
    if delta <= 0:
        return "normal"
    if ratio >= config.anomaly_ratio:
        return "anomaly"
    if ratio >= config.warning_ratio:
        return "warning"
    return "normal"


def _support_score(delta: float, ratio: float, config: AutoencoderConfig) -> float:
    if delta <= 0:
        return 0.0
    normalized = (ratio - 1.0) / max(config.anomaly_ratio - 1.0, 1e-6)
    return float(np.clip(normalized, 0.0, 1.0))


def _confidence(decision: str, ratio: float) -> str:
    if decision == "anomaly":
        return "high" if ratio >= 1.35 else "medium"
    if decision == "warning":
        return "medium" if ratio >= 1.18 else "low"
    return "high"


def _combined_confidence(decision: str, thermal_score: float, ae_support: float) -> str:
    if decision == "anomaly":
        return "high" if thermal_score >= 0.38 and ae_support >= 0.5 else "medium"
    if decision == "warning":
        return "medium" if thermal_score >= 0.24 or ae_support >= 0.5 else "low"
    return "high"
