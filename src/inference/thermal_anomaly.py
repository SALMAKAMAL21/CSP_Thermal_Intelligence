"""
Thermal anomaly scoring for tube_ref / tube_test.

This module does not train a model. It compares the relative thermal signal of
two segmented tubes in the same thermal RGB image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ThermalConfig:
    bins: int = 32
    target_width: int = 256
    target_height: int = 32
    min_pixels_per_bin: int = 8
    z_threshold: float = 3.0
    min_delta: float = 0.20
    strong_delta: float = 0.30
    min_contiguous_segments: int = 2
    ignore_edge_bins: int = 4
    robust_scale_floor: float = 0.03


DEFAULT_CONFIG = ThermalConfig()


def analyze_ref_test_thermal(
    image_np: np.ndarray,
    detections: list[dict[str, Any]],
    config: ThermalConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Compare tube_ref and tube_test and return a relative anomaly score."""
    height, width = image_np.shape[:2]
    ref_det = _pick_detection(detections, "tube_ref")
    test_det = _pick_detection(detections, "tube_test")

    if ref_det is None or test_det is None:
        return {
            "status": "insufficient_tubes",
            "enabled": True,
            "thermal_source": "rgb_relative",
            "reason": "tube_ref and tube_test are required",
        }

    ref_mask = _detection_to_mask(ref_det, height, width)
    test_mask = _detection_to_mask(test_det, height, width)
    if ref_mask is None or test_mask is None:
        return {
            "status": "missing_masks",
            "enabled": True,
            "thermal_source": "rgb_relative",
            "reason": "mask_polygon or bbox is required for both tubes",
        }

    intensity = _thermal_scalar(image_np)
    ref_mask = _clean_mask(ref_mask)
    test_mask = _clean_mask(test_mask)

    ref_norm = _normalize_tube(intensity, ref_mask, config)
    test_norm = _normalize_tube(intensity, test_mask, config)
    if ref_norm is None or test_norm is None:
        return {
            "status": "insufficient_pixels",
            "enabled": True,
            "thermal_source": "rgb_relative",
            "reason": "not enough mask pixels to build comparable tube profiles",
        }

    ref_patch, ref_patch_mask = ref_norm
    test_patch, test_patch_mask = test_norm

    ref_profile = _profile(ref_patch, ref_patch_mask, config)
    test_profile = _profile(test_patch, test_patch_mask, config)
    comparison = _compare_profiles(ref_profile, test_profile, config)

    return {
        "status": "ok",
        "enabled": True,
        "thermal_source": "rgb_relative",
        "note": "Dataset images are 8-bit RGB thermal renderings, not radiometric temperature maps.",
        "profile_bins": config.bins,
        "thresholds": {
            "z_threshold": config.z_threshold,
            "min_delta": config.min_delta,
            "strong_delta": config.strong_delta,
            "min_contiguous_segments": config.min_contiguous_segments,
            "ignore_edge_bins": config.ignore_edge_bins,
        },
        "tube_ref": _tube_summary(ref_patch, ref_patch_mask, ref_mask, ref_det),
        "tube_test": _tube_summary(test_patch, test_patch_mask, test_mask, test_det),
        **comparison,
    }


def _pick_detection(detections: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    candidates = [d for d in detections if str(d.get("class_name", "")).lower() == label]
    if not candidates:
        return None

    def score(det: dict[str, Any]) -> float:
        confidence = float(det.get("confidence", 1.0) or 1.0)
        area = float(det.get("mask_area", 0) or _bbox_area(det.get("bbox")) or 0)
        return confidence * max(area, 1.0)

    return max(candidates, key=score)


def _bbox_area(bbox: Any) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


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


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    cleaned = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    if int(cleaned.sum()) > 300:
        eroded = cv2.erode(cleaned, kernel, iterations=1)
        if int(eroded.sum()) > 80:
            cleaned = eroded
    return cleaned.astype(bool)


def _thermal_scalar(image_np: np.ndarray) -> np.ndarray:
    """Return a relative thermal scalar in [0, 1] from RGB or grayscale input."""
    image = image_np
    if image.ndim == 2:
        scalar = image.astype(np.float32)
    else:
        rgb = image[..., :3]
        if rgb.dtype != np.uint8:
            rgb = _normalize_to_uint8(rgb)
        scalar = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    if scalar.max() > scalar.min():
        scalar = (scalar - scalar.min()) / (scalar.max() - scalar.min())
    else:
        scalar = np.zeros_like(scalar, dtype=np.float32)
    return scalar.astype(np.float32)


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _normalize_tube(
    intensity: np.ndarray,
    mask: np.ndarray,
    config: ThermalConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    ys, xs = np.where(mask)
    if len(xs) < config.min_pixels_per_bin * 2:
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    src = _order_box_long_axis_horizontal(box)

    dst = np.array(
        [
            [0, 0],
            [config.target_width - 1, 0],
            [config.target_width - 1, config.target_height - 1],
            [0, config.target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    patch = cv2.warpPerspective(
        intensity,
        matrix,
        (config.target_width, config.target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    patch_mask = cv2.warpPerspective(
        mask.astype(np.uint8),
        matrix,
        (config.target_width, config.target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)

    if int(patch_mask.sum()) < config.min_pixels_per_bin * config.bins // 3:
        return None
    return patch.astype(np.float32), patch_mask


def _order_box_long_axis_horizontal(box: np.ndarray) -> np.ndarray:
    ordered = _order_points(box)
    tl, tr, br, bl = ordered
    horizontal = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    vertical = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if vertical > horizontal:
        return np.array([bl, tl, tr, br], dtype=np.float32)
    return ordered.astype(np.float32)


def _order_points(points: np.ndarray) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    summed = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(summed)]
    rect[2] = pts[np.argmax(summed)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _profile(patch: np.ndarray, mask: np.ndarray, config: ThermalConfig) -> dict[str, np.ndarray]:
    medians = np.full(config.bins, np.nan, dtype=np.float32)
    means = np.full(config.bins, np.nan, dtype=np.float32)
    p90 = np.full(config.bins, np.nan, dtype=np.float32)
    stds = np.full(config.bins, np.nan, dtype=np.float32)
    counts = np.zeros(config.bins, dtype=np.int32)

    edges = np.linspace(0, patch.shape[1], config.bins + 1).astype(int)
    for i in range(config.bins):
        x1, x2 = edges[i], max(edges[i + 1], edges[i] + 1)
        segment_values = patch[:, x1:x2][mask[:, x1:x2]]
        counts[i] = int(segment_values.size)
        if segment_values.size < config.min_pixels_per_bin:
            continue
        medians[i] = float(np.median(segment_values))
        means[i] = float(np.mean(segment_values))
        p90[i] = float(np.percentile(segment_values, 90))
        stds[i] = float(np.std(segment_values))

    return {
        "median": medians,
        "mean": means,
        "p90": p90,
        "std": stds,
        "count": counts,
    }


def _compare_profiles(
    ref_profile: dict[str, np.ndarray],
    test_profile: dict[str, np.ndarray],
    config: ThermalConfig,
) -> dict[str, Any]:
    ref = ref_profile["median"]
    test = test_profile["median"]
    valid = np.isfinite(ref) & np.isfinite(test)
    if config.ignore_edge_bins > 0 and config.bins > config.ignore_edge_bins * 2:
        valid[: config.ignore_edge_bins] = False
        valid[-config.ignore_edge_bins :] = False

    if int(valid.sum()) < max(4, config.bins // 4):
        return {
            "anomaly_score": 0.0,
            "severity": "unknown",
            "suspect_segments": [],
            "suspect_regions": [],
            "reason": "not enough valid profile bins",
            "profiles": {},
        }

    delta = test - ref
    baseline = float(np.nanmedian(delta[valid]))
    centered = delta - baseline
    mad = float(np.nanmedian(np.abs(centered[valid])))
    scale = max(1.4826 * mad, config.robust_scale_floor)
    z = centered / scale

    abs_z = np.abs(z)
    abs_delta = np.abs(centered)
    suspects = (
        valid
        & (abs_delta >= config.min_delta)
        & ((abs_z >= config.z_threshold) | (abs_delta >= config.strong_delta))
    )
    groups = _contiguous_groups(np.where(suspects)[0].tolist())
    significant_groups = [
        group for group in groups if len(group) >= config.min_contiguous_segments
    ]

    z_strength = np.zeros(config.bins, dtype=np.float32)
    z_strength[valid] = np.clip(
        (abs_z[valid] - config.z_threshold) / max(1.0, 7.0 - config.z_threshold),
        0.0,
        1.0,
    )
    delta_strength = np.clip(
        (abs_delta - config.strong_delta) / max(0.1, 0.55 - config.strong_delta),
        0.0,
        1.0,
    ).astype(np.float32)
    segment_strength = np.maximum(z_strength, delta_strength)
    segment_strength[~suspects] = 0.0

    max_strength = float(np.nanmax(segment_strength[valid]))
    suspect_ratio = float(len([i for group in significant_groups for i in group]) / int(valid.sum()))
    if not significant_groups:
        score = 0.0
    else:
        score = min(1.0, max_strength * 0.72 + suspect_ratio * 0.28)

    severity = _severity(score)
    suspect_segments = [int(i) for group in significant_groups for i in group]

    return {
        "anomaly_score": round(float(score), 4),
        "severity": severity,
        "baseline_delta": round(baseline, 4),
        "robust_scale": round(scale, 4),
        "valid_segments": int(valid.sum()),
        "suspect_segments": suspect_segments,
        "suspect_regions": _regions(significant_groups, segment_strength),
        "profiles": {
            "tube_ref_median": _rounded_list(ref),
            "tube_test_median": _rounded_list(test),
            "delta_centered": _rounded_list(centered),
            "z_score": _rounded_list(z),
            "segment_strength": _rounded_list(segment_strength),
        },
    }


def _contiguous_groups(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    groups: list[list[int]] = [[indices[0]]]
    for idx in indices[1:]:
        if idx == groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def _regions(groups: list[list[int]], strength: np.ndarray) -> list[dict[str, Any]]:
    regions = []
    for group in groups:
        values = strength[group]
        regions.append(
            {
                "start_segment": int(group[0]),
                "end_segment": int(group[-1]),
                "score": round(float(np.nanmax(values)), 4),
            }
        )
    return regions


def _severity(score: float) -> str:
    if score < 0.2:
        return "none"
    if score < 0.4:
        return "low"
    if score < 0.7:
        return "moderate"
    return "high"


def _tube_summary(
    patch: np.ndarray,
    patch_mask: np.ndarray,
    full_mask: np.ndarray,
    detection: dict[str, Any],
) -> dict[str, Any]:
    values = patch[patch_mask]
    return {
        "bbox": _rounded_bbox(detection.get("bbox")),
        "mask_area": int(full_mask.sum()),
        "relative_mean": round(float(np.mean(values)), 4) if values.size else None,
        "relative_median": round(float(np.median(values)), 4) if values.size else None,
        "relative_p90": round(float(np.percentile(values, 90)), 4) if values.size else None,
    }


def _rounded_bbox(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    return [round(float(v), 2) for v in bbox]


def _rounded_list(values: np.ndarray) -> list[float | None]:
    output: list[float | None] = []
    for value in values:
        if not np.isfinite(value):
            output.append(None)
        else:
            output.append(round(float(value), 4))
    return output
