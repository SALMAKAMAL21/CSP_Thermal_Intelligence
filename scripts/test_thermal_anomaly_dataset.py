"""
Run the deterministic thermal anomaly module on normal dataset images.

The dataset labels provide tube_ref and tube_test polygons. This script uses
those labels directly, so the test focuses on the thermal comparison module and
does not depend on loading YOLO.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.thermal_anomaly import analyze_ref_test_thermal  # noqa: E402


NAMES = ["tube_ref", "tube_test"]


def image_to_label_path(image_path: Path, dataset_dir: Path) -> Path:
    relative = image_path.relative_to(dataset_dir)
    return dataset_dir / relative.parent.parent / "labels" / f"{image_path.stem}.txt"


def parse_yolo_seg_label(label_path: Path, image_shape: tuple[int, int]) -> list[dict]:
    height, width = image_shape
    detections = []

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue

        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
        points = []
        for x_norm, y_norm in zip(coords[0::2], coords[1::2]):
            points.append([x_norm * width, y_norm * height])
        polygon = np.array(points, dtype=np.float32)
        if len(polygon) < 3:
            continue

        x1, y1 = polygon.min(axis=0)
        x2, y2 = polygon.max(axis=0)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)

        detections.append(
            {
                "class_id": class_id,
                "class_name": NAMES[class_id] if class_id < len(NAMES) else str(class_id),
                "confidence": 1.0,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "mask_polygon": polygon.tolist(),
                "mask_area": int(mask.sum()),
            }
        )

    return detections


def collect_images(dataset_dir: Path, split: str) -> list[Path]:
    image_dir = dataset_dir / split / "images"
    return sorted(image_dir.glob("*_T_*.png"))


def run(args: argparse.Namespace) -> int:
    dataset_dir = ROOT / "data" / "tubes_csp_yolov11_grouped"
    images = collect_images(dataset_dir, args.split)
    if not images:
        print(f"No thermal PNG images found in split={args.split}")
        return 1

    random.seed(args.seed)
    selected = random.sample(images, min(args.limit, len(images)))
    results = []

    for image_path in selected:
        label_path = image_to_label_path(image_path, dataset_dir)
        if not label_path.exists():
            continue

        image = np.array(Image.open(image_path).convert("RGB"))
        detections = parse_yolo_seg_label(label_path, image.shape[:2])
        analysis = analyze_ref_test_thermal(image, detections)
        score = float(analysis.get("anomaly_score", 0.0) or 0.0)
        severity = analysis.get("severity", analysis.get("status", "?"))
        results.append(score)
        print(
            f"{image_path.name} | score={score:.4f} | severity={severity} | "
            f"status={analysis.get('status')} | suspects={analysis.get('suspect_segments', [])}"
        )

    if not results:
        print("No usable labelled thermal images were tested.")
        return 1

    arr = np.array(results, dtype=np.float32)
    print("-" * 80)
    print(
        "summary "
        f"n={len(arr)} "
        f"mean={arr.mean():.4f} "
        f"median={np.median(arr):.4f} "
        f"p90={np.percentile(arr, 90):.4f} "
        f"max={arr.max():.4f}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test thermal anomaly scoring on normal dataset images")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
