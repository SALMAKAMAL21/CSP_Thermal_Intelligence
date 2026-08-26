"""
Build a siamese dataset from normal/anomaly video folders.

The script runs the current segmentation model, extracts tube_ref and tube_test
crops, and saves aligned image pairs plus a metadata CSV that can be used for
siamese training without manual frame annotation.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.api import resolve_model_path  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
TUBE_LIKE_NAMES = {"tube_ref", "tube_test", "tube", "hce"}


@dataclass
class TubeTrackState:
    tube_ref: dict[str, Any] | None = None
    tube_test: dict[str, Any] | None = None
    miss_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tube_ref/tube_test siamese pairs from videos.")
    parser.add_argument(
        "--normal-dir",
        type=Path,
        default=ROOT / "data" / "data video" / "calibration" / "normal_videos",
        help="Directory containing normal videos",
    )
    parser.add_argument(
        "--anomaly-dir",
        type=Path,
        default=ROOT / "data" / "data video" / "calibration" / "videos_anomalies",
        help="Directory containing anomaly videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "siamese_pairs",
        help="Output directory for crops and metadata",
    )
    parser.add_argument("--model", type=Path, default=None, help="Segmentation model path (.pt)")
    parser.add_argument("--sample-every-sec", type=float, default=1.0, help="Frame sampling period in seconds")
    parser.add_argument("--max-frames-per-video", type=int, default=0, help="Optional cap of saved frames per video")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--input-size", type=int, default=128, help="Saved crop size for siamese training")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Video-level validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return parser.parse_args()


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect_videos(input_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]


def detection_score(det: dict[str, Any]) -> float:
    confidence = float(det.get("confidence", 0.0) or 0.0)
    area = float(det.get("mask_area", 0.0) or 0.0)
    if area <= 0:
        bbox = det.get("bbox") or [0, 0, 0, 0]
        area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return confidence * max(area, 1.0)


def clone_detection(det: dict[str, Any], class_name: str | None = None) -> dict[str, Any]:
    cloned = dict(det)
    if "bbox" in cloned and isinstance(cloned["bbox"], list):
        cloned["bbox"] = list(cloned["bbox"])
    if "mask_polygon" in cloned and isinstance(cloned["mask_polygon"], list):
        cloned["mask_polygon"] = [list(point) for point in cloned["mask_polygon"]]
    if class_name is not None:
        cloned["class_name"] = class_name
    return cloned


def detection_center(det: dict[str, Any]) -> tuple[float, float]:
    bbox = det.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return (float("inf"), float("inf"))
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def distance(det_a: dict[str, Any], det_b: dict[str, Any]) -> float:
    ax, ay = detection_center(det_a)
    bx, by = detection_center(det_b)
    return float(np.hypot(ax - bx, ay - by))


def assign_ref_test_roles(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tube_candidates = [det for det in detections if det.get("class_name") in TUBE_LIKE_NAMES]
    if len(tube_candidates) < 2:
        return detections
    current = sorted(tube_candidates, key=detection_score, reverse=True)[:2]
    current = sorted(current, key=lambda det: ((det["bbox"][1] + det["bbox"][3]) / 2.0))
    current[0]["class_name"] = "tube_ref"
    current[1]["class_name"] = "tube_test"
    return detections


def apply_tube_tracking(
    detections: list[dict[str, Any]],
    state: TubeTrackState,
    max_misses: int = 6,
) -> list[dict[str, Any]]:
    tube_candidates = [det for det in detections if det.get("class_name") in TUBE_LIKE_NAMES]
    if not tube_candidates:
        if state.tube_ref is not None and state.tube_test is not None and state.miss_count < max_misses:
            state.miss_count += 1
            recovered = list(detections)
            recovered.append(clone_detection(state.tube_ref, "tube_ref"))
            recovered.append(clone_detection(state.tube_test, "tube_test"))
            return recovered
        state.tube_ref = None
        state.tube_test = None
        state.miss_count = 0
        return detections

    current = sorted(tube_candidates, key=detection_score, reverse=True)[:2]
    current = sorted(current, key=lambda det: ((det["bbox"][1] + det["bbox"][3]) / 2.0))

    if len(current) >= 2:
        current[0]["class_name"] = "tube_ref"
        current[1]["class_name"] = "tube_test"
        state.tube_ref = clone_detection(current[0], "tube_ref")
        state.tube_test = clone_detection(current[1], "tube_test")
        state.miss_count = 0
        return detections

    current_det = current[0]
    if state.tube_ref is not None and state.tube_test is not None:
        distance_to_ref = distance(current_det, state.tube_ref)
        distance_to_test = distance(current_det, state.tube_test)
        recovered = list(detections)
        if distance_to_ref <= distance_to_test:
            current_det["class_name"] = "tube_ref"
            state.tube_ref = clone_detection(current_det, "tube_ref")
            recovered.append(clone_detection(state.tube_test, "tube_test"))
        else:
            current_det["class_name"] = "tube_test"
            state.tube_test = clone_detection(current_det, "tube_test")
            recovered.append(clone_detection(state.tube_ref, "tube_ref"))
        state.miss_count = 0
        return recovered

    current_det["class_name"] = "tube_ref"
    state.tube_ref = clone_detection(current_det, "tube_ref")
    state.miss_count += 1
    return detections


def format_detection(
    box: Any,
    model_names: dict[int, str],
    mask_data: Any,
    image_shape: tuple[int, int, int],
    index: int,
) -> dict[str, Any]:
    detection = {
        "class_id": int(box.cls[0]),
        "class_name": model_names[int(box.cls[0])],
        "confidence": round(float(box.conf[0]), 4),
        "bbox": box.xyxy[0].cpu().numpy().tolist(),
    }
    if mask_data is not None and index < len(mask_data):
        mask = mask_data[index].cpu().numpy()
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask_uint8, (image_shape[1], image_shape[0]))
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            detection["mask_polygon"] = largest.reshape(-1, 2).tolist()
            detection["mask_area"] = int(cv2.contourArea(largest))
    return detection


def detection_to_mask(det: dict[str, Any], height: int, width: int) -> np.ndarray | None:
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


def prepare_crop(image_np: np.ndarray, mask: np.ndarray, input_size: int) -> np.ndarray | None:
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
    return cv2.resize(canvas, (input_size, input_size), interpolation=cv2.INTER_AREA)


def save_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def split_videos(normal_videos: list[Path], anomaly_videos: list[Path], val_ratio: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    grouped = {0: list(normal_videos), 1: list(anomaly_videos)}
    video_split: dict[str, str] = {}
    for paths in grouped.values():
        rng.shuffle(paths)
        val_count = max(1, int(round(len(paths) * val_ratio))) if len(paths) >= 2 else 0
        val_names = {path.name for path in paths[:val_count]}
        for path in paths:
            video_split[path.name] = "val" if path.name in val_names else "train"
    return video_split


def process_video(
    video_path: Path,
    label: int,
    split: str,
    model: YOLO,
    args: argparse.Namespace,
    device: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats = {
        "sampled_frames": 0,
        "missing_pair": 0,
        "missing_mask": 0,
        "invalid_crop": 0,
        "saved_pairs": 0,
    }
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return rows

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_every = max(int(round(max(fps, 1.0) * args.sample_every_sec)), 1)
    track_state = TubeTrackState()

    frame_idx = 0
    saved = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue
        if args.max_frames_per_video > 0 and saved >= args.max_frames_per_video:
            break
        stats["sampled_frames"] += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = model.predict(
            frame_rgb,
            device=device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
        detections = [
            format_detection(box, model.names, result.masks.data if result.masks is not None else None, frame_rgb.shape, i)
            for i, box in enumerate(result.boxes)
        ]
        detections = assign_ref_test_roles(detections)
        detections = apply_tube_tracking(detections, track_state)

        ref_det = next((det for det in detections if det.get("class_name") == "tube_ref"), None)
        test_det = next((det for det in detections if det.get("class_name") == "tube_test"), None)
        if ref_det is None or test_det is None:
            stats["missing_pair"] += 1
            frame_idx += 1
            continue

        ref_mask = detection_to_mask(ref_det, frame_rgb.shape[0], frame_rgb.shape[1])
        test_mask = detection_to_mask(test_det, frame_rgb.shape[0], frame_rgb.shape[1])
        if ref_mask is None or test_mask is None:
            stats["missing_mask"] += 1
            frame_idx += 1
            continue

        ref_crop = prepare_crop(frame_rgb, ref_mask, args.input_size)
        test_crop = prepare_crop(frame_rgb, test_mask, args.input_size)
        if ref_crop is None or test_crop is None:
            stats["invalid_crop"] += 1
            frame_idx += 1
            continue

        pair_id = f"{video_path.stem}_f{frame_idx:06d}"
        ref_path = output_dir / split / ("anomaly" if label == 1 else "normal") / "ref" / f"{pair_id}.png"
        test_path = output_dir / split / ("anomaly" if label == 1 else "normal") / "test" / f"{pair_id}.png"
        save_image(ref_path, ref_crop)
        save_image(test_path, test_crop)

        rows.append(
            {
                "pair_id": pair_id,
                "split": split,
                "label": label,
                "label_name": "anomaly" if label == 1 else "normal",
                "video_name": video_path.name,
                "frame_index": frame_idx,
                "timestamp_s": round(frame_idx / max(fps, 1.0), 3),
                "ref_path": str(ref_path.relative_to(output_dir)),
                "test_path": str(test_path.relative_to(output_dir)),
                "total_video_frames": frame_count,
            }
        )
        saved += 1
        stats["saved_pairs"] += 1
        frame_idx += 1

    cap.release()
    if rows:
        print(
            f"{video_path.name}: sampled={stats['sampled_frames']} "
            f"saved={stats['saved_pairs']} missing_pair={stats['missing_pair']} "
            f"missing_mask={stats['missing_mask']} invalid_crop={stats['invalid_crop']}"
        )
    else:
        print(
            f"{video_path.name}: sampled={stats['sampled_frames']} "
            f"saved=0 missing_pair={stats['missing_pair']} "
            f"missing_mask={stats['missing_mask']} invalid_crop={stats['invalid_crop']}"
        )
    return rows


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model_path = args.model or resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Segmentation model not found: {model_path}")

    normal_videos = collect_videos(args.normal_dir)
    anomaly_videos = collect_videos(args.anomaly_dir)
    if not normal_videos:
        raise FileNotFoundError(f"No normal videos found in {args.normal_dir}")
    if not anomaly_videos:
        raise FileNotFoundError(f"No anomaly videos found in {args.anomaly_dir}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement du modele de segmentation: {model_path.name}")
    print(f"Device: {device}")
    model = YOLO(str(model_path))

    split_map = split_videos(normal_videos, anomaly_videos, args.val_ratio, args.seed)
    rows: list[dict[str, Any]] = []

    print(f"{len(normal_videos)} videos normales | {len(anomaly_videos)} videos anomalies")
    for label, videos in ((0, normal_videos), (1, anomaly_videos)):
        for video_path in videos:
            split = split_map[video_path.name]
            video_rows = process_video(video_path, label, split, model, args, device, output_dir)
            rows.extend(video_rows)
            print(f"{video_path.name}: {len(video_rows)} paires sauvegardees ({split}, label={label})")

    metadata_path = output_dir / "metadata.csv"
    fieldnames = [
        "pair_id",
        "split",
        "label",
        "label_name",
        "video_name",
        "frame_index",
        "timestamp_s",
        "ref_path",
        "test_path",
        "total_video_frames",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    train_count = sum(1 for row in rows if row["split"] == "train")
    val_count = sum(1 for row in rows if row["split"] == "val")
    normal_count = sum(1 for row in rows if int(row["label"]) == 0)
    anomaly_count = sum(1 for row in rows if int(row["label"]) == 1)

    print("-" * 60)
    print(f"Paires totales   : {len(rows)}")
    print(f"Train / Val      : {train_count} / {val_count}")
    print(f"Normal / Anomaly : {normal_count} / {anomaly_count}")
    print(f"Metadata         : {metadata_path}")


if __name__ == "__main__":
    main()
