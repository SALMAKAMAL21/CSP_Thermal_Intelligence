"""
Calibrate the relative thermal anomaly thresholds from normal videos.

This script samples frames from a directory of videos, runs the current
segmentation + ref/test thermal comparison pipeline, and exports:

- frame_metrics.csv: one row per sampled frame
- summary.json: aggregate statistics
- recommended_thresholds.json: suggested thresholds based on normal data
- top_frames/: highest-scoring frames saved for visual inspection

Usage:
    python scripts/calibrate_thermal_from_videos.py \
        --input-dir data/calibration/normal_videos \
        --output-dir artifacts/thermal_calibration
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.api import resolve_model_path  # noqa: E402
from src.inference.thermal_anomaly import DEFAULT_CONFIG, ThermalConfig, analyze_ref_test_thermal  # noqa: E402
from src.inference.video_decision import load_video_decision_config, summarize_video_scores  # noqa: E402


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
TUBE_LIKE_NAMES = {"tube_ref", "tube_test", "tube", "hce"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate thermal anomaly thresholds from normal videos."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing normal videos")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "thermal_calibration",
        help="Directory for CSV/JSON outputs",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to segmentation model (.pt). Defaults to backend resolution logic.",
    )
    parser.add_argument("--sample-every-sec", type=float, default=1.0, help="Frame sampling period in seconds")
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=0,
        help="Optional cap on sampled frames per video. 0 means no cap.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument(
        "--top-k-frames",
        type=int,
        default=20,
        help="Number of highest-scoring frames to export for inspection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device",
    )
    parser.add_argument("--z-margin", type=float, default=0.25, help="Safety margin added to normal p99 z-score")
    parser.add_argument(
        "--delta-margin",
        type=float,
        default=0.02,
        help="Safety margin added to normal p99 delta metrics",
    )
    parser.add_argument(
        "--video-score-percentile",
        type=float,
        default=99.0,
        help="Percentile of normal video max score used for alert recommendation",
    )
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
    videos = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return videos


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


def detection_score(det: dict[str, Any]) -> float:
    confidence = float(det.get("confidence", 0.0) or 0.0)
    area = float(det.get("mask_area", 0.0) or 0.0)
    if area <= 0:
        bbox = det.get("bbox") or [0, 0, 0, 0]
        area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return confidence * max(area, 1.0)


def assign_ref_test_roles(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tube_candidates = [det for det in detections if det.get("class_name") in TUBE_LIKE_NAMES]
    if len(tube_candidates) < 2:
        return detections

    current = sorted(tube_candidates, key=detection_score, reverse=True)[:2]
    current = sorted(current, key=lambda det: ((det["bbox"][1] + det["bbox"][3]) / 2.0))
    current[0]["class_name"] = "tube_ref"
    current[1]["class_name"] = "tube_test"
    return detections


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float32), q))


def safe_round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def flatten_metrics(
    video_path: Path,
    frame_index: int,
    timestamp_s: float,
    detections: list[dict[str, Any]],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    profiles = analysis.get("profiles") or {}
    delta_centered = [abs(v) for v in profiles.get("delta_centered", []) if isinstance(v, (int, float))]
    z_scores = [abs(v) for v in profiles.get("z_score", []) if isinstance(v, (int, float))]
    segment_strength = [v for v in profiles.get("segment_strength", []) if isinstance(v, (int, float))]
    tube_ref = analysis.get("tube_ref") or {}
    tube_test = analysis.get("tube_test") or {}

    return {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "frame_index": frame_index,
        "timestamp_s": round(timestamp_s, 3),
        "status": analysis.get("status"),
        "severity": analysis.get("severity"),
        "anomaly_score": safe_round(float(analysis.get("anomaly_score", 0.0) or 0.0)),
        "baseline_delta": safe_round(float(analysis.get("baseline_delta", 0.0) or 0.0)),
        "robust_scale": safe_round(float(analysis.get("robust_scale", 0.0) or 0.0)),
        "valid_segments": int(analysis.get("valid_segments", 0) or 0),
        "suspect_segment_count": len(analysis.get("suspect_segments", []) or []),
        "suspect_region_count": len(analysis.get("suspect_regions", []) or []),
        "max_abs_delta_centered": safe_round(max(delta_centered) if delta_centered else None),
        "max_abs_z_score": safe_round(max(z_scores) if z_scores else None),
        "max_segment_strength": safe_round(max(segment_strength) if segment_strength else None),
        "tube_ref_mean": safe_round(tube_ref.get("relative_mean")),
        "tube_ref_p90": safe_round(tube_ref.get("relative_p90")),
        "tube_test_mean": safe_round(tube_test.get("relative_mean")),
        "tube_test_p90": safe_round(tube_test.get("relative_p90")),
        "total_detections": len(detections),
        "tube_detection_count": sum(1 for det in detections if det.get("class_name") in TUBE_LIKE_NAMES),
    }


def summarize_frames(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    scores = [float(row["anomaly_score"]) for row in ok_rows if row.get("anomaly_score") is not None]
    deltas = [float(row["max_abs_delta_centered"]) for row in ok_rows if row.get("max_abs_delta_centered") is not None]
    z_scores = [float(row["max_abs_z_score"]) for row in ok_rows if row.get("max_abs_z_score") is not None]
    strengths = [float(row["max_segment_strength"]) for row in ok_rows if row.get("max_segment_strength") is not None]

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "total_frames": len(rows),
        "usable_frames": len(ok_rows),
        "usable_ratio": safe_round(len(ok_rows) / len(rows), 4) if rows else None,
        "status_counts": status_counts,
        "score_stats": build_stats(scores),
        "delta_stats": build_stats(deltas),
        "z_score_stats": build_stats(z_scores),
        "segment_strength_stats": build_stats(strengths),
    }


def summarize_videos(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decision_config = load_video_decision_config()
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_video.setdefault(str(row["video_name"]), []).append(row)

    videos = []
    video_max_scores = []
    video_mean_scores = []
    video_suspect_ratios = []

    for video_name, items in by_video.items():
        ok_items = [item for item in items if item.get("status") == "ok"]
        scores = [float(item["anomaly_score"]) for item in ok_items if item.get("anomaly_score") is not None]
        suspect_items = [item for item in ok_items if float(item.get("anomaly_score") or 0.0) >= 0.2]
        decision = summarize_video_scores(ok_items, decision_config)
        summary = {
            "video_name": video_name,
            "sampled_frames": len(items),
            "usable_frames": len(ok_items),
            "max_anomaly_score": safe_round(max(scores) if scores else None),
            "mean_anomaly_score": safe_round(float(np.mean(scores)) if scores else None),
            "suspect_frame_ratio": safe_round(len(suspect_items) / len(ok_items), 4) if ok_items else None,
            "longest_suspect_run": decision["longest_suspect_run"],
            "decision": decision["decision"],
            "decision_confidence": decision["decision_confidence"],
            "peak_frame_time": decision["peak_frame_time"],
        }
        videos.append(summary)

        if summary["max_anomaly_score"] is not None:
            video_max_scores.append(float(summary["max_anomaly_score"]))
        if summary["mean_anomaly_score"] is not None:
            video_mean_scores.append(float(summary["mean_anomaly_score"]))
        if summary["suspect_frame_ratio"] is not None:
            video_suspect_ratios.append(float(summary["suspect_frame_ratio"]))

    return {
        "video_count": len(videos),
        "videos": videos,
        "video_max_score_stats": build_stats(video_max_scores),
        "video_mean_score_stats": build_stats(video_mean_scores),
        "video_suspect_ratio_stats": build_stats(video_suspect_ratios),
    }


def build_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    arr = np.array(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": safe_round(float(arr.mean())),
        "median": safe_round(float(np.median(arr))),
        "p90": safe_round(float(np.percentile(arr, 90))),
        "p95": safe_round(float(np.percentile(arr, 95))),
        "p99": safe_round(float(np.percentile(arr, 99))),
        "max": safe_round(float(arr.max())),
    }


def recommend_thresholds(
    frame_summary: dict[str, Any],
    video_summary: dict[str, Any],
    base_config: ThermalConfig,
    z_margin: float,
    delta_margin: float,
    video_score_percentile: float,
) -> dict[str, Any]:
    score_p99 = frame_summary["score_stats"].get("p99")
    delta_p99 = frame_summary["delta_stats"].get("p99")
    z_p99 = frame_summary["z_score_stats"].get("p99")
    video_max_scores = [
        float(video["max_anomaly_score"])
        for video in video_summary.get("videos", [])
        if video.get("max_anomaly_score") is not None
    ]
    video_alert_score = percentile(video_max_scores, video_score_percentile)

    min_delta = max(base_config.min_delta, (delta_p99 or 0.0) + delta_margin)
    strong_delta = max(base_config.strong_delta, min_delta + delta_margin, (delta_p99 or 0.0) + (2 * delta_margin))
    z_threshold = max(base_config.z_threshold, (z_p99 or 0.0) + z_margin)

    return {
        "derived_from": {
            "frame_score_p99": safe_round(score_p99),
            "frame_delta_p99": safe_round(delta_p99),
            "frame_z_p99": safe_round(z_p99),
            "video_max_score_percentile": video_score_percentile,
            "video_max_score_value": safe_round(video_alert_score),
            "base_config": asdict(base_config),
        },
        "recommended_thresholds": {
            "z_threshold": safe_round(z_threshold),
            "min_delta": safe_round(min_delta),
            "strong_delta": safe_round(strong_delta),
            "video_alert_score": safe_round(max(0.2, video_alert_score or 0.0)),
            "video_warning_score": safe_round(max(0.15, (score_p99 or 0.0))),
        },
        "notes": [
            "Recommendations are derived from normal videos only.",
            "Validate these thresholds on defect videos before freezing them in config.",
            "If too many normal videos still trigger alerts, increase the safety margins.",
        ],
    }


def export_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_top_frames(top_frames: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, item in enumerate(top_frames, start=1):
        frame_bgr = item["frame_bgr"].copy()
        cv2.putText(
            frame_bgr,
            f"rank={rank} score={item['row']['anomaly_score']:.3f} t={item['row']['timestamp_s']:.2f}s",
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        filename = (
            f"{rank:02d}_"
            f"{Path(item['row']['video_name']).stem}_"
            f"f{int(item['row']['frame_index']):06d}_"
            f"s{item['row']['anomaly_score']:.3f}.jpg"
        )
        cv2.imwrite(str(output_dir / filename), frame_bgr)


def should_sample(frame_index: int, sample_stride: int) -> bool:
    return frame_index % sample_stride == 0


def run_video(
    video_path: Path,
    model: YOLO,
    device: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open {video_path}")
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    sample_stride = max(1, int(round(fps * args.sample_every_sec)))

    rows: list[dict[str, Any]] = []
    top_candidates: list[dict[str, Any]] = []
    frame_index = 0
    sampled = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if not should_sample(frame_index, sample_stride):
            frame_index += 1
            continue

        sampled += 1
        if args.max_frames_per_video > 0 and sampled > args.max_frames_per_video:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = model.predict(
            frame_rgb,
            device=device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )
        result = results[0]

        detections = []
        mask_data = result.masks.data if result.masks is not None else None
        for i, box in enumerate(result.boxes):
            detections.append(format_detection(box, model.names, mask_data, frame_rgb.shape, i))

        detections = assign_ref_test_roles(detections)
        analysis = analyze_ref_test_thermal(frame_rgb, detections)
        row = flatten_metrics(video_path, frame_index, frame_index / fps, detections, analysis)
        rows.append(row)

        if row["status"] == "ok":
            top_candidates.append({"row": row, "frame_bgr": frame_bgr.copy()})

        frame_index += 1

    cap.release()
    return rows, top_candidates


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(input_dir)
    if not videos:
        print(f"No videos found in {input_dir}")
        return 1

    model_path = args.model.resolve() if args.model else resolve_model_path().resolve()
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    device = get_device(args.device)
    print(f"[INFO] videos={len(videos)} model={model_path.name} device={device}")

    model = YOLO(str(model_path))
    all_rows: list[dict[str, Any]] = []
    all_top_candidates: list[dict[str, Any]] = []

    for video_path in videos:
        print(f"[INFO] processing {video_path.name}")
        rows, top_candidates = run_video(video_path, model, device, args)
        all_rows.extend(rows)
        all_top_candidates.extend(top_candidates)
        usable = sum(1 for row in rows if row.get("status") == "ok")
        print(f"[INFO] sampled_frames={len(rows)} usable_frames={usable}")

    if not all_rows:
        print("No frames were processed.")
        return 1

    frame_summary = summarize_frames(all_rows)
    video_summary = summarize_videos(all_rows)
    thresholds = recommend_thresholds(
        frame_summary=frame_summary,
        video_summary=video_summary,
        base_config=DEFAULT_CONFIG,
        z_margin=args.z_margin,
        delta_margin=args.delta_margin,
        video_score_percentile=args.video_score_percentile,
    )

    export_csv(all_rows, output_dir / "frame_metrics.csv")
    export_json(frame_summary, output_dir / "summary_frames.json")
    export_json(video_summary, output_dir / "summary_videos.json")
    export_json(thresholds, output_dir / "recommended_thresholds.json")

    sorted_top_frames = sorted(
        all_top_candidates,
        key=lambda item: float(item["row"].get("anomaly_score") or 0.0),
        reverse=True,
    )[: max(0, args.top_k_frames)]
    if sorted_top_frames:
        save_top_frames(sorted_top_frames, output_dir / "top_frames")

    print("[DONE] Outputs written to:")
    print(f"  - {output_dir / 'frame_metrics.csv'}")
    print(f"  - {output_dir / 'summary_frames.json'}")
    print(f"  - {output_dir / 'summary_videos.json'}")
    print(f"  - {output_dir / 'recommended_thresholds.json'}")
    if sorted_top_frames:
        print(f"  - {output_dir / 'top_frames'}")

    recommended = thresholds["recommended_thresholds"]
    print("[DONE] Recommended thresholds:")
    print(f"  - z_threshold={recommended['z_threshold']}")
    print(f"  - min_delta={recommended['min_delta']}")
    print(f"  - strong_delta={recommended['strong_delta']}")
    print(f"  - video_warning_score={recommended['video_warning_score']}")
    print(f"  - video_alert_score={recommended['video_alert_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
