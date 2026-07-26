"""
Video-level decision logic for ref/test thermal anomaly analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "thermal_video_decision.json"


@dataclass(frozen=True)
class VideoDecisionConfig:
    frame_warning_score: float = 0.24
    frame_alert_score: float = 0.38
    video_mean_warning_score: float = 0.02
    video_mean_alert_score: float = 0.05
    suspect_ratio_warning: float = 0.05
    suspect_ratio_alert: float = 0.10
    min_persistent_frames: int = 2


DEFAULT_VIDEO_DECISION_CONFIG = VideoDecisionConfig()


def load_video_decision_config(path: Path | None = None) -> VideoDecisionConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return DEFAULT_VIDEO_DECISION_CONFIG

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return VideoDecisionConfig(
        frame_warning_score=float(payload.get("frame_warning_score", DEFAULT_VIDEO_DECISION_CONFIG.frame_warning_score)),
        frame_alert_score=float(payload.get("frame_alert_score", DEFAULT_VIDEO_DECISION_CONFIG.frame_alert_score)),
        video_mean_warning_score=float(
            payload.get("video_mean_warning_score", DEFAULT_VIDEO_DECISION_CONFIG.video_mean_warning_score)
        ),
        video_mean_alert_score=float(
            payload.get("video_mean_alert_score", DEFAULT_VIDEO_DECISION_CONFIG.video_mean_alert_score)
        ),
        suspect_ratio_warning=float(
            payload.get("suspect_ratio_warning", DEFAULT_VIDEO_DECISION_CONFIG.suspect_ratio_warning)
        ),
        suspect_ratio_alert=float(
            payload.get("suspect_ratio_alert", DEFAULT_VIDEO_DECISION_CONFIG.suspect_ratio_alert)
        ),
        min_persistent_frames=int(
            payload.get("min_persistent_frames", DEFAULT_VIDEO_DECISION_CONFIG.min_persistent_frames)
        ),
    )


def summarize_video_scores(
    frame_results: list[dict[str, Any]],
    config: VideoDecisionConfig = DEFAULT_VIDEO_DECISION_CONFIG,
) -> dict[str, Any]:
    usable = [row for row in frame_results if row.get("status") == "ok"]
    if not usable:
        return {
            "decision": "unknown",
            "decision_confidence": "low",
            "peak_frame_time": None,
            "max_anomaly_score": 0.0,
            "mean_anomaly_score": 0.0,
            "suspect_frame_ratio": 0.0,
            "longest_suspect_run": 0,
            "warning_frame_count": 0,
            "alert_frame_count": 0,
            "config": asdict(config),
        }

    scores = [float(row.get("anomaly_score", 0.0) or 0.0) for row in usable]
    warning_flags = [score >= config.frame_warning_score for score in scores]
    alert_flags = [score >= config.frame_alert_score for score in scores]

    max_idx = max(range(len(scores)), key=lambda idx: scores[idx])
    max_score = scores[max_idx]
    mean_score = sum(scores) / len(scores)
    warning_count = sum(1 for flag in warning_flags if flag)
    alert_count = sum(1 for flag in alert_flags if flag)
    suspect_ratio = warning_count / len(scores)
    longest_run = _longest_true_run(warning_flags)
    peak_time = usable[max_idx].get("timestamp_s")

    if max_score >= config.frame_alert_score and (
        suspect_ratio >= config.suspect_ratio_alert or longest_run >= config.min_persistent_frames
    ):
        decision = "anomaly"
    elif (
        max_score >= config.frame_warning_score
        or mean_score >= config.video_mean_warning_score
        or suspect_ratio >= config.suspect_ratio_warning
    ):
        decision = "warning"
    else:
        decision = "normal"

    confidence = _decision_confidence(
        decision=decision,
        max_score=max_score,
        mean_score=mean_score,
        suspect_ratio=suspect_ratio,
        longest_run=longest_run,
        config=config,
    )

    return {
        "decision": decision,
        "decision_confidence": confidence,
        "peak_frame_time": round(float(peak_time), 3) if peak_time is not None else None,
        "max_anomaly_score": round(float(max_score), 4),
        "mean_anomaly_score": round(float(mean_score), 4),
        "suspect_frame_ratio": round(float(suspect_ratio), 4),
        "longest_suspect_run": int(longest_run),
        "warning_frame_count": int(warning_count),
        "alert_frame_count": int(alert_count),
        "config": asdict(config),
    }


def _longest_true_run(flags: list[bool]) -> int:
    longest = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _decision_confidence(
    decision: str,
    max_score: float,
    mean_score: float,
    suspect_ratio: float,
    longest_run: int,
    config: VideoDecisionConfig,
) -> str:
    if decision == "anomaly":
        strong_alert = (
            max_score >= config.frame_alert_score + 0.1
            and suspect_ratio >= max(config.suspect_ratio_alert, 0.15)
            and longest_run >= config.min_persistent_frames
        )
        return "high" if strong_alert else "medium"
    if decision == "warning":
        medium_warning = (
            max_score >= config.frame_warning_score + 0.05
            or mean_score >= config.video_mean_alert_score
            or suspect_ratio >= config.suspect_ratio_alert
        )
        return "medium" if medium_warning else "low"
    return "high"
