"""
API FastAPI pour l'inference YOLOv11n-seg
==========================================
Recoit des images du drone et retourne les segmentations des composants CSP.

Usage:
    uvicorn src.inference.api:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict          - Segmentation d'une image
    GET  /health           - Status de l'API
    GET  /model/info       - Informations sur le modele
"""

import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

from .autoencoder_anomaly import analyze_ref_test_autoencoder, build_combined_anomaly, load_autoencoder_model
from .siamese_anomaly import analyze_ref_test_siamese, build_combined_with_siamese, load_siamese_model
from .thermal_anomaly import analyze_ref_test_thermal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUTOENCODER_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "best_autoencoder.pt"
SIAMESE_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "best_siamese.pt"


def resolve_model_path() -> Path:
    """Resout le chemin du modele avec priorite a MODEL_PATH puis yolo_seg.pt."""
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path)

    candidates = [
        PROJECT_ROOT / "ml" / "best_model.pt",
        PROJECT_ROOT / "ml" / "yolo_seg.pt",
        PROJECT_ROOT / "src" / "models" / "best_model.pt",
        PROJECT_ROOT / "src" / "models" / "yolo_seg.pt",
        PROJECT_ROOT / "src" / "models" / "yolo-seg.pt",
        PROJECT_ROOT / "src" / "models" / "csp_seg1.pt",
        PROJECT_ROOT / "csp_seg1.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Fallback: premiere cible attendue
    return candidates[0]


MODEL_PATH = resolve_model_path()

app = FastAPI(
    title="Green Energy Park - CSP Tube Detection API",
    description="Detection et segmentation des tubes recepteurs CSP par drone",
    version="1.0.0",
)

# Variable globale pour le modele
model = None
autoencoder_model = None
siamese_model = None
siamese_config = None
inference_device = "cpu"


@dataclass
class TubeTrackState:
    tube_ref: dict | None = None
    tube_test: dict | None = None
    miss_count: int = 0
    updated_at: float = 0.0


TRACKER_TTL_SECONDS = 120
TRACKER_MAX_MISSES = 4
SESSION_TRACKERS: dict[str, TubeTrackState] = {}


def _cleanup_session_trackers() -> None:
    now = time.time()
    expired = [
        session_id
        for session_id, state in SESSION_TRACKERS.items()
        if now - state.updated_at > TRACKER_TTL_SECONDS
    ]
    for session_id in expired:
        SESSION_TRACKERS.pop(session_id, None)


def _detection_center(det: dict) -> tuple[float, float]:
    bbox = det.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return (float("inf"), float("inf"))
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _distance(det_a: dict, det_b: dict) -> float:
    ax, ay = _detection_center(det_a)
    bx, by = _detection_center(det_b)
    return float(np.hypot(ax - bx, ay - by))


def _detection_score(det: dict) -> float:
    confidence = float(det.get("confidence", 0.0) or 0.0)
    area = float(det.get("mask_area", 0.0) or 0.0)
    if area <= 0:
        bbox = det.get("bbox") or [0, 0, 0, 0]
        area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return confidence * max(area, 1.0)


def _clone_detection(det: dict, class_name: str | None = None) -> dict:
    cloned = dict(det)
    if "bbox" in cloned and isinstance(cloned["bbox"], list):
        cloned["bbox"] = list(cloned["bbox"])
    if "mask_polygon" in cloned and isinstance(cloned["mask_polygon"], list):
        cloned["mask_polygon"] = [list(point) for point in cloned["mask_polygon"]]
    if class_name is not None:
        cloned["class_name"] = class_name
    return cloned


def _assign_roles_from_current(detections: list[dict]) -> list[dict]:
    ordered = sorted(
        detections,
        key=lambda det: ((det["bbox"][1] + det["bbox"][3]) / 2.0),
    )
    if len(ordered) < 2:
        return []
    top = _clone_detection(ordered[0], "tube_ref")
    bottom = _clone_detection(ordered[-1], "tube_test")
    return [top, bottom]


def _apply_tube_tracking(detections: list[dict], session_id: str | None, reset_tracker: bool) -> list[dict]:
    tube_like_names = {"tube_ref", "tube_test", "tube", "hce"}
    tube_candidates = [det for det in detections if det.get("class_name") in tube_like_names]
    if not tube_candidates:
        return detections

    current = sorted(tube_candidates, key=_detection_score, reverse=True)[:2]
    current = sorted(current, key=lambda det: ((det["bbox"][1] + det["bbox"][3]) / 2.0))

    if not session_id:
        if len(current) >= 2:
            assigned = _assign_roles_from_current(current)
            for source, mapped in zip(current[:2], assigned):
                source["class_name"] = mapped["class_name"]
        return detections

    _cleanup_session_trackers()
    if reset_tracker or session_id not in SESSION_TRACKERS:
        SESSION_TRACKERS[session_id] = TubeTrackState(updated_at=time.time())

    state = SESSION_TRACKERS[session_id]
    state.updated_at = time.time()

    if len(current) >= 2:
        assigned = _assign_roles_from_current(current)
        state.tube_ref = assigned[0]
        state.tube_test = assigned[1]
        state.miss_count = 0
        for source, mapped in zip(current[:2], assigned):
            source["class_name"] = mapped["class_name"]
        return detections

    if len(current) == 1 and state.tube_ref is not None and state.tube_test is not None:
        current_det = current[0]
        distance_to_ref = _distance(current_det, state.tube_ref)
        distance_to_test = _distance(current_det, state.tube_test)
        if distance_to_ref <= distance_to_test:
            current_det["class_name"] = "tube_ref"
            state.tube_ref = _clone_detection(current_det, "tube_ref")
        else:
            current_det["class_name"] = "tube_test"
            state.tube_test = _clone_detection(current_det, "tube_test")
        state.miss_count = 0
        return detections

    if len(current) == 1:
        current[0]["class_name"] = "tube_ref" if _detection_center(current[0])[1] < float("inf") else "tube"
        return detections

    if state.tube_ref is not None and state.tube_test is not None and state.miss_count < TRACKER_MAX_MISSES:
        state.miss_count += 1
        return detections

    state.tube_ref = None
    state.tube_test = None
    state.miss_count = 0
    return detections


@app.on_event("startup")
async def load_model():
    """Charge le modele au demarrage de l'API."""
    global model, autoencoder_model, siamese_model, siamese_config, inference_device

    if not MODEL_PATH.exists():
        print(f"[WARN] Modele non trouve : {MODEL_PATH}")
        print("       L'API demarre sans modele. Lancez d'abord l'entrainement.")
        return

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    inference_device = device

    print(f"[API] Chargement du modele sur {device}...")
    model = YOLO(str(MODEL_PATH))
    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, device=device, verbose=False)
    print("[API] Modele pret.")

    if AUTOENCODER_MODEL_PATH.exists():
        try:
            autoencoder_model = load_autoencoder_model(AUTOENCODER_MODEL_PATH, device=device)
            print(f"[API] Autoencoder charge depuis {AUTOENCODER_MODEL_PATH.name}.")
        except Exception as exc:
            autoencoder_model = None
            print(f"[WARN] Chargement autoencoder impossible: {exc}")
    else:
        print(f"[WARN] Autoencoder non trouve : {AUTOENCODER_MODEL_PATH}")

    if SIAMESE_MODEL_PATH.exists():
        try:
            siamese_model, siamese_config = load_siamese_model(SIAMESE_MODEL_PATH, device=device)
            print(f"[API] Siamese charge depuis {SIAMESE_MODEL_PATH.name}.")
        except Exception as exc:
            siamese_model = None
            siamese_config = None
            print(f"[WARN] Chargement siamese impossible: {exc}")
    else:
        print(f"[WARN] Siamese non trouve : {SIAMESE_MODEL_PATH}")


@app.get("/health")
async def health():
    """Verifie que l'API fonctionne."""
    return {
        "status": "ok" if model is not None else "no_model",
        "model_loaded": model is not None,
        "autoencoder_loaded": autoencoder_model is not None,
        "siamese_loaded": siamese_model is not None,
    }


@app.get("/model/info")
async def model_info():
    """Retourne les informations sur le modele charge."""
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Modele non charge"})

    return {
        "model": str(MODEL_PATH.name),
        "classes": model.names,
        "task": "segment",
        "autoencoder_model": AUTOENCODER_MODEL_PATH.name if autoencoder_model is not None else None,
        "siamese_model": SIAMESE_MODEL_PATH.name if siamese_model is not None else None,
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.45,
    analyze_thermal: bool = True,
    session_id: str | None = None,
    reset_tracker: bool = False,
):
    """
    Segmentation d'une image drone.

    Retourne les detections avec bounding boxes, masques, classes et scores.
    """
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Modele non charge"})

    img_np = await _read_upload_image(image)

    # Inference
    start = time.time()

    device = inference_device

    results = model.predict(img_np, device=device, conf=conf, iou=iou, imgsz=640, verbose=False)
    inference_time = (time.time() - start) * 1000

    result = results[0]

    detections = _format_detections(result, model.names, img_np)

    # Post-regle metier: si on detecte des tubes, forcer la convention
    # tube_ref = tube le plus haut, tube_test = tube le plus bas.
    # Cela evite les inversions de classes quand les deux tubes se ressemblent visuellement.
    detections = _apply_tube_tracking(detections, session_id=session_id, reset_tracker=reset_tracker)

    # Filtrer pour ne garder que les HCE (tubes recepteurs) si demande
    hce_detections = [d for d in detections if d["class_name"] in {"hce", "tube_ref", "tube_test", "tube"}]

    if analyze_thermal:
        thermal_anomaly, autoencoder_anomaly, siamese_anomaly, combined_anomaly = _analyze_with_existing_detections(
            img_np,
            detections,
            device,
        )
    else:
        thermal_anomaly = {
            "status": "disabled",
            "enabled": False,
            "reason": "thermal analysis not requested",
        }
        autoencoder_anomaly = {
            "status": "disabled",
            "enabled": autoencoder_model is not None,
            "reason": "autoencoder analysis not requested",
        }
        siamese_anomaly = {
            "status": "disabled",
            "enabled": siamese_model is not None,
            "reason": "siamese analysis not requested",
        }
        combined_anomaly = {
            "status": "disabled",
            "enabled": False,
            "reason": "combined analysis not requested",
        }

    return {
        "inference_time_ms": round(inference_time, 1),
        "image_size": {"width": img_np.shape[1], "height": img_np.shape[0]},
        "total_detections": len(detections),
        "hce_count": len(hce_detections),
        "thermal_anomaly": thermal_anomaly,
        "autoencoder_anomaly": autoencoder_anomaly,
        "siamese_anomaly": siamese_anomaly,
        "combined_anomaly": combined_anomaly,
        "detections": detections,
    }


@app.post("/analyze-anomaly")
async def analyze_anomaly(
    image: UploadFile = File(...),
    detections_json: str = Form(...),
):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Modele non charge"})

    try:
        detections = json.loads(detections_json)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "detections_json invalide"})

    if not isinstance(detections, list):
        return JSONResponse(status_code=400, content={"error": "detections_json doit etre une liste"})

    img_np = await _read_upload_image(image)
    start = time.time()
    thermal_anomaly, autoencoder_anomaly, siamese_anomaly, combined_anomaly = _analyze_with_existing_detections(
        img_np,
        detections,
        inference_device,
    )
    analysis_time_ms = (time.time() - start) * 1000

    hce_detections = [d for d in detections if d.get("class_name") in {"hce", "tube_ref", "tube_test", "tube"}]

    return {
        "inference_time_ms": round(analysis_time_ms, 1),
        "image_size": {"width": img_np.shape[1], "height": img_np.shape[0]},
        "total_detections": len(detections),
        "hce_count": len(hce_detections),
        "thermal_anomaly": thermal_anomaly,
        "autoencoder_anomaly": autoencoder_anomaly,
        "siamese_anomaly": siamese_anomaly,
        "combined_anomaly": combined_anomaly,
        "detections": detections,
    }


async def _read_upload_image(image: UploadFile) -> np.ndarray:
    contents = await image.read()
    img = Image.open(io.BytesIO(contents))
    return np.array(img)


def _format_detections(result, model_names: dict[int, str], img_np: np.ndarray) -> list[dict]:
    detections = []
    for i, box in enumerate(result.boxes):
        detection = {
            "class_id": int(box.cls[0]),
            "class_name": model_names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": box.xyxy[0].cpu().numpy().tolist(),
        }

        if result.masks is not None and i < len(result.masks):
            mask = result.masks.data[i].cpu().numpy()
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_uint8, (img_np.shape[1], img_np.shape[0]))
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                detection["mask_polygon"] = largest.reshape(-1, 2).tolist()
                detection["mask_area"] = int(cv2.contourArea(largest))

        detections.append(detection)
    return detections


def _analyze_with_existing_detections(
    img_np: np.ndarray,
    detections: list[dict],
    device: str,
) -> tuple[dict, dict, dict, dict]:
    try:
        thermal_anomaly = analyze_ref_test_thermal(img_np, detections)
    except Exception as exc:
        thermal_anomaly = {
            "status": "error",
            "enabled": True,
            "error": str(exc),
        }

    try:
        autoencoder_anomaly = analyze_ref_test_autoencoder(
            img_np,
            detections,
            model=autoencoder_model,
            device=device,
        )
    except Exception as exc:
        autoencoder_anomaly = {
            "status": "error",
            "enabled": autoencoder_model is not None,
            "error": str(exc),
        }

    try:
        combined_anomaly = build_combined_anomaly(thermal_anomaly, autoencoder_anomaly)
    except Exception as exc:
        combined_anomaly = {
            "status": "error",
            "enabled": True,
            "error": str(exc),
        }

    try:
        siamese_anomaly = analyze_ref_test_siamese(
            img_np,
            detections,
            model=siamese_model,
            device=device,
            config=siamese_config,
        )
    except Exception as exc:
        siamese_anomaly = {
            "status": "error",
            "enabled": siamese_model is not None,
            "error": str(exc),
        }

    try:
        combined_anomaly = build_combined_with_siamese(combined_anomaly, siamese_anomaly)
    except Exception as exc:
        combined_anomaly = {
            "status": "error",
            "enabled": True,
            "error": str(exc),
        }

    return thermal_anomaly, autoencoder_anomaly, siamese_anomaly, combined_anomaly
