"""
API FastAPI pour l'inference YOLOv11n-seg
==========================================
Recoit des images du drone et retourne les segmentations des composants CSP.

Usage:
uvicorn src.inference.api:app --host 0.0.0.0 --port 8002

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
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

from .fusion_anomaly import FusionPredictor, temperature_delta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AUTOENCODER_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "best_autoencoder.pt"
SIAMESE_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "best_siamese.pt"


def resolve_model_path() -> Path:
    """Resout le chemin du modele avec priorite a MODEL_PATH puis yolo_seg.pt."""
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path)

    candidates = [
        # PROJECT_ROOT / "ml" / "best_model.pt",
        # PROJECT_ROOT / "ml" / "yolo_seg.pt",
        PROJECT_ROOT / "src" / "models" / "opt_best.pt",
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
fusion_predictor = None
fusion_error = None
FUSION_MODEL_PATH = Path(os.getenv("FUSION_MODEL_PATH", str(PROJECT_ROOT / "src/models/best_fusion_model.pt")))
FUSION_CONFIG_PATH = Path(os.getenv("FUSION_CONFIG_PATH", str(PROJECT_ROOT / "configs/fusion.json")))


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
    global model, inference_device, fusion_predictor, fusion_error

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

    # Les anciens checkpoints restent sur disque, sans chargement ni vote.
    try:
        fusion_predictor = FusionPredictor(FUSION_MODEL_PATH, FUSION_CONFIG_PATH, device)
        fusion_error = None
        print("[API] Fusion ordinale chargée.")
    except Exception as exc:
        fusion_predictor = None
        fusion_error = str(exc)
        print(f"[WARN] Fusion indisponible : {exc}")


@app.get("/health")
async def health():
    """Verifie que l'API fonctionne."""
    return {
        "status": "ok" if model is not None else "no_model",
        "model_loaded": model is not None,
        "fusion_loaded": fusion_predictor is not None,
        "fusion_error": fusion_error,
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
        "fusion_model": FUSION_MODEL_PATH.name if fusion_predictor is not None else None,
        "autoencoder_model": AUTOENCODER_MODEL_PATH.name if autoencoder_model is not None else None,
        "siamese_model": SIAMESE_MODEL_PATH.name if siamese_model is not None else None,
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    conf: float = 0.5,
    iou: float = 0.45,
    analyze_thermal: bool = True,
    session_id: str | None = None,
    reset_tracker: bool = False,
    t_ref: float | None = Form(None),
    t_test: float | None = Form(None),
):
    """
    Segmentation d'une image drone.

    Retourne les detections avec bounding boxes, masques, classes et scores.
    """
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Modele non charge"})

    img_np = await _read_upload_image(image)

    if analyze_thermal:
        _require_fusion(t_ref, t_test)

    # Inference
    start = time.time()

    device = inference_device

    # Ultralytics attend BGR pour une entrée numpy ; les ROI de fusion restent RGB.
    results = model.predict(img_np[:, :, ::-1].copy(), device=device, conf=conf, iou=iou, imgsz=640, verbose=False)
    inference_time = (time.time() - start) * 1000

    result = results[0]

    detections = _format_detections(result, model.names, img_np)

    # Conserver les classes YOLO, comme extract_rois.py à l'entraînement.
    # Ne pas réattribuer les rôles selon la position verticale.

    # Filtrer pour ne garder que les HCE (tubes recepteurs) si demande
    hce_detections = [d for d in detections if d["class_name"] in {"hce", "tube_ref", "tube_test", "tube"}]

    fusion = fusion_predictor.predict(img_np, detections, t_ref, t_test) if analyze_thermal else {"status": "disabled"}

    return {
        "inference_time_ms": round(inference_time, 1),
        "image_size": {"width": img_np.shape[1], "height": img_np.shape[0]},
        "total_detections": len(detections),
        "hce_count": len(hce_detections),
        "detections": detections,
        "fusion": fusion,
    }


@app.post("/analyze-anomaly")
async def analyze_anomaly(
    image: UploadFile = File(...),
    detections_json: str = Form(...),
    t_ref: float = Form(...),
    t_test: float = Form(...),
):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Modele non charge"})

    try:
        detections = json.loads(detections_json)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "detections_json invalide"})

    if not isinstance(detections, list) or not all(isinstance(d, dict) for d in detections):
        return JSONResponse(status_code=400, content={"error": "detections_json doit etre une liste"})

    img_np = await _read_upload_image(image)
    start = time.time()
    _require_fusion(t_ref, t_test)
    for detection in detections:
        try:
            confidence = float(detection.get("confidence", 0))
            if not np.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError()
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail="Confiance de détection invalide")
    fusion = fusion_predictor.predict(img_np, detections, t_ref, t_test)
    analysis_time_ms = (time.time() - start) * 1000

    hce_detections = [d for d in detections if d.get("class_name") in {"hce", "tube_ref", "tube_test", "tube"}]

    return {
        "inference_time_ms": round(analysis_time_ms, 1),
        "image_size": {"width": img_np.shape[1], "height": img_np.shape[0]},
        "total_detections": len(detections),
        "hce_count": len(hce_detections),
        "detections": detections,
        "fusion": fusion,
    }


async def _read_upload_image(image: UploadFile) -> np.ndarray:
    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Image illisible")
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


def _require_fusion(t_ref, t_test):
    try:
        temperature_delta(t_ref, t_test)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if fusion_predictor is None:
        raise HTTPException(status_code=503, detail=f"Fusion indisponible : {fusion_error or 'checkpoint absent'}")
