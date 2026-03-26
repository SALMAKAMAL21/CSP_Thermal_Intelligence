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
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "experiments" / "aerialcsp_yolo11n_seg" / "weights" / "best.pt"

app = FastAPI(
    title="Green Energy Park - CSP Tube Detection API",
    description="Detection et segmentation des tubes recepteurs CSP par drone",
    version="1.0.0",
)

# Variable globale pour le modele
model = None


@app.on_event("startup")
async def load_model():
    """Charge le modele au demarrage de l'API."""
    global model

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

    print(f"[API] Chargement du modele sur {device}...")
    model = YOLO(str(MODEL_PATH))
    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, device=device, verbose=False)
    print("[API] Modele pret.")


@app.get("/health")
async def health():
    """Verifie que l'API fonctionne."""
    return {
        "status": "ok" if model is not None else "no_model",
        "model_loaded": model is not None,
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
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.45,
):
    """
    Segmentation d'une image drone.

    Retourne les detections avec bounding boxes, masques, classes et scores.
    """
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Modele non charge"})

    # Lire l'image
    contents = await image.read()
    img = Image.open(io.BytesIO(contents))
    img_np = np.array(img)

    # Inference
    start = time.time()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    results = model.predict(
        img_np,
        device=device,
        conf=conf,
        iou=iou,
        imgsz=640,
        verbose=False,
    )
    inference_time = (time.time() - start) * 1000

    result = results[0]

    # Formater les resultats
    detections = []
    for i, box in enumerate(result.boxes):
        detection = {
            "class_id": int(box.cls[0]),
            "class_name": model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": box.xyxy[0].cpu().numpy().tolist(),
        }

        # Ajouter le masque (en format RLE compact ou polygone)
        if result.masks is not None and i < len(result.masks):
            mask = result.masks.data[i].cpu().numpy()
            # Convertir en contours pour un format compact
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_uint8, (img_np.shape[1], img_np.shape[0]))
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Plus grand contour en liste de points
                largest = max(contours, key=cv2.contourArea)
                detection["mask_polygon"] = largest.reshape(-1, 2).tolist()
                detection["mask_area"] = int(cv2.contourArea(largest))

        detections.append(detection)

    # Filtrer pour ne garder que les HCE (tubes recepteurs) si demande
    hce_detections = [d for d in detections if d["class_name"] == "hce"]

    return {
        "inference_time_ms": round(inference_time, 1),
        "image_size": {"width": img_np.shape[1], "height": img_np.shape[0]},
        "total_detections": len(detections),
        "hce_count": len(hce_detections),
        "detections": detections,
    }
