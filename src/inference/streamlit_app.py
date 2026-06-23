"""Streamlit app pour tester le modele YOLOv11s-seg fine-tune."""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "src" / "models" / "hce_thermal_best.pt"


def detect_device() -> str:
    """Retourne le meilleur device disponible."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    """Charge le modele Ultralytics et le met en cache."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Modele introuvable : {path}")
    return YOLO(str(path))


def format_detections(result) -> pd.DataFrame:
    """Construit un DataFrame a partir des detections YOLO."""
    rows: List[dict] = []
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        row = {
            "class_id": cls_id,
            "class_name": result.names[cls_id],
            "confidence": float(box.conf[0]),
        }
        bbox = box.xyxy[0].cpu().numpy().tolist()
        row.update({"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]})
        if result.masks is not None and i < len(result.masks):
            mask = result.masks.data[i].cpu().numpy()
            row["mask_area_px"] = float(mask.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Green Energy Park - Segmentation CSP", layout="wide")
    st.title("Segmentation des composants CSP")
    st.write(
        "Telecharge n'importe quelle image drone de centrale CSP pour visualiser les"
        " detections (miroirs, tubes, structures)."
    )

    sidebar = st.sidebar
    sidebar.header("Configuration")
    model_path = sidebar.text_input("Chemin du modele", value=str(DEFAULT_MODEL))
    confidence = sidebar.slider("Seuil de confiance", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    iou = sidebar.slider("Seuil IoU", min_value=0.3, max_value=0.95, value=0.45, step=0.05)
    imgsz = sidebar.selectbox("Resolution inference", options=[512, 640, 960], index=1)

    device = detect_device()
    sidebar.success(f"Device : {device.upper()}")

    try:
        model = load_model(model_path)
    except FileNotFoundError as exc:  # pragma: no cover - feedback utilisateur
        st.error(str(exc))
        st.stop()

    uploaded = st.file_uploader("Image CSP (JPG/PNG)", type=["jpg", "jpeg", "png"]) 

    if uploaded is None:
        st.info("Charge une image pour lancer l'inference.")
        return

    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(image, caption="Image originale", use_container_width=True)

    if st.button("Lancer l'inference", type="primary"):
        with st.spinner("Inference en cours..."):
            np_img = np.array(image)
            results = model.predict(
                np_img,
                imgsz=imgsz,
                conf=confidence,
                iou=iou,
                device=device,
                verbose=False,
            )
            result = results[0]

        annotated = result.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.subheader("Resultats")
        st.image(annotated, caption="Detections YOLOv11s-seg", use_container_width=True)

        speed = result.speed or {}
        st.caption(
            " | ".join(
                [
                    f"preprocess: {speed.get('preprocess', 0):.1f} ms",
                    f"inference: {speed.get('inference', 0):.1f} ms",
                    f"postprocess: {speed.get('postprocess', 0):.1f} ms",
                ]
            )
        )

        df = format_detections(result)
        if df.empty:
            st.warning("Aucune detection. Ajuste les seuils ou verifie l'image.")
        else:
            st.dataframe(df, use_container_width=True)
            counts = df["class_name"].value_counts().to_dict()
            st.write("**Comptage par classe :**", counts)


if __name__ == "__main__":
    main()
