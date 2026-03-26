"""
Script d'evaluation du modele entraine
========================================
Evalue le modele YOLOv11n-seg fine-tune sur le jeu de test AerialCSP
et genere un rapport detaille avec les metriques.

Usage:
    python src/training/evaluate.py
    python src/training/evaluate.py --model experiments/aerialcsp_yolo11n_seg/weights/best.pt
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "experiments" / "aerialcsp_yolo11n_seg" / "weights" / "best.pt"


def evaluate(model_path):
    print("=" * 60)
    print("  EVALUATION YOLOv11n-seg")
    print("=" * 60)

    if not Path(model_path).exists():
        print(f"[ERREUR] Modele introuvable : {model_path}")
        print("         Lance d'abord l'entrainement : python src/training/train.py")
        return

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[Device] {device}")

    # Charger le modele
    print(f"[Modele] {model_path}")
    model = YOLO(str(model_path))

    # Evaluation sur le jeu de test
    print("\n[Evaluation] Sur le jeu de test...")
    metrics = model.val(
        data=str(PROJECT_ROOT / "configs" / "aerialcsp_dataset.yaml"),
        split="test",
        device=device,
        plots=True,
        save_json=True,
        project=str(PROJECT_ROOT / "experiments"),
        name="evaluation_results",
        exist_ok=True,
    )

    # Afficher les resultats
    print("\n" + "=" * 60)
    print("  RESULTATS")
    print("=" * 60)

    # Detection
    print("\n  --- Detection (Bounding Boxes) ---")
    print(f"  mAP50       : {metrics.box.map50:.4f}")
    print(f"  mAP50-95    : {metrics.box.map:.4f}")

    # Segmentation
    if hasattr(metrics, 'seg'):
        print("\n  --- Segmentation (Masques) ---")
        print(f"  mAP50       : {metrics.seg.map50:.4f}")
        print(f"  mAP50-95    : {metrics.seg.map:.4f}")

    # Par classe
    print("\n  --- Par classe (mAP50) ---")
    class_names = model.names
    if metrics.box.ap50 is not None:
        for i, ap in enumerate(metrics.box.ap50):
            name = class_names.get(i, f"classe_{i}")
            seg_ap = metrics.seg.ap50[i] if hasattr(metrics, 'seg') else "N/A"
            print(f"  {name:<25} det: {ap:.4f}   seg: {seg_ap}")

    # Vitesse
    print(f"\n  --- Vitesse ---")
    print(f"  Preprocess  : {metrics.speed['preprocess']:.1f}ms")
    print(f"  Inference   : {metrics.speed['inference']:.1f}ms")
    print(f"  Postprocess : {metrics.speed['postprocess']:.1f}ms")

    print("\n" + "=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation du modele")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Chemin vers best.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model)
