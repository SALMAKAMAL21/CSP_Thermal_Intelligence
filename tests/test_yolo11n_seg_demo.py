"""
Premier test : YOLOv11n-seg sur une image de demo
===================================================
Ce script teste le modele YOLOv11n-seg pretraine (COCO) pour verifier
que le pipeline de segmentation fonctionne correctement.

Le modele pretraine COCO ne connait pas les tubes CSP, mais ce test permet de :
1. Verifier que l'installation fonctionne (PyTorch, Ultralytics, GPU/MPS)
2. Comprendre le format de sortie de YOLO (boxes, masks, classes, scores)
3. Visualiser les resultats de segmentation
4. Mesurer le temps d'inference sur notre machine

Prochaine etape : fine-tuner sur le dataset AerialCSP pour detecter les tubes.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "test_yolo11n_seg"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    """Detecte le meilleur device disponible."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[GPU] CUDA detecte : {gpu_name}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[GPU] Apple MPS detecte (Apple Silicon)")
    else:
        device = "cpu"
        print("[CPU] Aucun GPU detecte, utilisation du CPU")
    return device


def create_test_image():
    """Cree une image de test synthetique simulant des tubes CSP vus du ciel."""
    print("\n[1/4] Creation d'une image de test synthetique (simulation tubes CSP)...")

    img = np.zeros((640, 640, 3), dtype=np.uint8)

    # Fond : sol desert (sable)
    img[:] = (180, 200, 210)  # BGR - couleur sable

    # Simuler des rangees de miroirs paraboliques (rectangles gris metallique)
    for y_start in [80, 220, 360, 500]:
        # Miroir (gris clair, reflet metallique)
        cv2.rectangle(img, (30, y_start), (610, y_start + 80), (200, 200, 210), -1)
        # Tube recepteur au centre du miroir (ligne sombre)
        cv2.line(img, (30, y_start + 40), (610, y_start + 40), (40, 40, 60), 4)
        # Reflets sur le miroir
        cv2.line(img, (30, y_start + 20), (610, y_start + 20), (220, 220, 230), 1)
        cv2.line(img, (30, y_start + 60), (610, y_start + 60), (220, 220, 230), 1)

    # Ajouter du bruit pour realisme
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    test_img_path = OUTPUT_DIR / "test_synthetic_csp.jpg"
    cv2.imwrite(str(test_img_path), img)
    print(f"   Image sauvegardee : {test_img_path}")
    print(f"   Dimensions : {img.shape[1]}x{img.shape[0]} pixels")
    return str(test_img_path), img


def load_model(device):
    """Charge le modele YOLOv11n-seg pretraine."""
    print("\n[2/4] Chargement du modele YOLOv11n-seg (pretraine COCO)...")
    print("   (Premier lancement = telechargement automatique du modele)")

    start = time.time()
    model = YOLO("yolo11n-seg.pt")
    load_time = time.time() - start

    print(f"   Modele charge en {load_time:.2f}s")
    print(f"   Nombre de classes COCO : {len(model.names)}")
    print(f"   Taille du modele : yolo11n-seg (nano)")
    return model


def run_inference(model, image_path, device):
    """Execute l'inference et analyse les resultats."""
    print(f"\n[3/4] Inference sur l'image de test (device: {device})...")

    # Inference
    start = time.time()
    results = model.predict(
        source=image_path,
        device=device,
        conf=0.25,        # seuil de confiance minimum
        iou=0.45,         # seuil IoU pour NMS
        imgsz=640,        # taille d'entree du modele
        save=False,
        verbose=False,
    )
    inference_time = time.time() - start

    result = results[0]

    print(f"   Temps d'inference : {inference_time * 1000:.1f}ms")
    print(f"   Nombre d'objets detectes : {len(result.boxes)}")

    # Details des detections
    if len(result.boxes) > 0:
        print("\n   Detections :")
        print(f"   {'Classe':<20} {'Confiance':<12} {'Bbox (x1,y1,x2,y2)':<35} {'Masque'}")
        print(f"   {'-'*80}")

        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            has_mask = result.masks is not None and i < len(result.masks)
            print(f"   {cls_name:<20} {conf:<12.3f} {str(bbox):<35} {'Oui' if has_mask else 'Non'}")
    else:
        print("\n   Aucun objet detecte (normal : le modele COCO ne connait pas les tubes CSP)")
        print("   -> C'est attendu ! Le but est de verifier que le pipeline fonctionne.")
        print("   -> Apres fine-tuning sur AerialCSP, le modele detectera les tubes.")

    return result, inference_time


def save_results(result, original_img, model, inference_time):
    """Sauvegarde les resultats visuels."""
    print("\n[4/4] Sauvegarde des resultats...")

    # Image avec annotations YOLO
    annotated = result.plot()
    annotated_path = OUTPUT_DIR / "result_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    print(f"   Image annotee : {annotated_path}")

    # Sauvegarder les masques separement si disponibles
    if result.masks is not None and len(result.masks) > 0:
        masks_combined = np.zeros(original_img.shape[:2], dtype=np.uint8)
        for mask in result.masks.data:
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (original_img.shape[1], original_img.shape[0]))
            masks_combined = np.maximum(masks_combined, (mask_resized * 255).astype(np.uint8))

        mask_path = OUTPUT_DIR / "result_masks.jpg"
        cv2.imwrite(str(mask_path), masks_combined)
        print(f"   Masques : {mask_path}")

    # Rapport texte
    report_path = OUTPUT_DIR / "report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("RAPPORT DE TEST - YOLOv11n-seg\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date : 11 Mars 2026\n")
        f.write(f"Modele : yolo11n-seg (pretraine COCO)\n")
        f.write(f"Device : {get_device()}\n")
        f.write(f"Image : test synthetique CSP (640x640)\n")
        f.write(f"Temps d'inference : {inference_time * 1000:.1f}ms\n")
        f.write(f"Objets detectes : {len(result.boxes)}\n\n")

        f.write("CLASSES COCO DISPONIBLES :\n")
        for idx, name in model.names.items():
            f.write(f"  {idx:3d} : {name}\n")

        f.write(f"\nNOTE : Le modele COCO ne connait pas les tubes CSP.\n")
        f.write(f"Le fine-tuning sur AerialCSP est necessaire pour la detection de tubes.\n")

    print(f"   Rapport : {report_path}")


def run_benchmark(model, image_path, device, n_runs=20):
    """Benchmark de performance : temps d'inference moyen."""
    print(f"\n[BONUS] Benchmark de performance ({n_runs} inferences)...")

    times = []
    for i in range(n_runs):
        start = time.time()
        model.predict(source=image_path, device=device, verbose=False, save=False)
        times.append(time.time() - start)

    times_ms = [t * 1000 for t in times]
    print(f"   Temps moyen  : {np.mean(times_ms):.1f}ms")
    print(f"   Temps median : {np.median(times_ms):.1f}ms")
    print(f"   Min / Max    : {np.min(times_ms):.1f}ms / {np.max(times_ms):.1f}ms")
    print(f"   FPS estimes  : {1000 / np.mean(times_ms):.1f}")

    return times_ms


def main():
    print("=" * 60)
    print("  TEST YOLOv11n-seg - Green Energy Park PFE")
    print("=" * 60)

    # Detecter le device
    device = get_device()

    # Creer image de test
    image_path, original_img = create_test_image()

    # Charger le modele
    model = load_model(device)

    # Inference
    result, inference_time = run_inference(model, image_path, device)

    # Sauvegarder
    save_results(result, original_img, model, inference_time)

    # Benchmark
    times = run_benchmark(model, image_path, device)

    # Resume
    print("\n" + "=" * 60)
    print("  RESUME")
    print("=" * 60)
    print(f"  Pipeline fonctionnel     : OUI")
    print(f"  Device utilise           : {device}")
    print(f"  Temps inference moyen    : {np.mean(times):.1f}ms")
    print(f"  FPS estimes              : {1000 / np.mean(times):.1f}")
    print(f"  Resultats dans           : {OUTPUT_DIR}")
    print(f"\n  PROCHAINE ETAPE :")
    print(f"  -> Obtenir le dataset AerialCSP")
    print(f"  -> Fine-tuner yolo11n-seg sur les images CSP")
    print(f"  -> Tester sur de vraies images de Green Energy Park")
    print("=" * 60)


if __name__ == "__main__":
    main()
