"""
Script d'entrainement YOLOv11n-seg sur le dataset AerialCSP
=============================================================
Fine-tuning du modele YOLOv11n-seg (pretraine COCO) pour la segmentation
des composants CSP (tubes recepteurs HCE, miroirs, structures).

Usage:
    # Entrainement avec config par defaut
    python src/training/train.py

    # Entrainement avec parametres custom
    python src/training/train.py --epochs 100 --batch 8 --imgsz 640

    # Reprendre un entrainement interrompu
    python src/training/train.py --resume experiments/aerialcsp_yolo11n_seg/weights/last.pt
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_device():
    """Detecte le meilleur device disponible."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"[GPU] CUDA : {gpu_name} ({vram:.1f} GB VRAM)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[GPU] Apple MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("[CPU] Aucun GPU detecte")
    return device


def load_config():
    """Charge la configuration d'entrainement depuis le YAML."""
    config_path = PROJECT_ROOT / "configs" / "train_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def check_dataset(config):
    """Verifie que le dataset existe."""
    data_yaml = PROJECT_ROOT / config["data"]
    if not data_yaml.exists():
        print(f"[ERREUR] Fichier dataset introuvable : {data_yaml}")
        print("         As-tu telecharge le dataset AerialCSP ?")
        print("         Place-le dans data/aerialcsp/ avec la structure :")
        print("           data/aerialcsp/train/images/")
        print("           data/aerialcsp/train/labels/")
        print("           data/aerialcsp/val/images/")
        print("           data/aerialcsp/val/labels/")
        sys.exit(1)

    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)

    dataset_path = (data_yaml.parent / data_config["path"]).resolve()
    train_path = dataset_path / data_config["train"]

    if not train_path.exists():
        print(f"[ERREUR] Dossier d'entrainement introuvable : {train_path}")
        print("         Telecharge le dataset AerialCSP et place-le dans data/aerialcsp/")
        sys.exit(1)

    # Compter les images
    n_train = len(list((dataset_path / data_config["train"]).glob("*")))
    n_val = len(list((dataset_path / data_config["val"]).glob("*")))
    print(f"[Dataset] {n_train} images train, {n_val} images val")
    print(f"[Dataset] Classes : {data_config['names']}")

    return str(data_yaml)


def adjust_batch_size(config, device):
    """Ajuste le batch size selon le GPU disponible."""
    batch = config.get("batch", 16)

    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        if vram < 6:
            batch = 4
        elif vram < 8:
            batch = 8
        elif vram < 12:
            batch = 16
        else:
            batch = 32
        print(f"[Config] Batch size ajuste a {batch} (VRAM: {vram:.1f} GB)")
    elif device == "mps":
        batch = 8
        print(f"[Config] Batch size ajuste a {batch} (MPS)")

    return batch


def train(args):
    """Lance l'entrainement."""
    print("=" * 60)
    print("  ENTRAINEMENT YOLOv11n-seg sur AerialCSP")
    print("=" * 60)

    # Device
    device = get_device()

    # Configuration
    config = load_config()

    # Override avec les arguments CLI
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch:
        config["batch"] = args.batch
    if args.imgsz:
        config["imgsz"] = args.imgsz

    # Verifier le dataset
    data_yaml = check_dataset(config)

    # Ajuster le batch size
    if not args.batch:
        config["batch"] = adjust_batch_size(config, device)

    # Resume ou nouveau
    if args.resume:
        print(f"\n[Resume] Reprise depuis : {args.resume}")
        model = YOLO(args.resume)
    else:
        model_path = config.get("model", "yolo11n-seg.pt")
        print(f"\n[Modele] Chargement de {model_path} (pretraine COCO)")
        model = YOLO(model_path)

    # Afficher la config
    print(f"\n[Config] Entrainement :")
    print(f"   Epochs     : {config['epochs']}")
    print(f"   Batch size : {config['batch']}")
    print(f"   Image size : {config['imgsz']}")
    print(f"   Device     : {device}")
    print(f"   Patience   : {config.get('patience', 50)}")
    print(f"   Projet     : {config.get('project', 'experiments')}/{config.get('name', 'train')}")

    # Lancer l'entrainement
    print("\n" + "-" * 60)
    print("  Debut de l'entrainement...")
    print("-" * 60 + "\n")

    start_time = time.time()

    results = model.train(
        data=data_yaml,
        epochs=config["epochs"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        device=device,
        patience=config.get("patience", 50),
        optimizer=config.get("optimizer", "auto"),
        lr0=config.get("lr0", 0.01),
        lrf=config.get("lrf", 0.01),
        momentum=config.get("momentum", 0.937),
        weight_decay=config.get("weight_decay", 0.0005),
        warmup_epochs=config.get("warmup_epochs", 3.0),
        warmup_momentum=config.get("warmup_momentum", 0.8),
        hsv_h=config.get("hsv_h", 0.015),
        hsv_s=config.get("hsv_s", 0.7),
        hsv_v=config.get("hsv_v", 0.4),
        degrees=config.get("degrees", 0.0),
        translate=config.get("translate", 0.1),
        scale=config.get("scale", 0.5),
        fliplr=config.get("fliplr", 0.5),
        flipud=config.get("flipud", 0.0),
        mosaic=config.get("mosaic", 1.0),
        mixup=config.get("mixup", 0.0),
        conf=config.get("conf", 0.25),
        iou=config.get("iou", 0.7),
        project=str(PROJECT_ROOT / config.get("project", "experiments")),
        name=config.get("name", "aerialcsp_yolo11n_seg"),
        save_period=config.get("save_period", 25),
        exist_ok=True,
        plots=True,
        save=True,
    )

    train_time = time.time() - start_time

    # Resume
    print("\n" + "=" * 60)
    print("  ENTRAINEMENT TERMINE")
    print("=" * 60)
    print(f"  Duree totale : {train_time / 60:.1f} minutes")
    print(f"  Meilleur modele : {config.get('project', 'experiments')}/{config.get('name', 'train')}/weights/best.pt")
    print(f"  Dernier modele  : {config.get('project', 'experiments')}/{config.get('name', 'train')}/weights/last.pt")
    print("=" * 60)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Entrainement YOLOv11n-seg sur AerialCSP")
    parser.add_argument("--epochs", type=int, default=None, help="Nombre d'epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Taille des images")
    parser.add_argument("--resume", type=str, default=None, help="Chemin vers last.pt pour reprendre")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
