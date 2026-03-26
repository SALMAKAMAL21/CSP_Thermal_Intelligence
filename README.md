# Projet de Fin d'Etudes - Green Energy Park

## Informations Generales

| Champ | Detail |
|-------|--------|
| **Stagiaire** | Cedrix |
| **Etablissement** | ISMAGI |
| **Lieu de stage** | Green Energy Park |
| **Date de debut** | 11 Mars 2026 |
| **Sujet** | Detection d'anomalies sur les tubes recepteurs de miroirs solaires par Computer Vision |

---

## Description du Projet

Conception et developpement d'un modele de Computer Vision capable de detecter les anomalies sur les tubes de reception des miroirs solaires (CSP - Concentrated Solar Power). Le systeme integre egalement des donnees contextuelles telles que :

- **Temperature** des tubes et de l'environnement
- **Conditions atmospheriques** (humidite, vent, ensoleillement, etc.)
- **Autres parametres operationnels** pertinents

### Objectifs

1. Collecter et preparer un dataset d'images des tubes recepteurs (normal vs anomalies)
2. Identifier et classifier les types d'anomalies (fissures, depots, deformations, degradation du coating, etc.)
3. Developper un modele de detection d'anomalies performant
4. Integrer les donnees multi-modales (vision + capteurs) pour ameliorer la precision
5. Deployer une solution utilisable sur le terrain

---

## Architecture Technique Envisagee

```
Donnees d'entree
|
|-- Images des tubes recepteurs (Camera / Drone)
|-- Donnees de temperature (Capteurs thermiques)
|-- Donnees atmospheriques (Station meteo)
|
v
Preprocessing
|
|-- Augmentation d'images
|-- Normalisation des donnees capteurs
|-- Fusion des donnees multi-modales
|
v
Modele de Detection
|
|-- Backbone CNN (ResNet / EfficientNet / YOLO)
|-- Module de fusion multi-modale
|-- Tete de detection / classification
|
v
Sortie
|
|-- Localisation de l'anomalie
|-- Type d'anomalie
|-- Score de confiance
|-- Recommandation de maintenance
```

---

## Journal de Bord

### Semaine 1 (11 Mars - 15 Mars 2026)

#### Jour 1 - 11 Mars 2026
- Prise de contact avec Green Energy Park
- Presentation du sujet de stage
- Definition de la premiere etape : **segmentation des tubes recepteurs** pour usage drone
- Recherche de l'etat de l'art sur la detection/segmentation de tubes CSP

**Ressources identifiees :**
- **Dataset AerialCSP** : 18 058 images labellisees de centrales CSP par drone (GitHub: mpcutino/aerialcsp, arXiv: 2508.00440)
  - Benchmark YOLOv11-seg : mAP50 detection = 96.2%, segmentation = 73.3%
- **Volateq** (DLR spinoff) : solution commerciale de reference, drone DJI Mavic 3, inspecte 40 000 tubes/heure
- **Detection tubes casses** sur 7 centrales CSP reelles (Springer, 2023)
- **NREL Distant Observer** : caracterisation optique par drone

**Decisions prises :**
- Modele retenu : **YOLOv11n-seg** (nano, 2.6M params) - leger pour usage drone
- Pipeline : Drone RGB -> YOLOv11n-seg (segmentation tube) -> ROI -> Classification anomalies + donnees capteurs
- Framework : PyTorch + Ultralytics
- Option SAM2 en complement pour masques plus precis si necessaire
- Nomenclature YOLO clarifiee : `yolo11n-seg` = version 11 + nano (taille) + segmentation (tache)
- Possibilite de scaler vers `s`, `m`, `l` si precision insuffisante

**Environnement technique :**
- Python 3.12.0, environnement virtuel cree (`venv/`)
- Installation des dependances via `requirements.txt`
- Bibliotheques principales : PyTorch 2.10, Ultralytics 8.4.21, OpenCV 4.13, MLflow, FastAPI
- Apple MPS (GPU Apple Silicon) detecte et fonctionnel

**Clarification modele pretraine vs fine-tune :**
- `yolo11n-seg.pt` telecharge = pretraine sur **COCO** (80 classes generiques) -> ne connait PAS les tubes CSP
- Le paper AerialCSP a **entraine** YOLOv11 sur leur dataset -> obtient mAP50 96.2%
- **Pas de poids pretrained CSP disponibles** sur le repo AerialCSP -> on doit entrainer nous-memes
- Dataset AerialCSP : **5 classes** (mirror, torque_tube, supporting_structure, hce_support, **hce**)

**Test YOLOv11n-seg (COCO) :**
- Pipeline fonctionnel : inference en **18.6ms** (~53.7 FPS) sur Apple MPS
- 0 detection sur image CSP synthetique (attendu : COCO ne connait pas les tubes)
- Resultats dans `experiments/test_yolo11n_seg/`

**Scripts developpes :**
- `src/training/train.py` : entrainement avec config YAML, auto-detection GPU, resume possible
- `src/training/evaluate.py` : evaluation sur jeu de test avec metriques detaillees
- `src/inference/api.py` : API FastAPI pour inference sur le serveur NVIDIA
- `configs/train_config.yaml` : hyperparametres d'entrainement (200 epochs, meme que benchmark)
- `configs/aerialcsp_dataset.yaml` : configuration du dataset (5 classes CSP)

**Architecture de deploiement retenue : Option A (inference sur serveur)**
- Drone capture les images -> envoie au serveur NVIDIA via WiFi/4G
- Serveur fait l'inference via API FastAPI -> retourne les resultats JSON
- Dashboard Streamlit pour visualisation par les operateurs
- Developpement sur Mac local, entrainement complet sur serveur NVIDIA

---

## Stack Technique

| Composant | Technologie |
|-----------|-------------|
| Langage | Python 3.12 |
| Deep Learning | PyTorch >= 2.2 |
| Computer Vision | OpenCV, Albumentations |
| Detection / Segmentation | YOLOv11-seg (Ultralytics), SAM2 |
| Donnees | Pandas, NumPy, scikit-learn |
| Visualisation | Matplotlib, Seaborn, TensorBoard |
| Suivi d'experiences | MLflow |
| Annotation | LabelMe |
| Deploiement | FastAPI, Uvicorn, Streamlit |

---

## Structure du Projet

```
Green Energy Park/
|-- README.md                           # Ce fichier (journal de bord)
|-- requirements.txt                    # Dependances Python
|-- .gitignore                          # Fichiers ignores par git
|-- yolo11n-seg.pt                      # Poids pretrained COCO (telecharge auto)
|
|-- configs/
|   |-- train_config.yaml               # Hyperparametres d'entrainement
|   |-- aerialcsp_dataset.yaml          # Configuration du dataset (5 classes)
|
|-- src/
|   |-- training/
|   |   |-- train.py                    # Script d'entrainement (fine-tuning)
|   |   |-- evaluate.py                 # Evaluation sur jeu de test
|   |-- inference/
|   |   |-- api.py                      # API FastAPI (serveur NVIDIA)
|   |-- data/                           # Scripts de chargement (a venir)
|   |-- models/                         # Architectures custom (a venir)
|   |-- utils/                          # Fonctions utilitaires (a venir)
|
|-- tests/
|   |-- test_yolo11n_seg_demo.py        # Premier test pipeline (fait)
|
|-- data/
|   |-- raw/                            # Images brutes
|   |-- processed/                      # Images preprocessees
|   |-- annotations/                    # Labels et annotations
|   |-- sensors/                        # Donnees capteurs (temperature, meteo)
|   |-- aerialcsp/                      # Dataset AerialCSP (a telecharger)
|
|-- notebooks/                          # Jupyter notebooks d'exploration
|-- experiments/                        # Resultats d'entrainement et tests
|-- reports/                            # Rapport de stage et presentations
```

### Commandes principales

```bash
# Activer l'environnement
source venv/bin/activate

# Test du pipeline (deja fait)
python tests/test_yolo11n_seg_demo.py

# Entrainement (quand le dataset sera pret)
python src/training/train.py
python src/training/train.py --epochs 100 --batch 8

# Reprendre un entrainement interrompu
python src/training/train.py --resume experiments/aerialcsp_yolo11n_seg/weights/last.pt

# Evaluation
python src/training/evaluate.py

# Lancer l'API (sur le serveur NVIDIA)
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

---

## Types d'Anomalies a Detecter

| Anomalie | Description | Priorite |
|----------|-------------|----------|
| Fissures | Fissures visibles sur le tube en verre | Haute |
| Degradation du coating | Deterioration du revetement absorbant | Haute |
| Depots / Salissures | Accumulation de poussiere ou debris | Moyenne |
| Deformation | Deformation mecanique du tube | Haute |
| Bris de verre | Casse de l'enveloppe en verre | Haute |
| Fuite de vide | Perte de vide dans l'espace annulaire | Moyenne |
| Corrosion | Signes de corrosion sur les composants metalliques | Moyenne |

---

## Metriques d'Evaluation

- **mAP (mean Average Precision)** - pour la detection
- **Precision / Recall / F1-Score** - pour la classification
- **IoU (Intersection over Union)** - pour la localisation
- **Matrice de confusion** - analyse des erreurs
- **Temps d'inference** - performance en temps reel

---

## References et Ressources

### Datasets
- [x] **AerialCSP** - 18 058 images synthetiques de centrales CSP par drone (GitHub: mpcutino/aerialcsp)
  - Paper : "Reducing the gap between general purpose data and aerial images in CSP plants" (arXiv: 2508.00440)
  - Acces : formulaire Google Forms sur le repo GitHub
- [ ] Dataset de tubes casses sur 7 centrales CSP (Springer, 2023)
- [ ] Documentation technique des tubes recepteurs CSP
- [ ] Documentation Green Energy Park

### Articles et Papers
- [x] AerialCSP + YOLOv11 benchmark (arXiv: 2508.00440)
- [x] "Detecting broken receiver tubes in CSP plants using intelligent sampling and dual loss" (Springer, 2023)
- [x] NREL "Combined CV and DL Approach for Drone-Based Optical Characterization" (ASME, 2023)
- [ ] HOTSPOT-YOLO pour detection thermique (arXiv: 2508.18912)

### Outils et Modeles
- [x] Ultralytics YOLOv11-seg (ultralytics.com)
- [x] SAM2 - Segment Anything Model 2 (GitHub: facebookresearch/sam2)
- [x] Volateq - Solution commerciale drone CSP (volateq.de)

### Solutions Existantes (Etat de l'Art)
- **Volateq (DLR)** : Drones DJI Mavic 3, imagerie thermique+RGB, 40 000 tubes/heure
- **NREL Distant Observer** : Mesure de pente de surface et offset de tube via reflexion miroir
- **UGV-UAV Collaborative** : YOLOv5 combinant robots sol + aeriens (Springer, 2025)

---

## Notes pour le Rapport de Stage

> Cette section sera enrichie au fur et a mesure du stage pour faciliter la redaction du rapport final.

### Plan du Rapport (provisoire)

1. Introduction et contexte
2. Presentation de Green Energy Park
3. Etat de l'art
   - Technologies CSP
   - Computer Vision pour l'inspection industrielle
   - Approches multi-modales
4. Methodologie
5. Collecte et preparation des donnees
6. Conception du modele
7. Experimentations et resultats
8. Deploiement
9. Conclusion et perspectives

---

## Architecture de Deploiement (Option A - Inference sur serveur)

```
                        TERRAIN                                    SERVEUR LOCAL
          ┌──────────────────────────┐              ┌──────────────────────────────────┐
          │                          │              │                                  │
          │   Drone (camera RGB)     │    WiFi/4G   │   Serveur NVIDIA (GPU CUDA)      │
          │        |                 │ ──────────>  │        |                          │
          │   Capture images/video   │   images     │   API FastAPI (:8000)             │
          │                          │              │        |                          │
          └──────────────────────────┘              │   YOLOv11n-seg (inference)        │
                                                    │        |                          │
                                                    │   Resultats (JSON)               │
                                                    │        |                          │
                                                    │   Dashboard Streamlit (:8501)     │
                                                    │                                  │
                                                    └──────────────────────────────────┘
```

### Workflow de developpement

| Etape | Ou | Pourquoi |
|-------|-----|----------|
| Coder & prototyper | Mac local (MPS) | Confort, iteration rapide |
| Tests rapides GPU | Google Colab | GPU gratuit |
| Entrainement complet | Serveur NVIDIA (CUDA) | Puissance GPU, gros dataset |
| API d'inference | Serveur NVIDIA | Disponible 24/7 |
| Capture d'images | Drone | Son seul role |
| Visualisation | Dashboard Streamlit | Pour les operateurs |

### Acces au serveur

```bash
# Connexion
ssh user@ip_du_serveur

# Meme environnement que le PC local
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Transfert des donnees
scp -r data/ user@serveur:/chemin/projet/data/

# Entrainement
python src/training/train.py

# Deploiement API
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

---

## Pipeline du Projet (detaille)

```
Phase 1 : Segmentation des tubes recepteurs
============================================
Drone (camera RGB)
      |
      v
YOLOv11n-seg (pretraine sur AerialCSP)
      |
      v
Fine-tuning sur images Green Energy Park
      |
      v
Segmentation precise des tubes recepteurs

Phase 2 : Detection d'anomalies
================================
ROI du tube segmente
      |
      +-- Images RGB (anomalies visuelles)
      +-- Images thermiques (perte de vide, hydrogene)
      +-- Donnees capteurs (temperature, meteo)
      |
      v
Modele de classification multi-modal
      |
      v
Type d'anomalie + Localisation + Confiance + Recommandation
```

---

## Prochaines Etapes

- [ ] Recevoir l'acces au dataset AerialCSP (demande envoyee)
- [ ] Adapter `configs/aerialcsp_dataset.yaml` a la structure reelle du dataset
- [ ] Lancer le premier entrainement sur Mac local (petit subset pour valider)
- [ ] Entrainement complet sur le serveur NVIDIA (200 epochs)
- [ ] Evaluer les resultats (mAP50 cible : >95% detection, >70% segmentation)
- [ ] Collecter des images reelles a Green Energy Park pour fine-tuning supplementaire
- [ ] Tester l'API FastAPI avec des images du drone
- [ ] Phase 2 : detection d'anomalies sur les tubes segmentes

---

*Derniere mise a jour : 11 Mars 2026*
