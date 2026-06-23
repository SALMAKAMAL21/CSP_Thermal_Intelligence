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
| Detection / Segmentation | YOLOv11-seg (Ultralytics) |
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

---

## Pipeline Frontend/Backend (Etat Actuel)

Cette section decrit le systeme de test en local (annotation image/video) qui tourne avec Next.js + FastAPI.

### Objectif Metier

- detecter les tubes recepteurs par IA
- appliquer la convention metier:
  - `tube_ref` = tube du haut
  - `tube_test` = tube du bas

Comme les deux tubes se ressemblent fortement, la distinction visuelle pure est instable. La convention haut/bas est appliquee en post-traitement pour stabiliser les labels.

### Architecture

1. Frontend Next.js (`frontend/`)
- Upload image/video RGB ou thermique
- Extraction de frames video
- Appel des routes API Next:
  - `GET /api/segment-check`
  - `POST /api/segment-predict`
- Rendu des annotations (bbox/polygones) sur canvas
- Export video annotee `.webm`

2. Proxy API Next.js
- Forward vers backend Python configure par `SEGMENTATION_API_URL`
- Gestion d'erreurs claire (status backend + payload)

3. Backend FastAPI (`src/inference/api.py`)
- Charge le modele YOLO (priorite `ml/best_model.pt`)
- Endpoint `/predict`: infer sur image
- Applique la regle metier haut/bas pour `tube_ref`/`tube_test`
- Retourne detections JSON

### Flux Video

1. Upload video
2. Echantillonnage de frames
3. Inference frame par frame
4. Stabilisation temporelle (comble les trous intermittents)
5. Relecture video + overlays
6. Export video annotee

### Configuration rapide

Backend:
```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8002 --reload
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

`frontend/.env.local`:
```env
SEGMENTATION_API_URL=http://127.0.0.1:8002
```

### Documentation detaillee

Le guide complet (pas a pas) est dans:
- `frontend/README.md`

---

## Anomaly Detection Sans Labels Reel (Ref/Test)

### Pourquoi cette approche

Ton dataset actuel contient uniquement des tubes sains. Donc on ne peut pas entrainer directement un classifieur supervise "normal vs anomalie reelle".

La strategie mise en place dans le notebook `notebooks/train_anomaly_ref_test_zero_anomaly.ipynb` est:

1. Extraire les crops des deux tubes (`tube_ref` en haut, `tube_test` en bas)
2. Garder `tube_ref` intact (reference saine)
3. Injecter des anomalies synthetiques uniquement dans `tube_test`
4. Entrainer un modele binaire sur paires:
   - paire normale -> label `0`
   - paire anormale synthetique -> label `1`

Le modele apprend alors un **score de deviation** de `tube_test` par rapport a `tube_ref`.

### Comment les anomalies synthetiques sont creees

Fonction: `add_synthetic_anomaly(img)` dans le notebook.

Modes utilises:
- `scratch`: traits/rayures aleatoires (`draw.line`)
- `blob`: taches locales claires/sombres (`draw.ellipse`)
- `occlusion`: masque rectangulaire local (`draw.rectangle`)
- `blur_patch`: flou local d'un patch (`ImageFilter.GaussianBlur`)

Les parametres (position, taille, intensite) sont randomises a chaque appel.

### Bibliotheques utilisees

- `Pillow (PIL)`: manip image + dessin des anomalies synthetiques
- `NumPy`: operations numeriques
- `PyTorch`: modele, entrainement, dataloaders
- `torchvision`: backbone (`resnet18`), transformations
- `scikit-learn`: metriques (`roc_auc_score`, `average_precision_score`)

---

## Comparaison Honnette: Ref/Test Synthetique vs PatchCore/Autoencoder

### Option A - Ref/Test avec anomalies synthetiques (actuelle)

Avantages:
- Exploite ta connaissance metier forte: `tube_ref` est sain
- Simple a deployer et rapide a iterer
- Souvent efficace meme sans anomalies reelles
- Donne un score interpretable de divergence `test` vs `ref`

Limites:
- Le modele apprend les anomalies **qu'on simule** (biais de simulation)
- Peut manquer des vraies anomalies qui ne ressemblent pas aux syntheses
- Calibration seuil indispensable sur donnees terrain

### Option B - PatchCore / Autoencoder (one-class normal)

Principe:
- apprendre uniquement la distribution du "normal"
- toute deviation future -> anomalie

Avantages:
- Plus conforme au setup "zero anomaly labels"
- Souvent meilleur pour detecter anomalies inattendues
- PatchCore est fort en detection d'anomalies texture/locales

Limites:
- Plus sensible a changement de domaine (RGB/thermique, eclairage, angle)
- Peut produire plus de faux positifs si normal train pas assez divers
- Calibration + validation terrain encore plus importantes

### Recommandation brute pour ton cas

Court terme (livrable rapide):
- garder l'approche actuelle Ref/Test synthetique (deja integree et exploitable)

Moyen terme (plus robuste scientifiquement):
- ajouter un benchmark PatchCore (et/ou autoencoder) en parallele
- comparer sur memes videos terrain:
  - taux de faux positifs
  - sensibilite aux anomalies visuelles/thermiques
  - stabilite temporelle en video

Decision pratique:
- si objectif prioritaire = operationnel vite -> Ref/Test synthetique
- si objectif prioritaire = detection d'inconnu sans labels -> PatchCore a privilegier apres benchmark

---

## Historique Q/R (Session)

### Q: Creer un frontend Next.js pour upload video + temperatures + connexion IA + PDF
R: Frontend cree dans `frontend/` avec:
- upload video
- saisie temperatures
- test connexion backend segmentation
- generation PDF (phase initiale)

### Q: Brancher le test reel sur le modele segmentation
R: Flux video reel implemente:
- extraction frames
- appel `/predict`
- affichage resultats IA
- export video annotee

### Q: Changer objectif vers video annotee uniquement (sans comptage, sans PDF)
R: Pipeline adapte:
- sortie = video annotee
- support RGB + thermique
- telechargement `.webm`

### Q: Probleme 500/404 sur `/api/segment-predict`
R: Ajout debug proxy:
- remontee `backendStatus`, `backendUrl`, `backendResponse`
- correction config backend/port

### Q: Utiliser port 8002
R: Config frontend/backend basculee sur `8002`.

### Q: Lier le modele `yolo_seg`/`best_model`
R: API mise a jour pour resolution automatique du modele avec priorite:
1. `MODEL_PATH`
2. `ml/best_model.pt`
3. autres fallbacks

### Q: Ajouter test image pour comparer image vs video
R: Mode image ajoute dans frontend avec annotation directe et export image.

### Q: Notebook entrainement YOLO26 selon dataset
R: Notebook cree puis optimise (YOLO26s, split analyse, 2 phases freeze/unfreeze).

### Q: Diagnostic de la derive `tube_ref/tube_test`
R: Audit montre:
- classes quasi identiques visuellement
- convention top/bottom predominante
- besoin post-regle metier et split par video

### Q: Comparer datasets `Tubes CSP.v1i.yolo26.zip` vs `Tubes CSP.yolov11.zip`
R:
- `yolo26`: split complet, compact
- `yolov11`: plus lourd, meilleure resolution, mais split incomplet
- recommandation: creer split propre par video pour `yolov11`

### Q: Faire script de split par video et l'executer
R: Script cree: `scripts/split_yolov11_by_video.py`
- dataset genere: `data/tubes_csp_yolov11_grouped`
- train/valid/test coerents

### Q: Optimiser avec regle metier haut/bas
R:
- notebook adapte vers logique metier
- backend applique `tube_ref=haut`, `tube_test=bas`

### Q: Thermique: trous de detection au milieu
R:
- stabilisation temporelle ajoutee dans frontend
- recommandation d'ajuster `conf` (seuil confiance)

### Q: Demande d'explication architecture complete
R:
- documentation "cours" ajoutee dans `frontend/README.md`
- section resume ajoutee dans `README.md` racine

### Q: Creer notebook anomaly detection sans anomalies reelles
R: Notebook cree: `notebooks/train_anomaly_ref_test_zero_anomaly.ipynb`
- approche ref/test
- anomalies synthetiques sur `tube_test`
- score anomalie binaire

### Q: Comprendre generation anomalies synthetiques
R: Explication + ajout modes CSP-realistes:
- `longitudinal_defect`
- `thermal_band_irregularity`
- `hotspot_localized`
- `endcap_defect`

### Q: Clarifier YOLO-only vs architecture 2 etapes
R:
- comparaison honnete documentee
- YOLO-only propose pour delivery simple

### Q: Demande notebook YOLO-only et fine-tuning depuis `best.pt` en segmentation
R: Notebook cree et corrige:
- `notebooks/train_anomaly_yolo_only_synth.ipynb`
- fine-tuning depuis `best.pt`
- `task='segment'`

