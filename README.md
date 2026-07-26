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



J’ai implémenté la couche de décision vidéo au-dessus du
  score frame-level existant.

  - ajout d’une config versionnée pour les seuils vidéo dans
    configs/thermal_video_decision.json:1

  - ajout du module Python de décision vidéo dans src/
    inference/video_decision.py:1

  - extension du script de calibration pour produire aussi
    decision, decision_confidence, longest_suspect_run et
    peak_frame_time par vidéo dans scripts/
    calibrate_thermal_from_videos.py:243

  - côté frontend, le résumé d’analyse utilise maintenant
    une vraie logique vidéo avec normal / warning / anomaly,
    ratio de frames suspectes, persistance et confiance dans
    frontend/lib/thermal.ts:56

  - l’UI affiche désormais la décision vidéo comme statut
    principal dans frontend/components/csp/anomaly-
    section.tsx:1, frontend/components/csp/segmentation-
    section.tsx:1 et le PDF reprend aussi cette décision
    dans frontend/hooks/use-csp-analysis.ts:440

  La règle active par défaut est:

  - warning si score max, score moyen ou ratio suspect
    dépassent les seuils faibles

  - anomaly seulement si le score fort est atteint avec
    persistance ou ratio suspect suffisant

  - sinon normal

  Vérifications faites:

  - python -m py_compile ... sur les modules Python modifiés
  - npm run build dans frontend/ réussi

  Le fichier README.md était déjà modifié avant mon travail,
  je ne l’ai pas touché. Si tu veux, l’étape logique
  suivante est de faire tourner à nouveau le script de
  calibration ou un jeu de vidéos défectueuses pour voir
  quelles vidéos passent en warning ou anomaly avec cette
  nouvelle logique.

  ne change pas n'est ce pas?


• Oui, exact.
  thermal_anomaly.py:14 n’ont pas changé:

  - z_threshold = 3.0
  - min_delta = 0.20
  - strong_delta = 0.30
  - etc.

  Ce que j’ai ajouté, c’est une couche au-dessus:

  - le score frame-level est toujours calculé avec
    ThermalConfig

  - ensuite on applique une décision vidéo avec d’autres
    seuils dans configs/thermal_video_decision.json:1

  Donc aujourd’hui il y a bien 2 niveaux de seuils:

  1. Seuils frame-level internes

  - dans thermal_anomaly.py
  - servent à calculer anomaly_score sur une image/frame

  2. Seuils décision vidéo

  - dans thermal_video_decision.json
  - servent à conclure normal / warning / anomaly sur toute
    la vidéo

  C’était volontaire, pour ne pas casser brutalement le
  comportement actuel.

  Si tu veux aller plus loin maintenant, il y a 2 options:

  - soit on laisse ThermalConfig tel quel et on ajuste
    seulement la décision vidéo

  - soit on branche aussi ThermalConfig sur une config
    externe pour tester de nouveaux min_delta /
    strong_delta / z_threshold

  La deuxième option est probablement la prochaine étape
  utile si tu veux réellement exploiter les résultats de
  calibration jusque dans le cœur du score.


› pourquoi? c'etait bien pour trouver les bonne valeurs
  qu'on a fait le script de calibrage n'est ce pas? ou non
  c'est pour cela que video_decision est la? et puis est ce
  encore necessaire de faire le calibrage avec les videos
  de tube_test anomalies ?


• Oui, le script de calibrage servait bien à trouver de
  meilleures valeurs, mais il a révélé deux choses
  différentes.

  La première:

  - il donne des valeurs observées sur des vidéos normales
  - donc il aide à comprendre où se situe le bruit normal du
    système

  La deuxième:

  - il ne garantit pas que ces valeurs doivent être
    injectées telles quelles dans ThermalConfig

  - surtout quand certaines sortent de manière peu crédible,
    comme z_threshold ≈ 14.7

  C’est pour ça que j’ai séparé les rôles:

  - ThermalConfig dans src/inference/thermal_anomaly.py:14
    pilote le score frame-level

  - video_decision dans src/inference/video_decision.py:1
    pilote la décision finale sur une vidéo

  Pourquoi cette séparation est utile:

  - le calibrage sur vidéos normales a montré que les scores
    vidéo sont exploitables

  - mais il a aussi montré que certains seuils internes du
    score, surtout z_threshold, ne sont pas encore assez
    stables pour être repris aveuglément

  - donc on a sécurisé le système d’abord au niveau décision
    vidéo, là où les résultats étaient déjà interprétables

  Donc:

  - non, le calibrage n’était pas inutile
  - oui, video_decision est là justement pour exploiter
    proprement ce que la calibration a donné de plus fiable

  - non, ça ne veut pas encore dire que le cœur de
    ThermalConfig est calibré correctement

  Et oui, il est encore nécessaire de faire le calibrage ou
  plutôt la validation avec les vidéos où tube_test est

  Sur les vidéos normales, tu réponds à:

  - “jusqu’où le système peut monter sans défaut ?”

  - “à partir de quand le système sépare vraiment un défaut
    d’un cas normal ?”

  Sans vidéos anormales, tu peux régler pour limiter les
  faux positifs, mais tu ne sais pas encore si tu rates les
  vrais défauts.

  La bonne lecture est donc:

  1. vidéos normales:

  - calibrer le bruit normal
  - fixer des seuils prudents
  - éviter les faux positifs

  2. vidéos défectueuses:

  - vérifier la séparation
  - mesurer ce qui dépasse réellement
  - ajuster si les anomalies ne sortent pas assez fort

  La prochaine étape logique est:

  - tester les vidéos défectueuses avec le système actuel
  - regarder pour chaque vidéo:
      - max_anomaly_score
      - mean_anomaly_score
      - suspect_frame_ratio
      - longest_suspect_run
      - decision

  Et ensuite décider:

  - soit on garde ThermalConfig tel quel
  - soit on recalibre aussi min_delta, strong_delta,
    z_threshold dans le moteur frame-level

  Ma recommandation:

  - d’abord tester sur les vidéos défectueuses
  - ensuite seulement décider si on modifie ThermalConfig

  Parce que changer ThermalConfig maintenant, sans voir
  l’effet sur les vraies anomalies, serait encore un peu à
  l’aveugle.


› okay fine, j'utilise le meme script de calibrage pour les
  videos de tube avec anomalies?


• Oui, tu peux utiliser le même script.

  Il n’est pas limité techniquement aux vidéos normales. Il
  parcourt les vidéos, extrait les frames, calcule les
  métriques frame-level et produit un résumé vidéo. Donc
  pour les vidéos avec anomalie, il va aussi sortir:

  - max_anomaly_score
  - longest_suspect_run
  - decision
  - decision_confidence

  - sur les vidéos normales, tu t’en sers pour calibrer le
    bruit normal

  - sur les vidéos défectueuses, tu t’en sers pour valider
    la séparation et voir si les vraies anomalies montent
    au-dessus des normales

  Comme ça tu gardes séparé:

  - artifacts/thermal_calibration/ pour les normales
  - artifacts/thermal_eval_defect/ pour les défectueuses

  Point important:

  - ne prends pas les recommended_thresholds.json des vidéos
    défectueuses pour recalibrer directement les seuils
    “normaux”

  - utilise-les surtout pour comparer les distributions avec
    celles des vidéos normales

  Ce qu’on veut vérifier ensuite:

  - les vidéos normales restent surtout normal ou parfois
    warning

  - les vidéos défectueuses passent plutôt en warning fort
    ou anomaly

  Si tu veux, quand tu auras les résultats des vidéos
  défectueuses, envoie-moi surtout:

  - summary_videos.json
  - et idéalement les 5 vidéos avec les scores les plus
    élevés

  et je t’aiderai à décider si on doit toucher ThermalConfig
  ou seulement les seuils de décision vidéo.


› okay avant que je revienne avec les resultats j'ai fait
  l'entrainement de l'autoencoder et je voudrais savoir
  comment il sera implementer dans notre application?


• Je vais d’abord vérifier s’il y a déjà une base
  autoencoder dans le repo ou s’il faudra l’intégrer comme
  une nouvelle brique à côté du pipeline actuel segmentation
  -> comparaison thermique. Ensuite je te dirai la manière
  la plus propre de l’insérer dans l’application.

• Explored
  └ Search autoencoder|reconstruction|latent|encoder|decoder
  in
           src
    List rg --files
    Search model|models|ae|auto

────────────────────────────────────────────────────────────

• Il n’y a actuellement aucune intégration autoencoder dans
  le repo. Le pipeline existant reste:

  - segmentation des tubes
  - attribution tube_ref / tube_test
  - score thermique comparatif
  - décision vidéo

  Donc l’autoencoder ne va pas “remplacer” naturellement
  l’application actuelle. Il faut l’ajouter comme une
  nouvelle brique d’analyse.

  La bonne manière de l’intégrer est celle-ci:

  1. Où il se place dans le pipeline
  Le plus propre est:

  - vidéo/frame
  - segmentation tube_ref et tube_test
  - extraction du ROI du tube_test
  - envoi du ROI du tube_test à l’autoencoder
  - calcul d’un score de reconstruction
  - fusion avec le score comparatif actuel

  Donc l’autoencoder travaille surtout sur tube_test, pas
  sur l’image entière.

  2. Ce qu’il produit
  L’autoencoder doit renvoyer au minimum:

  - reconstruction_error
  - ae_decision ou ae_score
  - éventuellement une heatmap de reconstruction si tu veux
    visualiser où il échoue

  Dans votre application, ce score devient une seconde
  source d’évidence:

  - score comparatif tube_ref vs tube_test
  - score autoencoder sur tube_test

  3. Comment l’utiliser dans l’application
  Tu as 2 options réalistes.

  Option simple, recommandée:

  - garder le système actuel comme principal
  - ajouter l’autoencoder comme score complémentaire
  - la décision finale combine:
      - score comparatif
      - score autoencoder
      - persistance vidéo

  Exemple:

  - si les deux sont élevés -> anomalie forte
  - si un seul est élevé -> warning
  - si les deux sont faibles -> normal

  Option plus risquée:

  - utiliser l’autoencoder seul
  - je ne te le conseille pas tout de suite, car vous avez
    déjà un pipeline métier intéressant avec tube_ref

  4. Où l’implémenter
  Concrètement dans ce repo, l’endroit naturel serait:

  - nouveau module Python, par exemple src/inference/
    autoencoder_anomaly.py

  - chargement du modèle au démarrage dans src/inference/
    api.py:1

  - appel après segmentation, une fois tube_test trouvé
  - retour dans la réponse JSON avec un nouvel objet, par
    exemple:
      - autoencoder_anomaly
      - et plus tard combined_anomaly

  5. Ce qu’il faut pour que ce soit propre
  Il faut définir:

  - le format d’entrée du modèle
      - image RGB ?
      - thermique ?
      - tube normalisé/redressé ?

  - la taille d’entrée
      - 128x128, 256x64, etc.

  - la métrique d’erreur
      - MSE, MAE, SSIM inversé…

  - le seuil d’anomalie autoencoder
      - calibré sur des tubes normaux

  6. Le meilleur usage dans votre cas
  Vu votre projet, le meilleur design est:

  - la segmentation détecte les tubes
  - le module comparatif compare tube_ref et tube_test
  - l’autoencoder regarde seulement tube_test comme un
    détecteur d’écart au normal

  - la décision finale fusionne les deux

  C’est mieux qu’un simple classifieur, parce que:

  - vous gardez l’explicabilité métier avec tube_ref
  - vous ajoutez une détection “data-driven” avec
    l’autoencoder

  En résumé:

  - l’autoencoder sera implémenté comme un second moteur
    d’anomalie

  - il prendra le ROI de tube_test
  - il renverra un score de reconstruction
  - ce score sera fusionné avec la comparaison thermique
    actuelle dans l’API et l’UI

  Si tu veux, je peux maintenant te proposer l’architecture
  exacte d’intégration de l’autoencoder dans votre backend
  actuel, fichier par fichier et flux par flux.

Ces résultats sont bons. Ils montrent que les vidéos
  anormales montent beaucoup plus haut que les vidéos
  normales.

  En résumé:

  - sur les vidéos normales, tu avais environ:
      - frame p99 ≈ 0.2417
      - video alert ≈ 0.377

  - sur les vidéos anormales, tu as maintenant:
      - frame p99 ≈ 0.7195
      - video alert ≈ 0.7488

  Donc il y a une séparation nette entre normal et anomalie.
  C’est exactement ce qu’on voulait vérifier.

  Ce que ça veut dire sur tes stats anomalies:

  - usable_ratio = 0.9804
      - très bien, le pipeline marche sur presque toutes les
        frames

  - score_stats.p99 = 0.7195, max = 0.755
      - les vraies anomalies produisent des scores forts

  - delta_stats.p99 = 0.4872
      - les écarts tube_test vs tube_ref sont plus élevés
        que sur le normal

  - z_score_stats.p99 = 9.6828
      - encore élevé, donc le z_threshold reste peu stable
        comme critère principal

  - segment_strength_stats.p99 = 1.0
      - certaines frames anormales atteignent le niveau max
        de force segmentaire

  Le point clé:

  - oui, les anomalies sortent clairement au-dessus du
    normal

  - donc votre système comparatif thermique est utile
  - et la couche video_decision a du sens

  Par contre:

  - les recommended_thresholds obtenus sur les vidéos
    anormales ne doivent pas être copiés tels quels dans la
    config

  - ils servent surtout à comparer avec les normales, pas à
    définir le seuil final

  La bonne lecture pratique est:

  - vidéos normales:
      - bruit max typique vers 0.24
      - cas très limites vers 0.38

  - vidéos anormales:
      - fortes réponses vers 0.72 et plus

  Donc un seuil vidéo final entre les deux est crédible, par
  exemple:

  - warning autour de 0.24 - 0.30
  - anomaly autour de 0.40 - 0.50

  Autrement dit:

  - 0.72 est trop haut pour être le seuil d’alerte final
  - sinon tu raterais des anomalies plus faibles
  - il faut garder un seuil séparateur entre vos normales et
    vos anomalies, pas prendre le p99 des anomalies comme
    seuil

  Conclusion simple:

  - vos résultats anomalies valident le pipeline
  - ils montrent qu’il y a bien une séparation exploitable
  - ils justifient de garder la logique actuelle:
      - score thermique comparatif
      - décision vidéo
      - autoencoder en complément

  L’étape utile maintenant est de comparer côte à côte:

  - top scores des vidéos normales
  - top scores des vidéos anormales
  - décisions normal / warning / anomaly

  Si tu veux, je peux maintenant te dire quels seuils mettre
  concrètement dans:

  - configs/thermal_video_decision.json
  - et éventuellement dans ThermalConfig aussi.

  J’ai modifié configs/thermal_video_decision.json:1 comme
  suit:

  {
    "frame_warning_score": 0.26,
    "frame_alert_score": 0.42,
    "video_mean_warning_score": 0.02,
    "video_mean_alert_score": 0.04,
    "suspect_ratio_warning": 0.04,
    "suspect_ratio_alert": 0.1,
    "min_persistent_frames": 2
  }

  Pourquoi j’ai mis ça:

  - frame_warning_score = 0.26
      - vos normales ont p99 ≈ 0.2417
      - je le mets juste au-dessus pour éviter que les cas
        normaux borderline partent trop facilement en
        warning

  - frame_alert_score = 0.42
      - vos normales culminent vers 0.3964
      - vos anomalies montent vers 0.72+
      - 0.42 crée une vraie marge au-dessus du pire cas
        normal, sans attendre des scores extrêmes

  - video_mean_warning_score = 0.02
      - je le laisse, car vos vidéos normales avaient déjà
        des moyennes très basses

      - ça reste utile pour capter une anomalie faible mais
        persistante

  - video_mean_alert_score = 0.04
      - je l’abaisse de 0.05 à 0.04
      - parce que sur les vidéos anormales, le score moyen
        peut rester modéré même si certaines frames sont
        très parlantes

  - suspect_ratio_warning = 0.04
      - avant c’était 0.05
      - je le baisse légèrement pour capter plus tôt une
        dérive persistante, sans être trop agressif

  - suspect_ratio_alert = 0.10
      - je le garde
      - 10% de frames suspectes reste une bonne condition de
        persistance pour éviter un pic isolé

  - min_persistent_frames = 2
      - je le garde aussi
      - c’est un bon compromis avec votre échantillonnage
        actuel

  Ce que je n’ai pas changé:

  - les seuils de ThermalConfig dans src/inference/
    thermal_anomaly.py:14

  Pourquoi:

  - le z_threshold calibré reste trop instable (9.9 sur
    anomalies, 14.7 sur normales auparavant)

  - donc ce n’est pas encore un bon candidat à injecter
    directement dans le cœur du score frame-level

  - la séparation la plus fiable aujourd’hui est au niveau
    décision vidéo, pas au niveau z_threshold

  En pratique, avec ces valeurs:

  - une vidéo normale forte devrait plutôt finir en warning
    au pire

  - une vraie vidéo anormale avec persistance devrait mieux
    passer en anomaly

  La prochaine chose à faire est de relancer quelques vidéos
  normales et défectueuses dans l’application et regarder
  si:

  - les normales extrêmes ne passent plus en anomaly
  - les défectueuses passent bien en anomaly

  Si tu veux ensuite, je peux aussi faire une deuxième passe
  pour brancher une config externe sur ThermalConfig lui-
  même.

video_decision n’est pas un remplaçant de thermal_anomaly.
  C’est une couche au-dessus.

  Le flux actuel est:

  1. Segmentation

  - le backend détecte les tubes dans src/inference/
    api.py:248

  - il attribue tube_ref et tube_test

  2. Analyse frame par frame

  - pour chaque frame, src/inference/thermal_anomaly.py:34
    calcule:
      - anomaly_score
      - severity
      - suspect_segments
      - etc.

  - ça, c’est le cœur image/frame

  3. Complément autoencoder

  - dans l’API, on calcule aussi l’analyse autoencoder avec
    src/inference/autoencoder_anomaly.py:121

  - puis on fusionne thermique + autoencoder avec src/
    inference/autoencoder_anomaly.py:191

  4. Décision vidéo

  - video_decision intervient après, quand on a une liste de
    résultats frame par frame

  - la logique est dans src/inference/video_decision.py:58
  - elle prend plusieurs frames et calcule:
      - max_anomaly_score
      - mean_anomaly_score
      - suspect_frame_ratio
      - longest_suspect_run
      - decision = normal / warning / anomaly

  Donc la relation est:

  - thermal_anomaly = score local sur une frame
  - autoencoder_anomaly = signal complémentaire sur une
    frame

  - combined_anomaly = fusion sur une frame
  - video_decision = décision finale sur toute la vidéo

  Dans l’application frontend, c’est branché comme ça:

  - chaque frame analysée retourne thermal_anomaly,
    autoencoder_anomaly, combined_anomaly

  - le hook frontend/hooks/use-csp-analysis.ts:128 accumule
    toutes les frames

  - puis frontend/lib/thermal.ts:230 applique la logique
    vidéo

  - et l’UI affiche la décision finale

  Donc en une phrase:

  thermal_anomaly produit le score brut par frame, et
  video_decision transforme une séquence de ces scores en
  verdict vidéo exploitable.

  Si tu veux, je peux te faire juste après un petit schéma
  visuel en 6 lignes du pipeline complet.


  Concrètement: dans l’état actuel, video_decision.py n’est
  pas branché directement dans api.py.

  Il est connecté au système de cette manière:

  1. Backend frame-level

  - src/inference/api.py:1 analyse chaque frame
  - il renvoie:
      - thermal_anomaly
      - autoencoder_anomaly
      - combined_anomaly

  2. Décision vidéo côté script

  - src/inference/video_decision.py:1 est utilisé par le
    script de calibration

  - scripts/calibrate_thermal_from_videos.py:38
  - donc pour les scripts offline, c’est bien ce module
    Python qui fait normal / warning / anomaly


  - le hook accumule les frames dans frontend/hooks/use-csp-
    analysis.ts:279

  Donc la vérité simple est:

  - video_decision.py n’est pas nécessaire pour que l’API
    FastAPI tourne

  - il est utilisé pour les scripts d’analyse/calibration
  - l’application web utilise actuellement une logique
    équivalente mais en TypeScript

  Si tu veux envoyer le code à ton amie pour que ça
  fonctionne pareil chez elle, il faut lui envoyer au
  minimum:

  - src/inference/api.py:1
  - src/inference/thermal_anomaly.py:1
  - src/inference/autoencoder_anomaly.py:1
  - src/inference/video_decision.py:1 si elle veut aussi
    lancer le script de calibration

  - configs/thermal_video_decision.json:1
  - scripts/calibrate_thermal_from_videos.py:1
  - et bien sûr les modèles:
      - src/models/best_model.pt
      - src/models/best_autoencoder.pt

  Si elle veut juste faire tourner le backend + frontend:

  - video_decision.py n’est pas obligatoire au runtime de
    l’API

  - mais configs/thermal_video_decision.json et la logique
    frontend oui, parce que la décision vidéo affichée dans
    l’interface dépend surtout de frontend/lib/
    thermal.ts:230

  Donc en une phrase:
  video_decision.py est connecté au système surtout pour
  l’analyse offline, alors que l’application web prend les
  scores du backend et fait la décision vidéo côté frontend.

  Si tu veux, je peux aussi te dire exactement quels
  fichiers ziper et envoyer à ton amie pour éviter tout
  oubli.


› src/inference/video_decision.py:1 , j'ai bien compris,
  sert a quoi en alors, car si j'ai bien compris quand je
  lance le backend c'est seulement le  scripts/
  calibrate_thermal_from_videos.py qui est lancer aussi ,
  et et autre modification ce font cote frontend


• Non.

  Quand tu lances le backend avec uvicorn, scripts/
  calibrate_thermal_from_videos.py ne se lance pas.

  Il faut bien séparer 3 choses:

  1. Le backend FastAPI

  - lancé avec uvicorn src.inference.api:app ...
  - il charge:
      - la segmentation
      - thermal_anomaly.py
      - autoencoder_anomaly.py

  - il analyse une frame à la fois
  - il ne lance pas automatiquement video_decision.py
  - il ne lance pas automatiquement
    calibrate_thermal_from_videos.py

  2. Le script scripts/calibrate_thermal_from_videos.py

  - c’est un script manuel
  - tu le lances toi-même en terminal
  - il sert à:
      - parcourir un dossier de vidéos
      - extraire des frames
      - calculer les scores
      - résumer les vidéos
      - proposer des seuils

  - lui, il utilise src/inference/video_decision.py

  3. Le frontend

  - il appelle le backend frame par frame
  - il récupère les scores renvoyés
  - puis il fait la décision vidéo côté frontend dans
    frontend/lib/thermal.ts

  Donc, très précisément:

  - src/inference/video_decision.py
      - sert pour l’analyse offline
      - sert au script de calibration
      - ne sert pas directement quand tu lances juste
        FastAPI

  - scripts/calibrate_thermal_from_videos.py
      - ne se lance jamais tout seul
      - seulement si tu tapes la commande toi-même

  - le backend + frontend
      - fonctionnent sans lancer ce script
      - la décision vidéo visible dans l’app est
        actuellement refaite côté frontend

  Donc si ton amie veut:

  - utiliser l’application web: backend + frontend suffisent
  - faire aussi le calibrage/évaluation vidéo offline: il
    lui faut en plus video_decision.py et
    calibrate_thermal_from_videos.py

  La confusion venait de là:

  - video_decision.py n’est pas branché au runtime principal
    du backend

  - il est branché au runtime du script de calibration
  - et la logique équivalente existe aussi côté frontend

  Si tu veux, je peux te dire maintenant si on devrait
  déplacer video_decision dans le backend lui-même pour que
  toute la logique soit centralisée au même endroit.


Oui. Si l’objectif est de ne pas juste raffiner
  l’existant, mais d’ouvrir d’autres pistes, je te
  conseillerais de penser en familles d’approches.

  Aujourd’hui vous avez déjà:

  - comparaison thermique tube_ref vs tube_test
  - autoencoder comme signal complémentaire
  - décision vidéo

  Les nouveaux horizons intéressants seraient:

  1. Approches vidéo natives
  Au lieu d’analyser frame par frame, entraîner un modèle
  sur une séquence.
  Exemples:

  - CNN + LSTM
  - Temporal ConvNet
  - Vision Transformer vidéo léger
  - modèle sur clips de 8 à 32 frames

  Intérêt:

  - détecter une anomalie par son évolution temporelle
  - réduire les faux positifs dus à une frame isolée

  2. Siamese / metric learning
  Comme vous avez toujours tube_ref sain et tube_test à
  inspecter, vous avez un cas très adapté au comparatif
  appris.
  Idée:

  - deux branches prennent tube_ref et tube_test
  - le modèle apprend une distance entre les deux
  - sortie: “similaire” ou “anormalement différent”

  C’est probablement la piste la plus naturelle pour votre
  problème.

  3. One-class / self-supervised sur le normal
  Au lieu d’apprendre directement les défauts, entraîner sur
  le normal seulement:

  - One-Class CNN
  - Deep SVDD
  - student-teacher anomaly detection
  - patch-based anomaly detection

  Intérêt:

  - utile si vous avez peu d’anomalies réelles
  - cohérent avec le contexte industriel

  4. Détection localisée du défaut
  Au lieu d’un score global tube sain/défectueux:

  - détecter ou segmenter la zone exacte du défaut
  - bbox ou masque d’anomalie

  Exemples:

  - YOLO détection défaut
  - segmentation fine type U-Net / YOLO-seg
  - méthode weakly supervised si annotations limitées

  Intérêt:

  - résultat plus explicable
  - meilleur pour le rapport final et l’usage terrain

  5. Approches patch-based
  Découper tube_test en petites régions le long du tube.
  Puis:

  - score par patch
  - agrégation globale

  Intérêt:

  - meilleure sensibilité aux défauts locaux
  - plus robuste qu’un score global unique

  6. Fusion multimodale plus sérieuse
  Vous avez déjà l’idée vision + température.
  Vous pouvez aller plus loin avec:

  - image thermique
  - profil relatif ref/test
  - températures saisies T1..T4
  - métadonnées vidéo ou environnement

  Puis fusionner dans un petit modèle tabulaire ou
  multimodal.

  7. Teacher-student / distillation anomaly
  Très bon compromis moderne.
  Idée:

  - un réseau teacher fixe
  - un student apprend le normal
  - l’écart teacher-student devient le score d’anomalie

  Souvent plus stable qu’un autoencoder classique.

  8. Hybridation explicite métier + IA
  Au lieu d’un seul modèle:

  - score thermique relatif
  - score autoencoder
  - score siamese
  - score patch local
  - décision finale par fusion pondérée

  Pour un contexte industriel, c’est souvent meilleur qu’un
  unique gros modèle.

  Ma recommandation pragmatique, par ordre de priorité:

  1. Siamese tube_ref / tube_test
  2. Modèle vidéo sur clips
  3. Patch-based anomaly sur tube_test
  4. One-class plus robuste que l’autoencoder classique
  5. Détection localisée si vous obtenez des annotations

  Si tu veux une direction vraiment forte pour votre sujet,
  je choisirais:

  - axe 1: comparatif appris tube_ref vs tube_test
  - axe 2: anomalie vidéo temporelle
  - axe 3: fusion avec votre pipeline thermique existant

  Ça ferait un projet beaucoup plus solide scientifiquement.

  Si tu veux, je peux maintenant te proposer un roadmap
  expérimental concret:

  - quelles 3 nouvelles approches tester
  - dans quel ordre
  - avec quelles entrées, sorties, métriques et baseline.


› Au lieu d’analyser frame par frame, entraîner un modèle
    sur une séquence., ca signifie quoi entrainer sur une
  sequence?


• Ça veut dire que le modèle ne reçoit pas une seule image
  isolée, mais plusieurs frames consécutives d’une même
  vidéo.

  Au lieu de faire:

  - entrée = 1 frame
  - sortie = normal ou anomalie

  on fait:

  - entrée = une séquence de frames
  - sortie = normal ou anomalie

  Exemple:

  - frame à 0.0 s
  - frame à 0.25 s
  - frame à 0.5 s
  - frame à 0.75 s
  - frame à 1.0 s

  Ces 5 frames ensemble forment une séquence ou un clip.

  Pourquoi c’est utile:

  - une anomalie réelle ne se voit pas toujours bien sur une
    seule frame

  - mais si elle apparaît de manière cohérente pendant
    plusieurs frames, le modèle peut mieux la reconnaître

  - ça réduit les faux positifs causés par:
      - bruit thermique
      - mauvais masque
      - reflet momentané
      - flou ou angle temporaire

  Dans votre cas, ça voudrait dire par exemple:

  - on segmente tube_ref et tube_test
  - on extrait le tube_test sur plusieurs frames successives
  - on donne cette suite au modèle
  - le modèle apprend l’évolution temporelle du tube

  Exemple concret de séquence:

  clip_001 =
    frame_10: tube_test
    frame_11: tube_test
    frame_12: tube_test
    frame_13: tube_test
    frame_14: tube_test
  label = defectueux

  Ou:

  clip_002 =
    frame_50: tube_test
    frame_51: tube_test
    frame_52: tube_test
    frame_53: tube_test
    frame_54: tube_test
  label = normal

  Donc “entraîner sur une séquence” =
  apprendre à partir d’un petit morceau de vidéo, pas à
  partir d’une image seule.

  Les formes possibles:

  - 5 frames
  - 8 frames
  - 16 frames
  - parfois 32 frames

  Le modèle peut alors apprendre:

  - si l’anomalie reste présente
  - si elle se déplace
  - si elle s’intensifie
  - si le comportement thermique du tube est stable ou
    bizarre dans le temps

  Pour votre sujet, l’idée la plus naturelle serait:

  - entrée = séquence de ROI de tube_test
  - optionnel = séquence tube_ref + tube_test
  - sortie = normal / defectueux

  Si tu veux, je peux te montrer juste après la différence
  exacte entre dataset frame par frame et dataset par
  séquences, avec une structure de dossiers simple.


› Siamese / metric learning, oui c'est comme ma partie
  analyse thermique n'est ce pas?


• Oui, dans l’idée générale, c’est proche de votre analyse
  thermique actuelle.

  La similarité est:

  - vous avez toujours tube_ref
  - vous avez toujours tube_test
  - vous cherchez à mesurer à quel point ils sont différents

  Donc sur le plan conceptuel:

  - votre système actuel = comparaison manuelle /
    heuristique

  - un modèle siamese = comparaison apprise

  La différence clé est là.

  Votre approche actuelle
  Dans src/inference/thermal_anomaly.py:34, vous:

  - alignez les tubes
  - extrayez des profils thermiques
  - comparez leurs médianes / deltas / z-scores
  - produisez un score avec des règles fixées

  Donc:

  - la comparaison est définie par vous
  - les critères sont explicites
  - c’est très interprétable

  Approche siamese
  Avec un réseau siamese:

  - une branche prend tube_ref
  - une autre prend tube_test
  - le modèle apprend une représentation des deux
  - puis apprend une distance ou une similarité entre eux

  Donc:

  - la comparaison n’est plus écrite à la main
  - elle est apprise à partir des données
  - le modèle découvre lui-même quelles différences sont
    importantes

  En résumé:

  - oui, c’est la même philosophie métier:
      - comparer tube_ref et tube_test

  - non, ce n’est pas la même mécanique:
      - aujourd’hui: règles analytiques
      - siamese: distance apprise

  Tu peux voir ça comme une évolution naturelle de votre
  pipeline.

  Version actuelle

  tube_ref + tube_test
  -> comparaison thermique codée à la main
  -> score d'anomalie

  Version siamese

  tube_ref + tube_test
  -> réseau siamese
  -> embedding / distance apprise
  -> score d'anomalie

  Le gros avantage pour votre cas:

  - votre problème est naturellement pairwise
  - tube_ref sert déjà de référence saine
  - donc un siamese est probablement plus adapté qu’un
    classifieur classique sur tube_test seul

  La bonne stratégie, si vous testez cette voie, ce n’est
  pas de remplacer tout de suite l’analyse thermique.
  C’est plutôt:

  - garder l’analyse thermique comme baseline explicable
  - entraîner un siamese en parallèle
  - comparer les performances
  - éventuellement fusionner les deux

  Si tu veux, je peux maintenant te décrire à quoi
  ressemblerait concrètement un dataset siamese pour votre
  projet.


› d'accord fait cela


• Pour votre projet, un dataset siamese se construit en
  paires.

  Au lieu d’avoir:

  - une image tube_test avec label normal ou defectueux

  tu auras:

  - tube_ref
  - tube_test
  - un label qui décrit leur relation

  La formulation la plus simple est:

  - label = 0 → les deux sont cohérents / cas normal
  - label = 1 → tube_test est anormal par rapport à tube_ref

  Dans votre cas:

  - si tube_ref est normal et tube_test est normal:
      - paire normale

  - si tube_ref est normal et tube_test est defectueux:
      - paire anormale

  Donc le modèle apprend:

  - “à quoi ressemble un couple de tubes compatibles”
  - “à quoi ressemble un couple où le tube test dévie”

  Structure simple du dataset

  Tu peux partir sur un CSV comme ça:

  ref_path,test_path,label,video_id,frame_id
  data/roi/ref/v1_f001.jpg,data/roi/test/
  v1_f001.jpg,0,video_01,1
  data/roi/ref/v2_f014.jpg,data/roi/test/
  v2_f014.jpg,1,video_12,14

  C’est la structure la plus pratique.

  Ou bien en dossiers:

  data/siamese/
    train/
      normal_pairs.csv
      anomaly_pairs.csv
    val/
      normal_pairs.csv
      anomaly_pairs.csv
    test/
      normal_pairs.csv
      anomaly_pairs.csv

  Mais en pratique, un seul fichier pairs.csv par split est
  mieux.

  Ce qu’il faut extraire

  Pour chaque frame exploitable:

  1. segmenter tube_ref
  2. segmenter tube_test
  3. extraire les ROI
  4. sauvegarder les deux crops
  5. créer une ligne de paire avec le bon label

  Exemple:

  frame_083:
    ref crop -> normal
    test crop -> normal
    label pair = 0

  frame_214:
    ref crop -> normal
    test crop -> defectueux
    label pair = 1

  Très important
  Le split doit se faire par vidéo, pas par frame.

  Donc:

  - video_01 entier en train
  - video_02 entier en val
  - video_03 entier en test

  Pas de mélange de frames d’une même vidéo entre train et
  test.

  Deux variantes possibles

  1. Paire simple

  Ça demande plus de labels.

  Je te conseille de commencer par la paire binaire.
  Il faut uniformiser les crops:

  - même taille, par ex 128x128 ou 224x224
  - même orientation si possible
  - même prétraitement pour ref et test

  Sinon le réseau apprend le cadrage au lieu de la vraie
  différence.

  Ce que le modèle apprend
  Chaque branche siamese encode:

  - tube_ref
  - tube_test

  Puis on compare les embeddings:

  - distance faible = normal
  - distance forte = anomalie

  Résumé concret
  Ton dataset siamese, pour votre cas, c’est:

  - une paire d’images: tube_ref + tube_test
  - un label:
      - 0 si tube_test est normal
      - 1 si tube_test est defectueux

  Si tu veux, je peux maintenant te proposer la structure
  exacte des dossiers et du CSV adaptée à votre repo actuel.


› alors comment je peux faire ca? faire un dossier de video
  avec pair normal, et un autre dossier avec pair anormal,
  comme je l'avais fait pour la calibration et ensuite le
  mettre dans data et tu vas t'occuper de faire les
  transformer en dataset exploitable via les models de
  segementations qui sont present pour la recuperation du
  ROI. ou je dois obligatoirement le faire moi meme, car je
  ne comprend pas la partie annotation , par exemple si je
  dois faire l'annotation sur roboflow, quelles sont les
  labels que je dois utiliser exactement, car de ce que
  j'ai compris, c'est trois, normal, anomaly, pairs(lable
  qui donne l'information sur la relation entre tube_test
  et tube_ref), mais dans la pratique je sias pas comment
  faire , je dois encadrer chaque tubes comme pour la
  segmenation? et si oui les deux tubes prendront deux
  labels, soit normal-normal ou normal-anormal, maitenant
  le troisieme label je le mets ou? car il faut forcement
  l'information sur les pairs. Bon dit moi


• Tu n’as pas besoin d’annoter un label pair dans Roboflow.

  C’est ça le point clé.

  Pour un dataset siamese dans votre cas, le plus simple
  est:

  - Roboflow sert seulement à localiser les tubes
  - la notion de paire normale / paire anormale se crée
    après, dans un script

  Donc en pratique:

  ## Ce que tu annotes dans Roboflow

  Tu annotes comme pour la segmentation:

  - tube_ref
  - tube_test

  C’est tout.

  Pas:

  - normal
  - anomaly
  - pair

  Pourquoi:

  - normal / anomaly décrit surtout l’état du tube_test
  - pair décrit la relation entre deux ROI
  - Roboflow est fait pour annoter des objets dans une
    image, pas une relation logique entre deux objets

  ## Ce qu’il faut faire concrètement

  Tu crées deux dossiers de vidéos, comme pour la
  calibration:

  data/siamese_source/
    normal_pairs_videos/
    anomaly_pairs_videos/

  Signification:

  - normal_pairs_videos/:
      - tube_ref normal
      - tube_test normal

  - anomaly_pairs_videos/:
      - tube_ref normal
      - tube_test defectueux

  Ensuite:

  1. on extrait des frames
  2. le modèle de segmentation récupère tube_ref et
     tube_test

  3. on crop les deux ROI
  4. on crée une ligne de dataset:
      - ref_path
      - test_path
      - label=0 si vidéo normale
      - label=1 si vidéo anormale

  Donc le label de paire ne se met pas dans Roboflow.
  Il vient du dossier source de la vidéo.

  ## Réponse directe à ta question sur les labels

  Dans Roboflow, utilise exactement:

  - tube_ref
  - tube_test

  Et oui:

  - tu encadres / segmentes les deux tubes
  - chacun garde son label propre
  - pas de normal-normal
  - pas de normal-anormal
  - pas de pair dans l’annotation visuelle

  Le label de paire est stocké ailleurs, par exemple dans un
  CSV final:

  ref_path,test_path,label,video_name,frame_idx
  .../ref_001.jpg,.../test_001.jpg,0,video_normal_01,15
  .../ref_002.jpg,.../test_002.jpg,1,video_anomaly_03,42

  ## Donc la bonne méthode pour toi

  Fais simplement:

  - un dossier de vidéos normal_pairs_videos
  - un dossier de vidéos anomaly_pairs_videos

  Puis, si tu veux, oui je peux m’occuper du reste:

  - extraction des frames
  - segmentation automatique avec vos modèles existants
  - récupération des ROI
  - création du dataset siamese exploitable

  ## Résumé final

  Tu ne dois pas annoter 3 labels.

  Tu dois annoter seulement:

  - tube_ref
  - tube_test

  Et la paire:

  - 0 = normal pair
  - 1 = anomaly pair

  sera générée automatiquement à partir du dossier de
  provenance de la vidéo.

  Si tu veux, l’étape suivante utile est que je te définisse
  l’arborescence exacte à créer dans data/ pour que je
  puisse ensuite te coder le pipeline de génération du
  dataset siamese.



MODEL_PATH=./src/models/best_model.pt uvicorn src.inference.api:app --host 0.0.0.0 --port 8002 --reload
