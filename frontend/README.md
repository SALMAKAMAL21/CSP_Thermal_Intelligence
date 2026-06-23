# Frontend Next.js - Green Energy Park

Ce frontend sert à tester ton modèle de segmentation sur:
- images RGB/thermiques
- vidéos RGB/thermiques

Objectif métier:
- détecter les tubes recepteurs
- afficher les labels `tube_ref` et `tube_test` renvoyés par le backend
- saisir les températures associées à chaque label

---

## 1) Vue globale (Frontend + Backend)

### Frontend (Next.js)
Rôle:
- uploader une vidéo
- extraire des frames vidéo
- appeler l'API IA
- afficher les annotations
- générer une vidéo annotée de sortie
- générer un rapport PDF à partir des résultats AI et des températures

Fichier principal:
- `frontend/app/page.tsx`

### Backend (FastAPI)
Rôle:
- charger le modèle YOLO (`best_model.pt` prioritaire)
- recevoir une image (`/predict`)
- lancer l'inférence
- renvoyer détections (bbox, masques, score)
- appliquer la règle métier haut/bas pour stabiliser `tube_ref`/`tube_test`

Fichier principal:
- `src/inference/api.py`

### Liaison Frontend ↔ Backend
Le frontend n'appelle pas directement l'API Python. Il passe par 2 routes Next.js:
- `GET /api/segment-check` → proxy vers `{SEGMENTATION_API_URL}/health`
- `POST /api/segment-predict` → proxy vers `{SEGMENTATION_API_URL}/predict`

Pourquoi:
- centraliser la config backend (`.env.local`)
- gérer proprement les erreurs
- éviter CORS compliqué côté navigateur

---

## 2) Flux Image (simple)

1. Tu upload une image.
2. Le frontend envoie l'image à `/api/segment-predict`.
3. La route Next.js forwarde au backend FastAPI `/predict`.
4. Le backend retourne les détections.
5. Le frontend dessine les bbox/polygones sur un canvas.

Ce flux sert de debug pur du modèle (sans complexité vidéo).

---

## 3) Flux Vidéo (réel)

1. Tu upload une vidéo.
2. Le frontend lit la vidéo cachée (`<video>` interne).
3. Il échantillonne des frames (sampling).
4. Chaque frame est envoyée à `/api/segment-predict`.
5. Le frontend stocke les détections par instant `t`.
6. Stabilisation temporelle: si une frame a 0 détection, on reprend la détection valide la plus proche (avant/après).
7. Rendu final: le frontend rejoue la vidéo sur un canvas et dessine les annotations correspondantes.
8. Export: enregistrement via `MediaRecorder` → vidéo `.webm` annotée.

Pourquoi tu voyais des trous au milieu:
- certaines frames thermiques avaient confiance plus faible
- détections intermittentes
- maintenant partiellement compensé par la stabilisation temporelle

---

## 4) Logique métier des classes

Les deux tubes se ressemblent visuellement.
Le backend applique une convention métier stable après détection:
- tube le plus haut = `tube_ref`
- tube le plus bas = `tube_test`

Cette logique ne doit pas être forcée côté frontend: l'interface affiche seulement les labels renvoyés par l'API et collecte les températures associées.

Conséquence positive:
- moins d'inversions `ref/test` dans la vidéo
- cohérence métier même si le modèle hésite sur la classe brute

---

## 5) Paramètre `conf` (seuil de confiance)

`conf` = confiance minimale pour garder une détection.

- `conf` haut (ex 0.30): moins de faux positifs, mais plus de ratés
- `conf` plus bas (ex 0.18–0.22): récupère plus de tubes difficiles (utile en thermique)

Dans tes tests, baisser légèrement `conf` aide sur les frames du milieu thermique.

---

## 6) Modèle chargé par le backend

Ordre de priorité actuel:
1. variable d'environnement `MODEL_PATH` (si définie)
2. `ml/best_model.pt`
3. `ml/yolo_seg.pt`
4. autres fallbacks

Donc si tu as placé le modèle dans `ml/best_model.pt`, il sera utilisé automatiquement.

---

## 7) Commandes de démarrage

### Backend
```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8002 --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

`.env.local` (frontend):
```env
SEGMENTATION_API_URL=http://127.0.0.1:8002
```

---

## 8) Fichiers importants

- `frontend/app/page.tsx` : logique UI + pipeline image/vidéo
- `frontend/app/api/segment-predict/route.ts` : proxy prédiction
- `frontend/app/api/segment-check/route.ts` : proxy health
- `src/inference/api.py` : backend IA FastAPI

---

## 9) Limites actuelles

- vidéos thermiques: encore quelques cas difficiles (contraste variable, texture faible)
- la post-règle backend haut/bas suppose que les deux tubes restent visibles et ordonnés verticalement

Amélioration future recommandée:
- ajouter plus de frames thermiques "milieu de séquence" dans le dataset d'entraînement
