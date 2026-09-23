# Frontend — CSP Thermal Intelligence

Interface d’inspection thermique personnalisée par **[Salma Kamal](https://github.com/SALMAKAMAL21)** pour le projet **Green Energy Park**.

**Encadrant : Amine Moulay Taj**  
**Période du stage : du 09/03/2026 au 09/09/2026**

Le frontend utilise **Next.js 14**, **React 18** et **TypeScript**. Il permet de préparer une inspection, de visualiser la segmentation des tubes, de consulter le diagnostic et de générer un rapport PDF.

Consulter le [README principal](../README.md) pour installer le backend, les modèles et les dépendances Python.

## Installation et lancement

Depuis le dossier `frontend`, dans PowerShell :

```powershell
npm.cmd ci
if (-not (Test-Path .env.local)) {
    Copy-Item .env.example .env.local
}
npm.cmd run dev
```

L’application est accessible sur [http://localhost:3000](http://localhost:3000). Le backend FastAPI doit être lancé séparément sur le port `8002`.

Configuration de `.env.local` :

```env
SEGMENTATION_API_URL=http://127.0.0.1:8002
```

Redémarrer le serveur Next.js après une modification de cette configuration.

Pour compiler puis démarrer la version de production, arrêter le serveur de développement et exécuter :

```powershell
npm.cmd run build
npm.cmd run start
```

## Parcours opérateur

1. Saisir le nom de l’opérateur et le numéro de série du tube.
2. Importer une vidéo thermique.
3. Renseigner au moins quatre mesures pour chaque tube et l’observation visuelle.
4. Lancer l’analyse, puis consulter les annotations et l’interprétation.
5. Générer et télécharger `rapport_thermique_csp.pdf`.

La date et l’heure sont complétées automatiquement au lancement si elles sont laissées vides. L’analyse attend que les autres données requises soient complètes. Une modification des données invalide le diagnostic précédent.

## Fichiers principaux

| Fichier | Rôle |
|---|---|
| [app/page.tsx](app/page.tsx) | Formulaire d’inspection, vidéo, résultats et téléchargement du rapport |
| [app/inspection.css](app/inspection.css) | Mise en page de l’interface d’inspection |
| [hooks/use-csp-analysis.ts](hooks/use-csp-analysis.ts) | Lecture vidéo, appels API, cache des détections et exports |
| [lib/inspection.ts](lib/inspection.ts) | Validation des informations et date/heure automatiques |
| [lib/thermal.ts](lib/thermal.ts) | Mesures thermiques et agrégation des prédictions |
| [lib/inspection-results.ts](lib/inspection-results.ts) | Libellés, constats et interprétations des diagnostics |
| [lib/video-frame.ts](lib/video-frame.ts) | Extraction d’une image à un instant précis de la vidéo |
| [lib/fusion-report.ts](lib/fusion-report.ts) | Composition du rapport PDF avec jsPDF |
| [lib/backend.ts](lib/backend.ts) | Adresse et appels du backend FastAPI |
| [lib/api-response.ts](lib/api-response.ts) | Lecture et validation des réponses JSON dans le navigateur |

## Liaison avec le backend

Le navigateur passe par les routes du serveur Next.js :

| Route Next.js | Route FastAPI | Usage |
|---|---|---|
| `GET /api/segment-check` | `/health` et `/model/info` | État du backend et des modèles |
| `POST /api/segment-predict` | `/predict` | Segmentation et classification d’une image |
| `POST /api/segment-predict` avec `detections_json` | `/analyze-anomaly` | Classification utilisant les détections déjà calculées |

Le traitement échantillonne une image par seconde. Les annotations affichent les classes renvoyées par YOLO. La vidéo annotée est enregistrée au format WebM dans le navigateur avec `MediaRecorder`.

## Export du rapport

Le rapport utilise une image extraite au milieu de la vidéo source, avec ses annotations. L’image complète garde ses proportions pour préserver la visibilité des tubes.

Le tableau affiche les températures, les écarts absolus `|T_ref - T_test|` et leur moyenne. Le modèle de classification utilise l’écart **signé** entre les moyennes des deux tubes. Le diagnostic et l’interprétation proviennent du résultat de l’analyse.

La mise en page garde les rapports usuels de quatre à six mesures sur une page et prévoit une pagination pour les séries plus longues.

## Fichiers générés

- `.next-dev/` : fichiers générés par le serveur de développement.
- `.next/` : compilation de production.
- `node_modules/` : dépendances installées par npm.

Ces dossiers sont exclus du dépôt par `.gitignore`. `npm.cmd` permet de lancer npm sous Windows sans utiliser le script PowerShell `npm.ps1`.
