"use client";

import { FiActivity, FiAlertTriangle, FiCheck, FiDownload, FiFileText, FiFilm, FiUploadCloud } from "react-icons/fi";
import { AppHeader } from "@/components/csp/app-header";
import { TemperatureSection } from "@/components/csp/temperature-section";
import { Button, ButtonLink } from "@/components/ui/button";
import { useCspAnalysis } from "@/hooks/use-csp-analysis";
import { formatFileSize } from "@/lib/thermal";
import { InspectionResult } from "@/components/csp/inspection-result";
import { inspectionMetadataComplete } from "@/lib/inspection";
import "./inspection.css";

export default function HomePage() {
  const csp = useCspAnalysis();
  const busy = csp.running || csp.generatingReport;
  const summary = csp.hasAnomalyAnalysis ? csp.aiSummary : null;
  const ready = summary?.level != null;
  const mediaUrl = csp.outUrl || csp.videoPreviewUrl;
  const operatorReady = inspectionMetadataComplete(csp.inspection);
  const steps = [operatorReady, !!csp.videoFile, csp.allTemperaturesReady && !!csp.inspection.visualAnomaly];

  return (
    <main className="inspection-app">
      <AppHeader />
      <div className="inspection-intro">
        <div><span className="eyebrow">ESPACE D’INSPECTION</span><h1>Inspection thermique des tubes</h1><p>Préparez vos données, puis lancez une analyse complète.</p></div>
        <span className={`connection-pill ${csp.backendStatus.reachable && csp.backendStatus.fusionLoaded && csp.backendStatus.modelLoaded ? "connected" : ""}`}>
          <i />{csp.backendStatus.checking ? "Connexion en cours" : csp.backendStatus.reachable && csp.backendStatus.fusionLoaded && csp.backendStatus.modelLoaded ? "Analyse IA disponible" : "Analyse IA indisponible"}
        </span>
      </div>
      <form className="inspection-grid" onSubmit={event => { event.preventDefault(); if (csp.canAnalyze) csp.handleAnomalyAnalysisAction(); }}>
        <div className="inspection-inputs">
          <section className="inspection-panel" aria-labelledby="operator-title">
            <div className="panel-heading"><span className={`step-number ${steps[0] ? "complete" : ""}`}>{steps[0] ? <FiCheck /> : "01"}</span><div><h2 id="operator-title">Informations opérateur</h2><p>Identifiez cette inspection.</p></div><span className="required-note">Nom et numéro de série requis</span></div>
            <fieldset disabled={busy} className="operator-fields">
              <label>Nom de l’opérateur<input required autoComplete="name" maxLength={100} placeholder="Ex. Jean Dupont" value={csp.inspection.operator} onChange={e => csp.updateInspection("operator", e.target.value)} /></label>
              <label>Numéro de série du tube<input required maxLength={160} placeholder="Ex. CSP-2026-001" value={csp.inspection.tubeSerial} onChange={e => csp.updateInspection("tubeSerial", e.target.value)} /></label>
              <label>Date<input aria-describedby="automatic-timestamp" type="date" value={csp.inspection.date} onChange={e => csp.updateInspection("date", e.target.value)} /></label>
              <label>Heure<input aria-describedby="automatic-timestamp" type="time" value={csp.inspection.time} onChange={e => csp.updateInspection("time", e.target.value)} /></label>
            </fieldset>
            <p id="automatic-timestamp" className="timestamp-hint">Date et heure facultatives : si elles sont laissées vides, elles seront renseignées automatiquement au lancement de l’analyse.</p>
          </section>

          <section className="inspection-panel" id="source-video" aria-labelledby="source-title">
            <div className="panel-heading"><span className={`step-number ${steps[1] ? "complete" : ""}`}>{steps[1] ? <FiCheck /> : "02"}</span><div><h2 id="source-title">Source vidéo</h2><p>La vidéo sélectionnée s’affiche à droite.</p></div></div>
            <label className={`inspection-upload ${csp.videoFile ? "selected" : ""}`}>
              <input ref={csp.fileInputRef} type="file" accept="video/*" disabled={busy} onChange={csp.handleVideoChange} aria-label="Téléverser une vidéo thermique" />
              <FiUploadCloud aria-hidden="true" /><strong>{csp.videoFile ? csp.videoFile.name : "Téléverser une vidéo"}</strong>
              <span>{csp.videoFile ? `${formatFileSize(csp.videoFile.size)} · Cliquer pour remplacer` : "Cliquez pour sélectionner un fichier vidéo"}</span>
            </label>
          </section>

          <section className="inspection-panel measures-panel" aria-labelledby="measures-title">
            <div className="panel-heading"><span className={`step-number ${steps[2] ? "complete" : ""}`}>{steps[2] ? <FiCheck /> : "03"}</span><div><h2 id="measures-title">Données thermiques</h2><p>Au moins quatre mesures en °C pour chaque tube.</p></div></div>
            <TemperatureSection disabled={busy} refStats={csp.refStats} testStats={csp.testStats} tubeRefTemps={csp.tubeRefTemps} tubeTestTemps={csp.tubeTestTemps} onAddTemperatureField={csp.addTemperatureField} onTemperatureChange={csp.updateTemperature} />
            <fieldset disabled={busy} className="visual-observation">
              <legend>Voyez-vous une anomalie sur la vidéo ?</legend>
              <div className="observation-options">{([['yes', 'Oui'], ['no', 'Non']] as const).map(([value, label]) => <label key={value} className={csp.inspection.visualAnomaly === value ? "checked" : ""}><input type="radio" name="visualAnomaly" required value={value} checked={csp.inspection.visualAnomaly === value} onChange={e => csp.updateInspection("visualAnomaly", e.target.value)} />{label}</label>)}</div>
              <p>Votre observation accompagne l’inspection. Elle ne modifie pas la prédiction du modèle.</p>
            </fieldset>
          </section>
        </div>

        <section className="inspection-panel inspection-output" aria-labelledby="analysis-title">
          <div className="analysis-heading"><div><span className="eyebrow">VIDÉO & RÉSULTATS</span><h2 id="analysis-title">Analyse d’anomalies</h2></div><Button variant="dark" type="submit" disabled={!csp.canAnalyze} aria-describedby="analysis-requirements"><FiActivity aria-hidden="true" />{csp.running ? "Analyse en cours…" : "Lancer l’analyse"}</Button></div>
          <p id="analysis-requirements" className="analysis-requirements">{csp.running ? "Segmentation des tubes et classification en cours." : csp.canAnalyze ? "Tout est prêt. Vous pouvez lancer l’analyse." : "Complétez les informations, la vidéo, les températures et votre observation pour commencer."}</p>
          <div className="inspection-video video-stage">
            <canvas ref={csp.canvasRef} className={`processing-canvas ${csp.running ? "is-active" : ""}`} hidden={!csp.running} aria-label="Segmentation de la vidéo en cours" />
            {!csp.running && mediaUrl ? <video key={mediaUrl} className="source-preview" controls playsInline preload="metadata" src={mediaUrl} aria-label={csp.outUrl ? "Vidéo avec segmentation des tubes" : "Aperçu de la vidéo source"} /> : !csp.running && <div className="video-placeholder"><span><FiFilm /></span><strong>Votre vidéo apparaîtra ici</strong><p>Sélectionnez une vidéo thermique pour commencer.</p></div>}
          </div>
          <div className="video-caption"><span><i className="ref-key" />Tube référence <i className="test-key" />Tube test</span>{csp.outUrl ? <a href={csp.outUrl} download="video-annotee-csp.webm"><FiDownload /> Vidéo annotée</a> : <span>{csp.running ? csp.step : "Aperçu de l’inspection"}</span>}</div>
          {csp.running && <div className="inspection-progress" role="status"><div><span>{csp.step}</span><strong>{csp.progress}%</strong></div><progress value={csp.progress} max={100} /><small>{csp.predictionCount} images traitées</small></div>}
          {csp.error && <p role="alert" className="inspection-error"><FiAlertTriangle />{csp.error}</p>}

          <div className="inspection-results" aria-live="polite" aria-busy={csp.running}>
            <div className="results-title"><h3>Résultats de l’analyse</h3>{!ready && <span className="result-tag">En attente</span>}</div>
            {ready ? <InspectionResult level={summary.level} /> : <div className="results-placeholder"><FiActivity /><div><strong>{csp.running ? "Analyse de votre inspection…" : "Prêt à examiner vos tubes"}</strong><p>L’état du tube et l’interprétation s’afficheront ici après l’analyse.</p></div></div>}
          </div>
          <div className="inspection-pdf-action"><Button variant="dark" type="button" disabled={!csp.canGenerateReport} onClick={csp.handleReportAction}><FiFileText />{csp.generatingReport ? "Génération…" : "Générer PDF"}</Button></div>
          {csp.reportUrl && <div className="inspection-report"><ButtonLink href={csp.reportUrl} download="rapport_thermique_csp.pdf" variant="outline"><FiDownload />Télécharger le rapport</ButtonLink><iframe src={csp.reportUrl} title="Aperçu du rapport d’inspection" /></div>}
          {(csp.generatingReport || csp.reportStatus.startsWith("Erreur")) && <p role="status">{csp.reportStatus}</p>}
        </section>
      </form>
      <footer className="inspection-footer">© Copyright Green Energy Park. Tous droits réservés 2026.</footer>
      <video ref={csp.hiddenVideoRef} className="hidden-video" />
    </main>
  );
}
