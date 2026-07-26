import type { ChangeEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createRefTemperatures,
  createTestTemperatures,
  formatAnomalyScore,
  formatAverage,
  formatTime,
  getTemperaturePointLabel,
  getTemperatureStats,
  INITIAL_BACKEND_STATUS,
  pickColor,
  REQUIRED_TEMPERATURE_COUNT,
  REPORT_WAITING_STATUS,
  confidenceLabel,
  severityLabel,
  summarizePredictions,
  TARGET_LABELS,
  videoDecisionLabel,
  type AiSummary,
  type AutoencoderAnomaly,
  type CombinedAnomaly,
  type BackendConnectionStatus,
  type Detection,
  type FramePrediction,
  type PredictResponse,
  type ThermalAnomaly
} from "@/lib/thermal";

export function useCspAnalysis() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState("En attente");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [outUrl, setOutUrl] = useState<string | null>(null);
  const [predictionCount, setPredictionCount] = useState(0);
  const [tubeRefTemps, setTubeRefTemps] = useState<string[]>(createRefTemperatures);
  const [tubeTestTemps, setTubeTestTemps] = useState<string[]>(createTestTemperatures);
  const [aiSummary, setAiSummary] = useState<AiSummary | null>(null);
  const [hasAnomalyAnalysis, setHasAnomalyAnalysis] = useState(false);
  const [backendStatus, setBackendStatus] = useState<BackendConnectionStatus>(INITIAL_BACKEND_STATUS);
  const [reportUrl, setReportUrl] = useState<string | null>(null);
  const [reportStatus, setReportStatus] = useState(REPORT_WAITING_STATUS);
  const [generatingReport, setGeneratingReport] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const hiddenVideoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const refStats = useMemo(() => getTemperatureStats(tubeRefTemps), [tubeRefTemps]);
  const testStats = useMemo(() => getTemperatureStats(tubeTestTemps), [tubeTestTemps]);
  const allTemperaturesReady =
    refStats.completed >= REQUIRED_TEMPERATURE_COUNT && testStats.completed >= REQUIRED_TEMPERATURE_COUNT;
  const canGenerateReport = !!outUrl && !!aiSummary && hasAnomalyAnalysis && allTemperaturesReady && !generatingReport;

  const refreshBackendStatus = useCallback(async () => {
    setBackendStatus((current) => ({ ...current, checking: true }));

    try {
      const res = await fetch("/api/segment-check", { cache: "no-store" });
      const data = (await res.json()) as Omit<BackendConnectionStatus, "checking">;
      const nextStatus = { ...data, checking: false };
      setBackendStatus(nextStatus);
      return nextStatus;
    } catch (event) {
      const details = event instanceof Error ? event.message : "Connexion backend impossible";
      const nextStatus: BackendConnectionStatus = {
        checking: false,
        details,
        modelLoaded: false,
        reachable: false
      };
      setBackendStatus(nextStatus);
      return nextStatus;
    }
  }, []);

  useEffect(() => {
    if (!videoFile) {
      setVideoPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(videoFile);
    setVideoPreviewUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  useEffect(() => {
    return () => {
      if (outUrl) URL.revokeObjectURL(outUrl);
    };
  }, [outUrl]);

  useEffect(() => {
    return () => {
      if (reportUrl) URL.revokeObjectURL(reportUrl);
    };
  }, [reportUrl]);

  useEffect(() => {
    void refreshBackendStatus();
  }, [refreshBackendStatus]);

  function handleVideoChange(event: ChangeEvent<HTMLInputElement>) {
    setVideoFile(event.target.files?.[0] ?? null);
    setOutUrl(null);
    setProgress(0);
    setPredictionCount(0);
    setStep("En attente");
    setError(null);
    setAiSummary(null);
    setHasAnomalyAnalysis(false);
    setReportUrl(null);
    setReportStatus(REPORT_WAITING_STATUS);
  }

  function updateTemperature(tube: "ref" | "test", index: number, value: string) {
    const setter = tube === "ref" ? setTubeRefTemps : setTubeTestTemps;
    setter((current) => current.map((item, itemIndex) => (itemIndex === index ? value : item)));
    setReportStatus(outUrl && aiSummary ? "Rapport à régénérer avec les nouvelles températures." : REPORT_WAITING_STATUS);
    setReportUrl(null);
  }

  function addTemperatureField() {
    setTubeRefTemps((current) => [...current, ""]);
    setTubeTestTemps((current) => [...current, ""]);
    setReportStatus(outUrl && aiSummary ? "Rapport à régénérer après ajout d'un champ thermique." : REPORT_WAITING_STATUS);
    setReportUrl(null);
  }

  async function predictFrame(
    blob: Blob,
    idx: number,
    includeAnomaly: boolean,
    sessionId: string,
    resetTracker: boolean
  ): Promise<{ detections: Detection[]; anomaly: ThermalAnomaly | null; autoencoder: AutoencoderAnomaly | null; combined: CombinedAnomaly | null }> {
    const fd = new FormData();
    fd.append("image", new File([blob], `frame-${idx}.jpg`, { type: "image/jpeg" }));
    fd.append("conf", "0.25");
    fd.append("iou", "0.45");
    fd.append("analyze_thermal", includeAnomaly ? "true" : "false");
    fd.append("session_id", sessionId);
    fd.append("reset_tracker", resetTracker ? "true" : "false");

    const res = await fetch("/api/segment-predict", { method: "POST", body: fd });
    const data = (await res.json()) as PredictResponse;
    if (!res.ok || data.error) {
      const details =
        data.backendStatus || data.backendUrl
          ? ` [backendStatus=${data.backendStatus ?? "?"} backendUrl=${data.backendUrl ?? "?"}]`
          : "";
      const backendDetails = data.details ? ` ${data.details}` : "";
      const backendPayload = data.backendResponse ? ` payload=${JSON.stringify(data.backendResponse)}` : "";
      throw new Error((data.error || "Erreur de prédiction") + backendDetails + details + backendPayload);
    }

    return {
      detections: (data.detections || []).filter((d) => TARGET_LABELS.has((d.class_name || "").toLowerCase())),
      anomaly: includeAnomaly ? data.thermal_anomaly || null : null,
      autoencoder: includeAnomaly ? data.autoencoder_anomaly || null : null,
      combined: includeAnomaly ? data.combined_anomaly || null : null
    };
  }

  function drawDetections(ctx: CanvasRenderingContext2D, detections: Detection[]) {
    for (const det of detections) {
      const label = (det.class_name || "unknown").toLowerCase();
      const color = pickColor(label);

      //if (Array.isArray(det.mask_polygon) && det.mask_polygon.length > 2) {
      // ctx.beginPath();
      //  det.mask_polygon.forEach((p, i) => {
      //    if (i === 0) ctx.moveTo(p[0], p[1]);
      //    else ctx.lineTo(p[0], p[1]);
      //  });
      //  ctx.closePath();
      //  ctx.strokeStyle = color;
      //  ctx.lineWidth = 3;
      //  ctx.stroke();
      //  ctx.fillStyle = `${color}24`;
      //  ctx.fill();
      //} 
       if (Array.isArray(det.bbox) && det.bbox.length === 4) {
        const [x1, y1, x2, y2] = det.bbox;
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }

      const [rawX, rawY] = Array.isArray(det.bbox) && det.bbox.length === 4 ? [det.bbox[0], det.bbox[1]] : [20, 32];
      const x = Math.min(Math.max(0, rawX), Math.max(0, ctx.canvas.width - 126));
      const y = Math.max(0, rawY);
      const tag = `${label} ${(det.confidence || 0).toFixed(2)}`;
      ctx.font = "700 14px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      const tagWidth = Math.min(Math.max(ctx.measureText(tag).width + 20, 118), Math.max(118, ctx.canvas.width - x - 8));
      const tagY = Math.max(0, y - 28);

      ctx.fillStyle = "rgba(0, 0, 0, 0.78)";
      ctx.fillRect(x, tagY, tagWidth, 26);
      ctx.strokeStyle = color;
      ctx.strokeRect(x + 0.5, tagY + 0.5, tagWidth - 1, 25);
      ctx.fillStyle = color;
      ctx.fillText(tag, x + 10, Math.max(18, tagY + 18));
    }
  }

  async function annotateVideo(includeAnomaly: boolean) {
    if (!videoFile) return;
    setRunning(true);
    setError(null);
    setOutUrl(null);
    setProgress(0);
    setPredictionCount(0);
    setAiSummary(null);
    setHasAnomalyAnalysis(false);
    setReportUrl(null);
    setReportStatus(REPORT_WAITING_STATUS);
    setStep("Connexion backend IA");

    const backend = await refreshBackendStatus();
    if (!backend.reachable) {
      setError(`Backend IA hors ligne${backend.backendUrl ? ` (${backend.backendUrl})` : ""}. ${backend.details ?? ""}`.trim());
      setStep("Backend hors ligne");
      setRunning(false);
      return;
    }

    if (!backend.modelLoaded) {
      setError(`Backend connecté, mais modèle IA non chargé${backend.backendUrl ? ` (${backend.backendUrl})` : ""}. Vérifiez MODEL_PATH et le démarrage FastAPI.`);
      setStep("Modèle IA indisponible");
      setRunning(false);
      return;
    }

    const video = hiddenVideoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) {
      setError("Composants vidéo/canvas indisponibles");
      setRunning(false);
      return;
    }

    const sourceUrl = URL.createObjectURL(videoFile);
    video.src = sourceUrl;
    video.muted = true;
    video.playsInline = true;

    try {
      setStep("Chargement vidéo");
      await new Promise<void>((resolve, reject) => {
        video.onloadedmetadata = () => resolve();
        video.onerror = () => reject(new Error("Lecture vidéo impossible"));
      });

      const duration = Math.max(video.duration || 0, 1);
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas context indisponible");

      setStep(includeAnomaly ? "Segmentation + analyse" : "Segmentation YOLO");
      const samplingFps = 4;
      const sessionId =
        typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
          ? crypto.randomUUID()
          : `session-${Date.now()}`;
      const sampleTimes: number[] = [];
      for (let t = 0; t < duration; t += 1 / samplingFps) sampleTimes.push(t);
      if (sampleTimes[sampleTimes.length - 1] !== duration) sampleTimes.push(duration);

      const predictions: FramePrediction[] = [];

      for (let i = 0; i < sampleTimes.length; i++) {
        const t = sampleTimes[i];
        await new Promise<void>((resolve) => {
          const onSeeked = () => {
            video.removeEventListener("seeked", onSeeked);
            resolve();
          };
          video.addEventListener("seeked", onSeeked);
          video.currentTime = t;
        });

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob((b) => resolve(b), "image/jpeg", 0.9));
        if (!blob) continue;

        const frameResult = await predictFrame(blob, i + 1, includeAnomaly, sessionId, i === 0);
        predictions.push({
          t,
          detections: frameResult.detections,
          anomaly: frameResult.anomaly,
          autoencoder: frameResult.autoencoder,
          combined: frameResult.combined
        });
        setPredictionCount(predictions.length);
        setStep(includeAnomaly ? "Segmentation + anomalies" : "Segmentation YOLO");
        setProgress(Math.round(((i + 1) / sampleTimes.length) * 55));
      }

      const summary = summarizePredictions(predictions);
      setAiSummary(summary);
      setHasAnomalyAnalysis(includeAnomaly);

      setStep("Rendu vidéo");
      video.currentTime = 0;

      const stream = canvas.captureStream(30);
      const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
      const recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 3_000_000 });
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      };

      const done = new Promise<Blob>((resolve) => {
        recorder.onstop = () => resolve(new Blob(chunks, { type: mime }));
      });

      const getNearest = (time: number) => {
        let nearest = predictions[0];
        let best = Infinity;
        for (const prediction of predictions) {
          const distance = Math.abs(prediction.t - time);
          if (distance < best) {
            best = distance;
            nearest = prediction;
          }
        }
        return nearest?.detections || [];
      };

      video.playbackRate = 0.5;
      recorder.start();
      await video.play();

      await new Promise<void>((resolve) => {
        const render = () => {
          if (video.paused || video.ended) {
            resolve();
            return;
          }

          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          drawDetections(ctx, getNearest(video.currentTime));

          const nextProgress = 55 + Math.round((video.currentTime / duration) * 45);
          setProgress(Math.min(100, nextProgress));
          requestAnimationFrame(render);
        };
        requestAnimationFrame(render);
      });

      recorder.stop();
      const outBlob = await done;
      video.playbackRate = 1.0;
      const annotatedUrl = URL.createObjectURL(outBlob);
      setOutUrl(annotatedUrl);
      setProgress(100);
      setStep("Terminé");
      setReportStatus(
        includeAnomaly
          ? allTemperaturesReady
            ? `Décision vidéo ${videoDecisionLabel(summary.decision)}. Score max ${formatAnomalyScore(summary.maxAnomalyScore)}, persistance ${summary.longestSuspectRun} frame(s). Rapport PDF prêt à générer.`
            : `Décision vidéo ${videoDecisionLabel(summary.decision)}. Score max ${formatAnomalyScore(summary.maxAnomalyScore)}, persistance ${summary.longestSuspectRun} frame(s). Compléter les 8 températures pour générer le rapport PDF.`
          : allTemperaturesReady
            ? "Segmentation terminée. Lancez l'analyse pour calculer les anomalies, puis générez le rapport PDF."
            : "Segmentation terminée. Lancez l'analyse puis complétez les 8 températures pour générer le rapport PDF."
      );
    } catch (event) {
      const message = event instanceof Error ? event.message : "Erreur d'annotation vidéo";
      setError(message);
      setStep("Erreur");
    } finally {
      URL.revokeObjectURL(sourceUrl);
      setRunning(false);
    }
  }

  async function generateReportPdf() {
    if (!canGenerateReport || !aiSummary) return;

    setGeneratingReport(true);
    setReportStatus("Génération du rapport PDF...");

    try {
      const { jsPDF } = await import("jspdf");
      const doc = new jsPDF({ unit: "mm", format: "a4" });
      const createdAt = new Date();
      const generatedAt = createdAt.toLocaleString("fr-FR", {
        dateStyle: "medium",
        timeStyle: "short"
      });

      doc.setProperties({
        title: "Rapport analyse thermique CSP",
        subject: "Résultats AI et températures tubes CSP"
      });

      const accent = [248, 132, 47] as const;
      const text = [24, 24, 27] as const;
      const muted = [105, 105, 112] as const;
      const border = [232, 232, 235] as const;

      const sectionTitle = (label: string, y: number) => {
        doc.setTextColor(...text);
        doc.setFont("helvetica", "bold");
        doc.setFontSize(15);
        doc.text(label, 16, y);
        doc.setDrawColor(...accent);
        doc.setLineWidth(0.35);
        doc.line(16, y + 7, 194, y + 7);
      };

      const metric = (label: string, value: string, x: number, y: number, w = 52) => {
        doc.setDrawColor(...border);
        doc.setFillColor(250, 250, 251);
        doc.roundedRect(x, y, w, 22, 3, 3, "FD");
        doc.setTextColor(...muted);
        doc.setFont("helvetica", "bold");
        doc.setFontSize(7);
        doc.text(label.toUpperCase(), x + 4, y + 7);
        doc.setTextColor(...text);
        doc.setFontSize(13);
        doc.text(value, x + 4, y + 16);
      };

      const buildInterpretation = () => {
        const decisionText =
          aiSummary.decision === "anomaly"
            ? "La vidéo présente une anomalie probable sur le tube test."
            : aiSummary.decision === "warning"
              ? "La vidéo présente un comportement atypique à surveiller sur le tube test."
              : "La vidéo reste globalement compatible avec un comportement normal du tube test.";

        const thermalText =
          aiSummary.maxAnomalyScore >= 0.42
            ? `La comparaison thermique a relevé un écart marqué entre tube_ref et tube_test, avec un score maximal de ${formatAnomalyScore(aiSummary.maxAnomalyScore)} et une persistance de ${aiSummary.longestSuspectRun} frame(s).`
            : aiSummary.maxAnomalyScore >= 0.26
              ? `La comparaison thermique a détecté un écart modéré entre tube_ref et tube_test, avec un score maximal de ${formatAnomalyScore(aiSummary.maxAnomalyScore)}.`
              : `La comparaison thermique n'a pas mis en évidence d'écart fort entre tube_ref et tube_test, le score maximal restant à ${formatAnomalyScore(aiSummary.maxAnomalyScore)}.`;

        const autoencoderText =
          aiSummary.autoencoderPeakRatio >= 1.25
            ? `L'autoencoder renforce l'hypothèse d'anomalie: le tube test apparaît significativement plus atypique que le tube de référence (ratio maximal ${formatAnomalyScore(aiSummary.autoencoderPeakRatio)}).`
            : aiSummary.autoencoderPeakRatio >= 1.1
              ? `L'autoencoder apporte un signal complémentaire modéré: le tube test est légèrement plus atypique que le tube de référence (ratio maximal ${formatAnomalyScore(aiSummary.autoencoderPeakRatio)}).`
              : `L'autoencoder n'apporte pas de confirmation forte: le tube test reste proche du tube de référence du point de vue reconstruction (ratio maximal ${formatAnomalyScore(aiSummary.autoencoderPeakRatio)}).`;

        const synthesisText =
          aiSummary.decision === "anomaly" && aiSummary.autoencoderPeakRatio >= 1.1
            ? "La convergence entre comparaison thermique et autoencoder renforce la crédibilité de la détection."
            : aiSummary.decision === "anomaly"
              ? "L'alerte est principalement portée par la comparaison thermique; une vérification visuelle des frames au pic est recommandée."
              : aiSummary.decision === "warning"
                ? "Le résultat doit être interprété comme une alerte prudente, utile pour orienter une inspection manuelle ciblée."
                : "L'absence de convergence forte entre les indicateurs est cohérente avec une vidéo normale ou faiblement contrastée.";

        return [decisionText, thermalText, autoencoderText, synthesisText];
      };

      doc.setFillColor(8, 8, 10);
      doc.rect(0, 0, 210, 52, "F");
      doc.setTextColor(...accent);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(12);
      doc.text("Green Energy Park", 16, 18);
      doc.setTextColor(255, 255, 255);
      doc.setFontSize(26);
      doc.text("Rapport thermique CSP", 16, 33);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.setTextColor(178, 178, 184);
      doc.text(`Généré le ${generatedAt}`, 16, 43);

      sectionTitle("Résumé AI", 68);
      doc.setTextColor(...text);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.text(`Vidéo source: ${videoFile?.name ?? "non renseignée"}`, 16, 84, { maxWidth: 172 });
      metric("Frames", `${aiSummary.sampledFrames}`, 16, 96);
      metric("Détections", `${aiSummary.totalDetections}`, 75, 96);
      metric("Tube ref", `${aiSummary.tubeRefFrames}`, 134, 96);
      metric("Tube test", `${aiSummary.tubeTestFrames}`, 16, 124);

      sectionTitle("Détection d'anomalies", 162);
      metric("Décision", videoDecisionLabel(aiSummary.decision), 16, 176, 58);
      metric("Score max", formatAnomalyScore(aiSummary.maxAnomalyScore), 81, 176, 50);
      metric("Score moyen", formatAnomalyScore(aiSummary.averageAnomalyScore), 138, 176, 56);
      metric("Frames suspectes", `${aiSummary.warningFrameCount}`, 16, 204, 58);
      metric("Pic temporel", formatTime(aiSummary.peakFrameTime), 81, 204, 50);
      metric("Confiance", confidenceLabel(aiSummary.decisionConfidence), 138, 204, 56);

      doc.setTextColor(...muted);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(9);
      doc.text(
        `Sévérité pic: ${severityLabel(aiSummary.peakSeverity)} · Ratio suspect: ${(aiSummary.suspectFrameRatio * 100).toFixed(1)}% · Persistance max: ${aiSummary.longestSuspectRun} frame(s)`,
        16,
        233,
        { maxWidth: 178 }
      );
      doc.text(
        `Support autoencoder: ${aiSummary.autoencoderSupportFrames} frames · Delta moyen AE: ${formatAnomalyScore(aiSummary.autoencoderAverageDelta)} · Ratio max AE: ${formatAnomalyScore(aiSummary.autoencoderPeakRatio)}`,
        16,
        238,
        { maxWidth: 178 }
      );

      sectionTitle("Températures relevées", 242);
      const rows = [
        { label: "tube-ref", values: tubeRefTemps, average: refStats.average },
        { label: "tube-test", values: tubeTestTemps, average: testStats.average }
      ];

      let y = 256;
      rows.forEach((row) => {
        doc.setDrawColor(...border);
        doc.setFillColor(250, 250, 251);
        doc.roundedRect(16, y, 178, 20, 3, 3, "FD");
        doc.setTextColor(...text);
        doc.setFont("helvetica", "bold");
        doc.setFontSize(10);
        doc.text(row.label, 22, y + 8);
        doc.setTextColor(...muted);
        doc.setFont("helvetica", "normal");
        doc.text(`Moyenne: ${formatAverage(row.average)}`, 142, y + 8);
        const labels = row.values.map((_, index) => `${getTemperaturePointLabel(index)}: ${row.values[index] || "--"} °C`);
        const lineA = labels.filter((_, index) => index % 2 === 0).join("    ");
        const lineB = labels.filter((_, index) => index % 2 === 1).join("    ");
        doc.text(lineA, 22, y + 16, { maxWidth: 150 });
        if (lineB) doc.text(lineB, 22, y + 22, { maxWidth: 150 });
        y += lineB ? 32 : 26;
      });

      doc.setFillColor(246, 246, 247);
      doc.roundedRect(16, y + 2, 178, 18, 3, 3, "F");
      doc.setTextColor(...muted);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.text(
        "Rapport généré localement à partir des résultats IA disponibles et des températures saisies.",
        22,
        y + 13,
        { maxWidth: 166 }
      );

      doc.addPage();
      sectionTitle("Interprétation Finale", 24);
      doc.setTextColor(...text);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(12);
      doc.text("Lecture des indicateurs", 16, 40);
      doc.setTextColor(...muted);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.text(
        "La comparaison thermique mesure l'écart relatif entre tube_ref (référence saine) et tube_test. L'autoencoder mesure à quel point le tube test s'écarte d'un comportement visuel/thermique appris comme normal.",
        16,
        50,
        { maxWidth: 178 }
      );

      const interpretation = buildInterpretation();
      let interpretationY = 74;
      interpretation.forEach((paragraph) => {
        const lines = doc.splitTextToSize(paragraph, 174);
        doc.setTextColor(...text);
        doc.text(lines, 20, interpretationY);
        interpretationY += lines.length * 6 + 6;
      });

      doc.setDrawColor(...border);
      doc.setFillColor(250, 250, 251);
      doc.roundedRect(16, interpretationY + 2, 178, 30, 3, 3, "FD");
      doc.setTextColor(...muted);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(9);
      doc.text("Recommandation opérationnelle", 22, interpretationY + 12);
      doc.setFont("helvetica", "normal");
      doc.text(
        aiSummary.decision === "anomaly"
          ? "Inspecter prioritairement les frames autour du pic temporel et confronter la détection aux observations terrain."
          : aiSummary.decision === "warning"
            ? "Contrôler visuellement les frames signalées et confirmer l'écart thermique avant de conclure à une anomalie."
            : "Aucune anomalie forte n'est confirmée; conserver la vidéo comme référence normale si l'inspection terrain confirme l'état sain.",
        22,
        interpretationY + 20,
        { maxWidth: 166 }
      );

      const blob = doc.output("blob");
      const url = URL.createObjectURL(blob);
      setReportUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return url;
      });
      setReportStatus(`PDF généré le ${generatedAt}.`);
    } catch (event) {
      const message = event instanceof Error ? event.message : "Erreur de génération PDF";
      setReportStatus(`Erreur: ${message}`);
    } finally {
      setGeneratingReport(false);
    }
  }

  function handleSegmentationAction() {
    if (running) return;
    if (!videoFile) {
      fileInputRef.current?.click();
      return;
    }

    void annotateVideo(false);
  }

  function handleAnomalyAnalysisAction() {
    if (running) return;
    if (!videoFile) {
      fileInputRef.current?.click();
      return;
    }

    void annotateVideo(true);
  }

  function handleReportAction() {
    if (!canGenerateReport) {
      setReportStatus("Lancez d'abord l'analyse d'anomalies, puis vérifiez les températures avant de générer le PDF.");
      return;
    }

    void generateReportPdf();
  }

  return {
    aiSummary,
    backendStatus,
    canGenerateReport,
    canvasRef,
    error,
    fileInputRef,
    generatingReport,
    hasAnomalyAnalysis,
    handleAnomalyAnalysisAction,
    handleReportAction,
    handleSegmentationAction,
    handleVideoChange,
    hiddenVideoRef,
    outUrl,
    predictionCount,
    progress,
    refStats,
    reportStatus,
    reportUrl,
    running,
    step,
    testStats,
    tubeRefTemps,
    tubeTestTemps,
    addTemperatureField,
    updateTemperature,
    videoFile,
    videoPreviewUrl
  };
}
