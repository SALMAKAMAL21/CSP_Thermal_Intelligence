import type { ChangeEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createRefTemperatures,
  createTestTemperatures,
  formatAnomalyScore,
  formatAverage,
  formatTime,
  getTemperatureStats,
  INITIAL_BACKEND_STATUS,
  pickColor,
  REPORT_WAITING_STATUS,
  severityLabel,
  summarizePredictions,
  TARGET_LABELS,
  TEMPERATURE_POINTS,
  type AiSummary,
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
  const allTemperaturesReady = refStats.completed === 4 && testStats.completed === 4;
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

  async function predictFrame(
    blob: Blob,
    idx: number,
    includeAnomaly: boolean,
    sessionId: string,
    resetTracker: boolean
  ): Promise<{ detections: Detection[]; anomaly: ThermalAnomaly | null }> {
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
      anomaly: includeAnomaly ? data.thermal_anomaly || null : null
    };
  }

  function drawDetections(ctx: CanvasRenderingContext2D, detections: Detection[]) {
    for (const det of detections) {
      const label = (det.class_name || "unknown").toLowerCase();
      const color = pickColor(label);

      if (Array.isArray(det.mask_polygon) && det.mask_polygon.length > 2) {
        ctx.beginPath();
        det.mask_polygon.forEach((p, i) => {
          if (i === 0) ctx.moveTo(p[0], p[1]);
          else ctx.lineTo(p[0], p[1]);
        });
        ctx.closePath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.fillStyle = `${color}24`;
        ctx.fill();
      } else if (Array.isArray(det.bbox) && det.bbox.length === 4) {
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
        predictions.push({ t, detections: frameResult.detections, anomaly: frameResult.anomaly });
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
            ? `Résultat AI disponible. Score anomalie max ${formatAnomalyScore(summary.maxAnomalyScore)}. Rapport PDF prêt à générer.`
            : `Résultat AI disponible. Score anomalie max ${formatAnomalyScore(summary.maxAnomalyScore)}. Compléter les 8 températures pour générer le rapport PDF.`
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
      metric("Sévérité", severityLabel(aiSummary.peakSeverity), 16, 176, 58);
      metric("Score max", formatAnomalyScore(aiSummary.maxAnomalyScore), 81, 176, 50);
      metric("Score moyen", formatAnomalyScore(aiSummary.averageAnomalyScore), 138, 176, 56);
      metric("Frames suspectes", `${aiSummary.anomalyFrames}`, 16, 204, 58);
      metric("Pic temporel", formatTime(aiSummary.peakFrameTime), 81, 204, 50);
      metric("Segments", `${aiSummary.suspectSegmentCount}`, 138, 204, 56);

      sectionTitle("Températures relevées", 242);
      const rows = [
        { label: "Tube Référence", values: tubeRefTemps, average: refStats.average },
        { label: "Tube Test", values: tubeTestTemps, average: testStats.average }
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
        row.values.forEach((value, index) => {
          doc.text(`${TEMPERATURE_POINTS[index]}: ${value || "--"} °C`, 22 + index * 40, y + 16);
        });
        y += 26;
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
    updateTemperature,
    videoFile,
    videoPreviewUrl
  };
}
