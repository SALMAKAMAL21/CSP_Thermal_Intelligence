import type { ChangeEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createRefTemperatures, createTestTemperatures, getTemperatureStats,
  INITIAL_BACKEND_STATUS, pickColor, REQUIRED_TEMPERATURE_COUNT,
  REPORT_WAITING_STATUS, summarizePredictions, TARGET_LABELS, levelLabel,
  type AiSummary, type BackendConnectionStatus, type Detection,
  type FramePrediction, type PredictResponse, type FusionResult
} from "@/lib/thermal";
import { drawFusionReport, type ReportCapture } from "@/lib/fusion-report";
import { captureVideoFrame, seekVideoFrame } from "@/lib/video-frame";
import { readApiJson } from "@/lib/api-response";

import { emptyInspection, completeInspectionTimestamp, inspectionComplete, temperaturesComplete, type InspectionDetails } from "@/lib/inspection";

type SampledFrameCache = {
  blob: Blob;
  detections: Detection[];
  t: number;
};

export function useCspAnalysis() {
  const [inspection, setInspection] = useState<InspectionDetails>(emptyInspection);
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
  const reportCaptureRef = useRef<ReportCapture | null>(null);
  const sampledFramesRef = useRef<SampledFrameCache[]>([]);
  const midpointFrameRef = useRef<SampledFrameCache | null>(null);

  const refStats = useMemo(() => getTemperatureStats(tubeRefTemps), [tubeRefTemps]);
  const testStats = useMemo(() => getTemperatureStats(tubeTestTemps), [tubeTestTemps]);
  const allTemperaturesReady = temperaturesComplete(tubeRefTemps) && temperaturesComplete(tubeTestTemps);
  const canAnalyze = !!videoFile && inspectionComplete(inspection) && allTemperaturesReady && !running && !generatingReport;
  const canGenerateReport = !!aiSummary && aiSummary.level !== null && hasAnomalyAnalysis &&
    allTemperaturesReady && inspectionComplete(inspection) && !generatingReport && !running;

  const refreshBackendStatus = useCallback(async () => {
    setBackendStatus((current) => ({ ...current, checking: true }));

    try {
      const res = await fetch("/api/segment-check", { cache: "no-store" });
      const data = await readApiJson<Omit<BackendConnectionStatus, "checking">>(res, "/api/segment-check");
      if (!res.ok || typeof data.reachable !== "boolean" || typeof data.modelLoaded !== "boolean") {
        throw new Error(`Impossible de vérifier FastAPI : erreur de l’interface sur /api/segment-check (HTTP ${res.status}). Consultez le terminal Next.js.`);
      }
      const nextStatus = { ...data, checking: false, frontendError: false };
      setBackendStatus(nextStatus);
      return nextStatus;
    } catch (event) {
      const details = event instanceof Error ? event.message : "Connexion à l’interface impossible";
      const nextStatus: BackendConnectionStatus = {
        checking: false,
        frontendError: true,
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
    if (running || generatingReport) return;
    reportCaptureRef.current = null;
    setTubeRefTemps(createRefTemperatures());
    setTubeTestTemps(createTestTemperatures());
    setVideoFile(event.target.files?.[0] ?? null);
    sampledFramesRef.current = [];
    midpointFrameRef.current = null;
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

  function updateInspection(field: keyof InspectionDetails, value: string) {
    if (running || generatingReport) return;
    setInspection(current => ({ ...current, [field]: value }));
    setAiSummary(null);
    setHasAnomalyAnalysis(false);
    setReportUrl(null);
    reportCaptureRef.current = null;
    setReportStatus("Informations modifiées : relancez l’analyse pour actualiser le résultat.");
  }

  function updateTemperature(tube: "ref" | "test", index: number, value: string) {
    if (running || generatingReport) return;
    setAiSummary(null);
    setHasAnomalyAnalysis(false);
    reportCaptureRef.current = null;
    const setter = tube === "ref" ? setTubeRefTemps : setTubeTestTemps;
    setter((current) => current.map((item, itemIndex) => (itemIndex === index ? value : item)));
    setReportStatus("Températures modifiées : relancez la classification avant de générer le rapport.");
    setReportUrl(null);
  }

  function addTemperatureField() {
    if (running || generatingReport) return;
    setAiSummary(null);
    setHasAnomalyAnalysis(false);
    reportCaptureRef.current = null;
    setTubeRefTemps((current) => [...current, ""]);
    setTubeTestTemps((current) => [...current, ""]);
    setReportStatus("Mesures modifiées : relancez la classification avant de générer le rapport.");
    setReportUrl(null);
  }

  async function predictFrame(
    blob: Blob,
    idx: number,
    includeAnomaly: boolean,
    sessionId: string,
    resetTracker: boolean,
    detections?: Detection[]
  ): Promise<{ detections: Detection[]; fusion: FusionResult | null }> {
    const fd = new FormData();
    fd.append("image", new File([blob], `frame-${idx}.jpg`, { type: "image/jpeg" }));
    fd.append("conf", "0.5");
    if (includeAnomaly) {
      fd.append("t_ref", String(refStats.average));
      fd.append("t_test", String(testStats.average));
    }
    fd.append("iou", "0.45");
    fd.append("analyze_thermal", includeAnomaly ? "true" : "false");
    fd.append("session_id", sessionId);
    fd.append("reset_tracker", resetTracker ? "true" : "false");
    if (detections) fd.append("detections_json", JSON.stringify(detections));

    const res = await fetch("/api/segment-predict", { method: "POST", body: fd });
    const data = await readApiJson<PredictResponse>(res, "/api/segment-predict");
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
      detections: ((detections ? data.detections || detections : data.detections) || []).filter((d) =>
        TARGET_LABELS.has((d.class_name || "").toLowerCase())
      ),
      fusion: includeAnomaly ? data.fusion || null : null
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
    if (!canAnalyze) {
      setError("Complétez les informations opérateur, la vidéo, toutes les températures et votre observation visuelle avant de lancer l’analyse.");
      return;
    }
    if (!videoFile) return;
    if (includeAnomaly && !allTemperaturesReady) {
      setError("Renseignez au moins quatre températures valides par tube avant la classification.");
      return;
    }
    setInspection(completeInspectionTimestamp(inspection));
    reportCaptureRef.current = null;
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
      setError(backend.frontendError ? backend.details ?? "Erreur de connexion à l’interface" : `Backend IA hors ligne${backend.backendUrl ? ` (${backend.backendUrl})` : ""}. ${backend.details ?? ""}`.trim());
      setStep(backend.frontendError ? "Erreur de l’interface" : "Backend hors ligne");
      setRunning(false);
      return;
    }

    if (!backend.modelLoaded) {
      setError(`Backend connecté, mais modèle IA non chargé${backend.backendUrl ? ` (${backend.backendUrl})` : ""}. Vérifiez MODEL_PATH et le démarrage FastAPI.`);
      setStep("Modèle IA indisponible");
      setRunning(false);
      return;
    }

    if (includeAnomaly && !backend.fusionLoaded) {
      setError(`Modèle de fusion indisponible. ${backend.details ?? "Vérifiez le checkpoint et configs/fusion.json."}`);
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
    let captureStream: MediaStream | null = null;
    let activeRecorder: MediaRecorder | null = null;
    let animationFrame: number | null = null;
    video.muted = true;
    video.playsInline = true;

    try {
      setStep("Chargement vidéo");
      await new Promise<void>((resolve, reject) => {
        video.onloadeddata = () => resolve();
        video.onerror = () => reject(new Error("Lecture vidéo impossible"));
        video.src = sourceUrl;
        video.load();
      });

      if (!Number.isFinite(video.duration) || video.duration <= 0) throw new Error("Durée vidéo invalide");
      const duration = video.duration;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Canvas context indisponible");

      setStep(includeAnomaly ? "Segmentation + analyse" : "Segmentation YOLO");
      const samplingFps = 1; // Même cadence que les frames d’entraînement.
      const sessionId =
        typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
          ? crypto.randomUUID()
          : `session-${Date.now()}`;
      const sampleTimes: number[] = [];
      for (let t = 0; t < duration; t += 1 / samplingFps) sampleTimes.push(t);


      const predictions: FramePrediction[] = [];
      const canReuseSegmentation = includeAnomaly && sampledFramesRef.current.length === sampleTimes.length;

      if (canReuseSegmentation) {
        setStep("Classification sans re-segmentation");
        for (let i = 0; i < sampledFramesRef.current.length; i++) {
          const cached = sampledFramesRef.current[i];
          const frameResult = await predictFrame(cached.blob, i + 1, true, sessionId, false, cached.detections);
          predictions.push({
            t: cached.t,
            detections: cached.detections,
            fusion: frameResult.fusion
          });
          setPredictionCount(predictions.length);
          setProgress(Math.round(((i + 1) / sampledFramesRef.current.length) * 55));
        }
      } else {
        const sampledFrames: SampledFrameCache[] = [];
        for (let i = 0; i < sampleTimes.length; i++) {
          const t = sampleTimes[i];
          await new Promise<void>((resolve, reject) => {
            if (Math.abs(video.currentTime - t) < 0.001 && video.readyState >= 2) { resolve(); return; }
            const timeout = window.setTimeout(() => {
              video.removeEventListener("seeked", onSeeked);
              reject(new Error("Délai de lecture vidéo dépassé"));
            }, 15000);
            const onSeeked = () => {
              window.clearTimeout(timeout);
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
            fusion: frameResult.fusion
          });
          drawDetections(ctx, frameResult.detections);
          sampledFrames.push({
            t,
            blob,
            detections: frameResult.detections
          });
          setPredictionCount(predictions.length);
          setStep(includeAnomaly ? "Segmentation + classification" : "Segmentation YOLO");
          setProgress(Math.round(((i + 1) / sampleTimes.length) * 55));
        }
        sampledFramesRef.current = sampledFrames;
      }

      const summary = summarizePredictions(predictions, includeAnomaly ? refStats.average : null, includeAnomaly ? testStats.average : null);
      if (includeAnomaly && summary.validFrames === 0) throw new Error("Aucune paire de tubes exploitable : classification indisponible.");
      if (includeAnomaly) {
        setStep("Capture du milieu de la vidéo");
        const midpoint = duration / 2;
        const capture = await captureVideoFrame(video, midpoint);
        const captureContext = capture.getContext("2d");
        if (!captureContext) throw new Error("Capture du rapport indisponible.");
        if (!midpointFrameRef.current) {
          const blob = await new Promise<Blob | null>(resolve => capture.toBlob(resolve, "image/jpeg", 0.95));
          if (!blob) throw new Error("L’image du milieu de la vidéo n’a pas pu être extraite.");
          const sampled = sampledFramesRef.current.find(frame => Math.abs(frame.t - midpoint) < 0.001);
          // Les boîtes doivent correspondre à cette image précise, sans influencer la synthèse vidéo.
          const detections = sampled?.detections ?? (await predictFrame(blob, 0, false, `${sessionId}-report`, true)).detections;
          midpointFrameRef.current = { blob, detections, t: midpoint };
        }
        drawDetections(captureContext, midpointFrameRef.current.detections);
        reportCaptureRef.current = {
          dataUrl: capture.toDataURL("image/jpeg", 0.95),
          width: capture.width, height: capture.height, time: midpoint
        };
      }
      setAiSummary(summary);
      setHasAnomalyAnalysis(includeAnomaly);

      setStep("Rendu vidéo");
      await seekVideoFrame(video, 0);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      drawDetections(ctx, predictions[0]?.detections ?? []);

      const stream = canvas.captureStream(30);
      captureStream = stream;
      const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
      const recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 3_000_000 });
      activeRecorder = recorder;
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      };

      const done = new Promise<Blob>((resolve, reject) => {
        recorder.onstop = () => resolve(new Blob(chunks, { type: mime }));
        recorder.onerror = () => reject(new Error("Échec de l’enregistrement vidéo"));
      });
      // L'enregistreur peut échouer pendant l'attente de video.play().
      void done.catch(() => {});

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

      video.playbackRate = 1.0;
      recorder.start();
      await video.play();

      await Promise.race([done, new Promise<void>((resolve, reject) => {
        video.onerror = () => reject(new Error("Lecture vidéo interrompue"));
        const render = () => {
          if (video.paused || video.ended) {
            resolve();
            return;
          }

          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          drawDetections(ctx, getNearest(video.currentTime));

          const nextProgress = 55 + Math.round((video.currentTime / duration) * 45);
          setProgress(Math.min(100, nextProgress));
          animationFrame = requestAnimationFrame(render);
        };
        animationFrame = requestAnimationFrame(render);
      })]);

      if (recorder.state !== "inactive") recorder.stop();
      const outBlob = await done;
      stream.getTracks().forEach(track => track.stop());
      video.playbackRate = 1.0;
      const annotatedUrl = URL.createObjectURL(outBlob);
      setOutUrl(annotatedUrl);
      setProgress(100);
      setStep("Terminé");
      setReportStatus(includeAnomaly
        ? `Niveau ${summary.level} : ${levelLabel(summary.level)}. ${summary.validFrames}/${summary.sampledFrames} images analysées. Rapport prêt.`
        : "Segmentation terminée. Renseignez les températures, puis lancez la classification.");
    } catch (event) {
      const message = event instanceof Error ? event.message : "Erreur d'annotation vidéo";
      setError(message);
      setStep("Erreur");
    } finally {
      if (animationFrame !== null) cancelAnimationFrame(animationFrame);
      if (activeRecorder && activeRecorder.state !== "inactive") activeRecorder.stop();
      captureStream?.getTracks().forEach(track => track.stop());
      video.pause();
      video.onloadeddata = null;
      video.onerror = null;
      video.removeAttribute("src");
      video.load();
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
      const generatedAt = new Date().toLocaleString("fr-FR");
      drawFusionReport(doc, aiSummary, reportCaptureRef.current, { ref: tubeRefTemps.map(Number), test: tubeTestTemps.map(Number) }, inspection);
      setReportUrl(URL.createObjectURL(doc.output("blob")));
      setReportStatus(`Rapport synthétique généré le ${generatedAt}.`);
    } catch (event) {
      const message = event instanceof Error ? event.message : "Erreur de génération PDF";
      setReportStatus(`Erreur: ${message}`);
    } finally {
      setGeneratingReport(false);
    }
  }

  function handleSegmentationAction() {
    if (running || generatingReport) return;
    if (!videoFile) {
      fileInputRef.current?.click();
      return;
    }

    void annotateVideo(false);
  }

  function handleAnomalyAnalysisAction() {
    if (running || generatingReport) return;
    if (!videoFile) {
      fileInputRef.current?.click();
      return;
    }

    void annotateVideo(true);
  }

  function handleReportAction() {
    if (!canGenerateReport) {
      setReportStatus("Renseignez les températures puis lancez la classification avant de générer le PDF.");
      return;
    }

    void generateReportPdf();
  }

  return {
    inspection,
    updateInspection,
    canAnalyze,
    allTemperaturesReady,
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
