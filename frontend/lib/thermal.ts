export type Detection = {
  class_name?: string;
  confidence?: number;
  bbox?: number[];
  mask_polygon?: number[][];
};

export type SuspectRegion = {
  start_segment?: number;
  end_segment?: number;
  score?: number;
};

export type ThermalAnomaly = {
  status?: string;
  enabled?: boolean;
  thermal_source?: string;
  note?: string;
  profile_bins?: number;
  anomaly_score?: number;
  severity?: string;
  suspect_segments?: number[];
  suspect_regions?: SuspectRegion[];
  reason?: string;
  valid_segments?: number;
};

export type PredictResponse = {
  detections?: Detection[];
  thermal_anomaly?: ThermalAnomaly;
  error?: string;
  details?: string;
  backendStatus?: number;
  backendUrl?: string;
  backendResponse?: unknown;
};

export type BackendConnectionStatus = {
  backendUrl?: string;
  checking: boolean;
  classes?: unknown;
  details?: string;
  model?: string;
  modelLoaded: boolean;
  reachable: boolean;
  status?: string;
  task?: string;
};

export type FramePrediction = {
  t: number;
  detections: Detection[];
  anomaly: ThermalAnomaly | null;
};

export type AiSummary = {
  sampledFrames: number;
  tubeRefFrames: number;
  tubeTestFrames: number;
  totalDetections: number;
  anomalyFrames: number;
  thermalFrames: number;
  maxAnomalyScore: number;
  averageAnomalyScore: number;
  peakSeverity: string;
  peakFrameTime: number | null;
  suspectSegmentCount: number;
};

export type TemperatureStats = {
  average: number | null;
  completed: number;
};

export const TARGET_LABELS = new Set(["tube_ref", "tube_test", "tube", "hce"]);
export const TEMPERATURE_POINTS = ["T1", "T2", "T3", "T4"] as const;
export const REF_TEMP_PLACEHOLDERS = ["345.0", "346.1", "344.8", "345.0"] as const;
export const TEST_TEMP_PLACEHOLDERS = ["288.0", "289.2", "288.1", "288.3"] as const;
export const REPORT_WAITING_STATUS = "Rapport en attente de la vidéo traitée.";
export const INITIAL_BACKEND_STATUS: BackendConnectionStatus = {
  checking: true,
  modelLoaded: false,
  reachable: false
};

export const createRefTemperatures = () => Array.from({ length: TEMPERATURE_POINTS.length }, () => "");
export const createTestTemperatures = () => Array.from({ length: TEMPERATURE_POINTS.length }, () => "");

export function pickColor(label: string) {
  return label === "tube_ref" ? "#f8fafc" : "#22c55e";
}

export function formatFileSize(size: number) {
  if (size < 1024 * 1024) return `${Math.max(1, Math.round(size / 1024))} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

export function getTemperatureStats(values: string[]): TemperatureStats {
  const parsed = values.map((value) => Number.parseFloat(value)).filter(Number.isFinite);
  const average = parsed.length ? parsed.reduce((sum, value) => sum + value, 0) / parsed.length : null;

  return {
    average,
    completed: parsed.length
  };
}

export function formatAverage(value: number | null) {
  return value === null ? "-- °C" : `${value.toFixed(1)} °C`;
}

export function getAnomalyScore(anomaly: ThermalAnomaly | null | undefined) {
  const score = Number(anomaly?.anomaly_score ?? 0);
  return Number.isFinite(score) ? score : 0;
}

export function formatAnomalyScore(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "--";
  return value.toFixed(3);
}

export function formatTime(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "--";
  return `${value.toFixed(2)} s`;
}

export function severityLabel(severity: string | null | undefined) {
  switch ((severity || "").toLowerCase()) {
    case "high":
      return "Élevée";
    case "moderate":
      return "Modérée";
    case "low":
      return "Faible";
    case "none":
      return "Aucune";
    case "unknown":
      return "Indéterminée";
    default:
      return "--";
  }
}

export function summarizePredictions(predictions: FramePrediction[]): AiSummary {
  const summary = predictions.reduce<AiSummary>(
    (summary, frame) => {
      const labels = frame.detections.map((detection) => (detection.class_name || "").toLowerCase());
      const score = getAnomalyScore(frame.anomaly);
      const isThermalFrame = frame.anomaly?.status === "ok";
      const isAnomalyFrame = score >= 0.2;
      const suspectSegments = frame.anomaly?.suspect_segments?.length || 0;
      const isPeak = score > summary.maxAnomalyScore;

      return {
        sampledFrames: summary.sampledFrames + 1,
        tubeRefFrames: summary.tubeRefFrames + (labels.includes("tube_ref") ? 1 : 0),
        tubeTestFrames: summary.tubeTestFrames + (labels.includes("tube_test") ? 1 : 0),
        totalDetections: summary.totalDetections + frame.detections.length,
        anomalyFrames: summary.anomalyFrames + (isAnomalyFrame ? 1 : 0),
        thermalFrames: summary.thermalFrames + (isThermalFrame ? 1 : 0),
        maxAnomalyScore: isPeak ? score : summary.maxAnomalyScore,
        averageAnomalyScore: summary.averageAnomalyScore + score,
        peakSeverity: isPeak ? frame.anomaly?.severity || "none" : summary.peakSeverity,
        peakFrameTime: isPeak ? frame.t : summary.peakFrameTime,
        suspectSegmentCount: summary.suspectSegmentCount + suspectSegments
      };
    },
    {
      sampledFrames: 0,
      tubeRefFrames: 0,
      tubeTestFrames: 0,
      totalDetections: 0,
      anomalyFrames: 0,
      thermalFrames: 0,
      maxAnomalyScore: 0,
      averageAnomalyScore: 0,
      peakSeverity: "none",
      peakFrameTime: null,
      suspectSegmentCount: 0
    }
  );

  return {
    ...summary,
    averageAnomalyScore: summary.thermalFrames ? summary.averageAnomalyScore / summary.thermalFrames : 0
  };
}
