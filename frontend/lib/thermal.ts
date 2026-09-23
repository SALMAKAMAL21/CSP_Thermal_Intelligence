export type Detection = { class_name?: string; confidence?: number; bbox?: number[]; mask_polygon?: number[][] };
export type FusionResult = {
  status: string; level?: number; label?: string; cumulative_probabilities?: number[];
  delta_t?: number; reason?: string;
};
export type PredictResponse = {
  detections?: Detection[]; fusion?: FusionResult; error?: string; details?: string;
  backendStatus?: number; backendUrl?: string; backendResponse?: unknown;
};
export type BackendConnectionStatus = {
  checking: boolean; reachable: boolean; modelLoaded: boolean; fusionLoaded?: boolean;
  backendUrl?: string; details?: string; model?: string; status?: string; frontendError?: boolean;
};
export type FramePrediction = { t: number; detections: Detection[]; fusion: FusionResult | null };
export type AiSummary = {
  sampledFrames: number; validFrames: number; skippedFrames: number; level: number | null;
  cumulativeProbabilities: number[]; representativeTime: number | null;
  tRef: number | null; tTest: number | null; deltaT: number | null;
};
export type TemperatureStats = { average: number | null; completed: number; valid: boolean };
export const LEVEL_NAMES = ["Normal", "Quantité minimale d'air", "10 L d'air", "18 L d'air", "Absence de vide"];
export const TARGET_LABELS = new Set(["tube_ref", "tube_test", "tube", "hce"]);
export const REQUIRED_TEMPERATURE_COUNT = 4;
export const REPORT_WAITING_STATUS = "Le rapport sera disponible après la classification de la vidéo.";
export const INITIAL_BACKEND_STATUS: BackendConnectionStatus = { checking: true, reachable: false, modelLoaded: false, fusionLoaded: false };
export const createRefTemperatures = () => Array.from({ length: REQUIRED_TEMPERATURE_COUNT }, () => "");
export const createTestTemperatures = () => Array.from({ length: REQUIRED_TEMPERATURE_COUNT }, () => "");
export const getTemperaturePointLabel = (index: number) => `T${index + 1}`;
export const getTemperaturePlaceholder = (_tube: "ref" | "test", _index: number) => "°C";
export const levelLabel = (level: number | null | undefined) => level == null ? "Indisponible" : LEVEL_NAMES[level] ?? "Indisponible";
export const pickColor = (label: string) => label === "tube_ref" ? "#f8fafc" : "#22c55e";
export const formatAverage = (value: number | null) => value === null ? "-- °C" : `${value.toFixed(1)} °C`;
export function formatFileSize(size: number) {
  return size < 1024 * 1024 ? `${Math.max(1, Math.round(size / 1024))} KB` : `${(size / (1024 * 1024)).toFixed(1)} MB`;
}
export function getTemperatureStats(values: string[]): TemperatureStats {
  const parsed = values.filter(v => v.trim() !== "").map(Number);
  const valid = parsed.every(Number.isFinite);
  return { average: valid && parsed.length ? parsed.reduce((a, b) => a + b, 0) / parsed.length : null,
    completed: parsed.filter(Number.isFinite).length, valid };
}
export function summarizePredictions(predictions: FramePrediction[], tRef: number | null = null, tTest: number | null = null): AiSummary {
  const valid = predictions.filter(frame => {
    const p = frame.fusion?.cumulative_probabilities;
    return frame.fusion?.status === "ok" && p?.length === 4 &&
      p.every((v, i) => Number.isFinite(v) && v >= 0 && v <= 1 && (i === 0 || v <= p[i - 1]));
  });
  const cumulativeProbabilities = valid.length ? [0, 1, 2, 3].map(k =>
    valid.reduce((sum, frame) => sum + frame.fusion!.cumulative_probabilities![k], 0) / valid.length) : [];
  const level = valid.length ? cumulativeProbabilities.filter(p => p > 0.5).length : null;
  // Capture la plus proche des probabilités moyennes, parmi les paires valides.
  const distance = (frame: FramePrediction) => frame.fusion!.cumulative_probabilities!
    .reduce((sum, p, k) => sum + Math.abs(p - cumulativeProbabilities[k]), 0);
  const representative = valid.reduce<FramePrediction | null>((best, frame) =>
    !best || distance(frame) < distance(best) ? frame : best, null);
  return { sampledFrames: predictions.length, validFrames: valid.length,
    skippedFrames: predictions.length - valid.length, level, cumulativeProbabilities,
    representativeTime: representative?.t ?? null, tRef, tTest,
    deltaT: tRef !== null && tTest !== null ? tRef - tTest : null };
}
