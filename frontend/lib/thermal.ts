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

export type AutoencoderAnomaly = {
  status?: string;
  enabled?: boolean;
  source?: string;
  tube_ref_score?: number;
  tube_test_score?: number;
  score_delta?: number;
  score_ratio?: number;
  support_score?: number;
  decision?: VideoDecision;
  confidence?: VideoDecisionConfidence;
  input_size?: number;
  reason?: string;
  error?: string;
};

export type SiameseAnomaly = {
  status?: string;
  enabled?: boolean;
  source?: string;
  pair_probability?: number;
  embedding_distance?: number;
  support_score?: number;
  decision?: VideoDecision;
  confidence?: VideoDecisionConfidence;
  input_size?: number;
  reason?: string;
  error?: string;
};

export type CombinedAnomaly = {
  status?: string;
  enabled?: boolean;
  source?: string;
  anomaly_score?: number;
  decision?: VideoDecision;
  confidence?: VideoDecisionConfidence;
  thermal_score?: number;
  autoencoder_support?: number;
  siamese_support?: number;
  note?: string;
  error?: string;
};

export type PredictResponse = {
  detections?: Detection[];
  thermal_anomaly?: ThermalAnomaly;
  autoencoder_anomaly?: AutoencoderAnomaly;
  siamese_anomaly?: SiameseAnomaly;
  combined_anomaly?: CombinedAnomaly;
  error?: string;
  details?: string;
  backendStatus?: number;
  backendUrl?: string;
  backendResponse?: unknown;
};

export type BackendConnectionStatus = {
  backendUrl?: string;
  autoencoderLoaded?: boolean;
  checking: boolean;
  classes?: unknown;
  details?: string;
  model?: string;
  modelLoaded: boolean;
  reachable: boolean;
  siameseLoaded?: boolean;
  siameseModel?: string | null;
  status?: string;
  task?: string;
};

export type FramePrediction = {
  t: number;
  detections: Detection[];
  anomaly: ThermalAnomaly | null;
  autoencoder: AutoencoderAnomaly | null;
  siamese: SiameseAnomaly | null;
  combined: CombinedAnomaly | null;
};

export type VideoDecision = "normal" | "warning" | "anomaly" | "unknown";
export type VideoDecisionConfidence = "low" | "medium" | "high";

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
  suspectFrameRatio: number;
  longestSuspectRun: number;
  warningFrameCount: number;
  alertFrameCount: number;
  decision: VideoDecision;
  decisionConfidence: VideoDecisionConfidence;
  autoencoderSupportFrames: number;
  autoencoderPeakRatio: number;
  autoencoderAverageDelta: number;
  siameseSupportFrames: number;
  siamesePeakProbability: number;
};

export type TemperatureStats = {
  average: number | null;
  completed: number;
};

export type TemperatureDeltaInterpretation = {
  delta: number | null;
  level: number | null;
  label: string;
  description: string;
};

export const TARGET_LABELS = new Set(["tube_ref", "tube_test", "tube", "hce"]);
export const BASE_TEMPERATURE_POINTS = ["T1", "T2", "T3", "T4"] as const;
export const REQUIRED_TEMPERATURE_COUNT = BASE_TEMPERATURE_POINTS.length;
export const REF_TEMP_PLACEHOLDERS = ["345.0", "346.1", "344.8", "345.0"] as const;
export const TEST_TEMP_PLACEHOLDERS = ["288.0", "289.2", "288.1", "288.3"] as const;
export const REPORT_WAITING_STATUS = "Rapport en attente de la vidéo traitée.";
export const INITIAL_BACKEND_STATUS: BackendConnectionStatus = {
  autoencoderLoaded: false,
  checking: true,
  modelLoaded: false,
  reachable: false,
  siameseLoaded: false
};

export const createRefTemperatures = () => Array.from({ length: REQUIRED_TEMPERATURE_COUNT }, () => "");
export const createTestTemperatures = () => Array.from({ length: REQUIRED_TEMPERATURE_COUNT }, () => "");

export function getTemperaturePointLabel(index: number) {
  return `T${index + 1}`;
}

export function getTemperaturePlaceholder(tube: "ref" | "test", index: number) {
  const source = tube === "ref" ? REF_TEMP_PLACEHOLDERS : TEST_TEMP_PLACEHOLDERS;
  return source[index] ?? "";
}
export const VIDEO_DECISION_THRESHOLDS = {
  frameWarningScore: 0.24,
  frameAlertScore: 0.38,
  videoMeanWarningScore: 0.02,
  videoMeanAlertScore: 0.05,
  suspectRatioWarning: 0.05,
  suspectRatioAlert: 0.1,
  minPersistentFrames: 2
} as const;

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

export function getTemperatureDeltaInterpretation(
  refAverage: number | null,
  testAverage: number | null,
): TemperatureDeltaInterpretation {
  if (refAverage === null || testAverage === null) {
    return {
      delta: null,
      level: null,
      label: "Interprétation indisponible",
      description: "Les moyennes tube-ref et tube-test sont nécessaires pour interpréter l'écart thermique.",
    };
  }

  // The magnitude of the thermal gap matters here, regardless of sensor order.
  const delta = Math.abs(testAverage - refAverage);

  if (delta < 10) {
    return {
      delta,
      level: 0,
      label: "Normal",
      description: "Le tube test reste compatible avec un sous-vide normal, avec un écart thermique proche du comportement attendu.",
    };
  }

  if (delta < 18) {
    return {
      delta,
      level: 1,
      label: "Faible perte de vide",
      description: "L'écart thermique suggère une quantité minimale d'air dans le tube test et un début de dégradation du vide.",
    };
  }

  if (delta < 35) {
    return {
      delta,
      level: 2,
      label: "Perte de vide modérée",
      description: "L'écart thermique indique une perte de vide déjà significative, cohérente avec un tube test partiellement dégradé.",
    };
  }

  if (delta < 50) {
    return {
      delta,
      level: 3,
      label: "Perte importante de vide",
      description: "L'écart thermique est élevé et traduit une perte importante du vide dans le tube test.",
    };
  }

  return {
    delta,
    level: 4,
    label: "Perte complète du vide",
    description: "L'écart thermique est compatible avec une perte complète ou quasi complète du vide dans le tube test.",
  };
}

export function getAnomalyScore(anomaly: ThermalAnomaly | null | undefined) {
  const score = Number(anomaly?.anomaly_score ?? 0);
  return Number.isFinite(score) ? score : 0;
}

export function getCombinedScore(anomaly: CombinedAnomaly | null | undefined, fallback?: ThermalAnomaly | null | undefined) {
  const score = Number(anomaly?.anomaly_score ?? fallback?.anomaly_score ?? 0);
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

export function videoDecisionLabel(decision: VideoDecision | null | undefined) {
  switch ((decision || "").toLowerCase()) {
    case "normal":
      return "Normale";
    case "warning":
      return "À surveiller";
    case "anomaly":
      return "Anomalie";
    case "unknown":
      return "Indéterminée";
    default:
      return "--";
  }
}

export function confidenceLabel(confidence: VideoDecisionConfidence | null | undefined) {
  switch ((confidence || "").toLowerCase()) {
    case "high":
      return "Élevée";
    case "medium":
      return "Moyenne";
    case "low":
      return "Faible";
    default:
      return "--";
  }
}

function longestTrueRun(flags: boolean[]) {
  let longest = 0;
  let current = 0;
  for (const flag of flags) {
    if (flag) {
      current += 1;
      if (current > longest) longest = current;
    } else {
      current = 0;
    }
  }
  return longest;
}

function decisionConfidence(
  decision: VideoDecision,
  maxScore: number,
  averageScore: number,
  suspectFrameRatio: number,
  longestSuspectRun: number
): VideoDecisionConfidence {
  if (decision === "anomaly") {
    const strongAlert =
      maxScore >= VIDEO_DECISION_THRESHOLDS.frameAlertScore + 0.1 &&
      suspectFrameRatio >= Math.max(VIDEO_DECISION_THRESHOLDS.suspectRatioAlert, 0.15) &&
      longestSuspectRun >= VIDEO_DECISION_THRESHOLDS.minPersistentFrames;
    return strongAlert ? "high" : "medium";
  }

  if (decision === "warning") {
    const mediumWarning =
      maxScore >= VIDEO_DECISION_THRESHOLDS.frameWarningScore + 0.05 ||
      averageScore >= VIDEO_DECISION_THRESHOLDS.videoMeanAlertScore ||
      suspectFrameRatio >= VIDEO_DECISION_THRESHOLDS.suspectRatioAlert;
    return mediumWarning ? "medium" : "low";
  }

  return "high";
}

export function summarizePredictions(predictions: FramePrediction[]): AiSummary {
  const summary = predictions.reduce<AiSummary>(
    (summary, frame) => {
      const labels = frame.detections.map((detection) => (detection.class_name || "").toLowerCase());
      const score = getCombinedScore(frame.combined, frame.anomaly);
      const isThermalFrame = frame.anomaly?.status === "ok" || frame.combined?.status === "ok";
      const isAnomalyFrame = score >= VIDEO_DECISION_THRESHOLDS.frameWarningScore;
      const isAlertFrame = score >= VIDEO_DECISION_THRESHOLDS.frameAlertScore;
      const suspectSegments = frame.anomaly?.suspect_segments?.length || 0;
      const isPeak = score > summary.maxAnomalyScore;
      const autoencoderDecision = (frame.autoencoder?.decision || "").toLowerCase();
      const autoencoderRatio = Number(frame.autoencoder?.score_ratio ?? 0);
      const autoencoderDelta = Number(frame.autoencoder?.score_delta ?? 0);
      const autoencoderSupport = autoencoderDecision === "warning" || autoencoderDecision === "anomaly";
      const siameseDecision = (frame.siamese?.decision || "").toLowerCase();
      const siameseProbability = Number(frame.siamese?.pair_probability ?? frame.siamese?.support_score ?? 0);
      const siameseSupport = siameseDecision === "warning" || siameseDecision === "anomaly";

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
        suspectSegmentCount: summary.suspectSegmentCount + suspectSegments,
        suspectFrameRatio: summary.suspectFrameRatio,
        longestSuspectRun: summary.longestSuspectRun,
        warningFrameCount: summary.warningFrameCount + (isAnomalyFrame ? 1 : 0),
        alertFrameCount: summary.alertFrameCount + (isAlertFrame ? 1 : 0),
        decision: isPeak ? ((frame.combined?.decision as VideoDecision | undefined) || summary.decision) : summary.decision,
        decisionConfidence: isPeak
          ? ((frame.combined?.confidence as VideoDecisionConfidence | undefined) || summary.decisionConfidence)
          : summary.decisionConfidence,
        autoencoderSupportFrames: summary.autoencoderSupportFrames + (autoencoderSupport ? 1 : 0),
        autoencoderPeakRatio: Math.max(summary.autoencoderPeakRatio, Number.isFinite(autoencoderRatio) ? autoencoderRatio : 0),
        autoencoderAverageDelta: summary.autoencoderAverageDelta + (Number.isFinite(autoencoderDelta) ? autoencoderDelta : 0),
        siameseSupportFrames: summary.siameseSupportFrames + (siameseSupport ? 1 : 0),
        siamesePeakProbability: Math.max(summary.siamesePeakProbability, Number.isFinite(siameseProbability) ? siameseProbability : 0)
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
      suspectSegmentCount: 0,
      suspectFrameRatio: 0,
      longestSuspectRun: 0,
      warningFrameCount: 0,
      alertFrameCount: 0,
      decision: "unknown",
      decisionConfidence: "low",
      autoencoderSupportFrames: 0,
      autoencoderPeakRatio: 0,
      autoencoderAverageDelta: 0,
      siameseSupportFrames: 0,
      siamesePeakProbability: 0
    }
  );

  const warningFlags = predictions.map(
    (frame) => getCombinedScore(frame.combined, frame.anomaly) >= VIDEO_DECISION_THRESHOLDS.frameWarningScore
  );
  const averageAnomalyScore = summary.thermalFrames ? summary.averageAnomalyScore / summary.thermalFrames : 0;
  const suspectFrameRatio = summary.thermalFrames ? summary.warningFrameCount / summary.thermalFrames : 0;
  const longestSuspectRun = longestTrueRun(warningFlags);

  let decision: VideoDecision = "normal";
  if (
    summary.maxAnomalyScore >= VIDEO_DECISION_THRESHOLDS.frameAlertScore &&
    (suspectFrameRatio >= VIDEO_DECISION_THRESHOLDS.suspectRatioAlert ||
      longestSuspectRun >= VIDEO_DECISION_THRESHOLDS.minPersistentFrames)
  ) {
    decision = "anomaly";
  } else if (
    summary.maxAnomalyScore >= VIDEO_DECISION_THRESHOLDS.frameWarningScore ||
    averageAnomalyScore >= VIDEO_DECISION_THRESHOLDS.videoMeanWarningScore ||
    suspectFrameRatio >= VIDEO_DECISION_THRESHOLDS.suspectRatioWarning
  ) {
    decision = "warning";
  }

  return {
    ...summary,
    averageAnomalyScore,
    suspectFrameRatio,
    longestSuspectRun,
    decision: summary.decision === "unknown" ? decision : summary.decision,
    decisionConfidence: summary.decision === "unknown"
      ? decisionConfidence(
          decision,
          summary.maxAnomalyScore,
          averageAnomalyScore,
          suspectFrameRatio,
          longestSuspectRun
        )
      : summary.decisionConfidence,
    autoencoderAverageDelta: summary.sampledFrames ? summary.autoencoderAverageDelta / summary.sampledFrames : 0
  };
}

export function buildAnomalyNarrative(summary: AiSummary | null, hasAnalysis: boolean) {
  if (!hasAnalysis || !summary) {
    return {
      headline: "L’analyse n’a pas encore été lancée.",
      overview: "Après la segmentation, lancez l’analyse pour obtenir une explication simple de l’état du tube test.",
      details: [
        "Le moteur thermique compare le tube test au tube de référence pour voir s’il chauffe différemment.",
        "L’autoencoder vérifie si l’apparence du tube test reste proche d’un comportement normal appris.",
        "Le modèle Siamese compare directement les deux tubes pour confirmer ou nuancer l’alerte.",
      ],
    };
  }

  const headline =
    summary.decision === "anomaly"
      ? "Le tube test présente une anomalie probable."
      : summary.decision === "warning"
        ? "Le tube test présente un comportement à surveiller."
        : "Le tube test reste globalement compatible avec un état normal.";

  const overview =
    summary.decision === "anomaly"
      ? "Plusieurs indices convergent vers un écart réel entre le tube test et le tube de référence."
      : summary.decision === "warning"
        ? "Le système détecte un écart mesurable, mais moins net qu’une anomalie forte."
        : "Les modèles ne voient pas de divergence forte entre le tube test et le tube de référence.";

  const thermalDetail =
    summary.maxAnomalyScore >= 0.42
      ? `Thermique: écart marqué détecté, avec un pic de ${formatAnomalyScore(summary.maxAnomalyScore)} et une persistance de ${summary.longestSuspectRun} frame(s).`
      : summary.maxAnomalyScore >= 0.26
        ? `Thermique: écart modéré détecté, avec un pic de ${formatAnomalyScore(summary.maxAnomalyScore)}.`
        : `Thermique: pas d’écart fort détecté, le pic restant à ${formatAnomalyScore(summary.maxAnomalyScore)}.`;

  const autoencoderDetail =
    summary.autoencoderPeakRatio >= 1.25
      ? `Autoencoder: confirme fortement l’écart visuel ou thermique du tube test (ratio max ${formatAnomalyScore(summary.autoencoderPeakRatio)}).`
      : summary.autoencoderPeakRatio >= 1.1
        ? `Autoencoder: apporte un soutien modéré à l’alerte (ratio max ${formatAnomalyScore(summary.autoencoderPeakRatio)}).`
        : "Autoencoder: ne confirme pas fortement une différence anormale du tube test.";

  const siameseDetail =
    summary.siamesePeakProbability >= 0.6
      ? `Siamese: estime une différence forte entre tube-ref et tube-test (probabilité max ${formatAnomalyScore(summary.siamesePeakProbability)}).`
      : summary.siamesePeakProbability >= 0.35
        ? `Siamese: observe une différence perceptible entre les deux tubes (probabilité max ${formatAnomalyScore(summary.siamesePeakProbability)}).`
        : "Siamese: ne renforce pas fortement l’idée d’une différence anormale entre les deux tubes.";

  return {
    headline,
    overview,
    details: [thermalDetail, autoencoderDetail, siameseDetail],
  };
}
