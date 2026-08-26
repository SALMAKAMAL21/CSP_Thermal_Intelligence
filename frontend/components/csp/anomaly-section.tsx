import type { ReactNode } from "react";
import { FiAlertTriangle, FiFilm, FiGrid, FiActivity } from "react-icons/fi";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import type { AiSummary } from "@/lib/thermal";
import { buildAnomalyNarrative, confidenceLabel, formatAnomalyScore, formatTime, severityLabel, videoDecisionLabel } from "@/lib/thermal";

type AnomalySectionProps = {
  aiSummary: AiSummary | null;
  hasAnomalyAnalysis: boolean;
  predictionCount: number;
  running: boolean;
  onAnalyze: () => void;
};

export function AnomalySection({ aiSummary, hasAnomalyAnalysis, predictionCount, running, onAnalyze }: AnomalySectionProps) {
  const decision = aiSummary?.decision ?? "unknown";
  const anomalyLabel = hasAnomalyAnalysis && aiSummary ? videoDecisionLabel(aiSummary.decision) : "En attente";
  const anomalyFrames = hasAnomalyAnalysis && aiSummary ? aiSummary.warningFrameCount.toString().padStart(2, "0") : "--";
  const peakScore = hasAnomalyAnalysis && aiSummary ? formatAnomalyScore(aiSummary.maxAnomalyScore) : "--";
  const peakTime = hasAnomalyAnalysis && aiSummary ? formatTime(aiSummary.peakFrameTime) : "--";
  const segments = hasAnomalyAnalysis && aiSummary ? aiSummary.longestSuspectRun.toString().padStart(2, "0") : "--";
  const confidence = hasAnomalyAnalysis && aiSummary ? confidenceLabel(aiSummary.decisionConfidence) : "--";
  const aeRatio = hasAnomalyAnalysis && aiSummary ? aiSummary.autoencoderPeakRatio.toFixed(2) : "--";
  const narrative = buildAnomalyNarrative(aiSummary, hasAnomalyAnalysis);

  return (
    <section className="analysis-section" aria-labelledby="analysis-title">
      <div className="section-copy is-centered">
        <h2 id="analysis-title">Analyse d'Anomalies</h2>
      </div>

      <div className="section-action-row">
        <Button className="workflow-action anomaly-action" type="button" size="md" onClick={onAnalyze} disabled={running}>
          <span>{running ? "Analyse..." : "Lancer l'analyse"}</span>
          <FiActivity aria-hidden="true" />
        </Button>
      </div>

      <div className="analysis-grid">
        <AnalysisCard
          className={`severity-${decision}`}
          icon={<FiAlertTriangle />}
          iconTone="danger"
          label={anomalyLabel}
          value={anomalyFrames}
          description={`Confiance ${confidence}`}
        />
        <AnalysisCard
          icon={<FiFilm />}
          iconTone="primary"
          label="Score max"
          value={peakScore}
          description={hasAnomalyAnalysis ? `Pic à ${peakTime}` : `${predictionCount} frames disponibles`}
        />
        <AnalysisCard
          icon={<FiGrid />}
          label="Persistance"
          value={segments}
          description={
            hasAnomalyAnalysis && aiSummary
              ? `${severityLabel(aiSummary.peakSeverity)} · AE ratio max ${aeRatio}`
              : "Séquence suspecte continue"
          }
        />
      </div>

      <Card className="analysis-explainer">
        <div className="analysis-explainer-head">
          <span className={`analysis-explainer-badge is-${decision}`}>{hasAnomalyAnalysis && aiSummary ? videoDecisionLabel(aiSummary.decision) : "Analyse guidée"}</span>
          <span>Lecture simplifiée pour opérateurs terrain</span>
        </div>
        <div className="analysis-explainer-copy">
          <h3>{narrative.headline}</h3>
          <p>{narrative.overview}</p>
        </div>
        <div className="analysis-explainer-list">
          {narrative.details.map((detail) => (
            <div key={detail} className="analysis-explainer-item">
              <span className="analysis-explainer-dot" aria-hidden="true" />
              <p>{detail}</p>
            </div>
          ))}
        </div>
      </Card>
    </section>
  );
}

type AnalysisCardProps = {
  className?: string;
  description: string;
  icon: ReactNode;
  iconTone?: "danger" | "primary";
  label: string;
  value: string;
};

function AnalysisCard({ className, description, icon, iconTone, label, value }: AnalysisCardProps) {
  return (
    <Card className={`analysis-card ${className ?? ""}`}>
      <div className="analysis-card-top">
        <span className={`analysis-icon ${iconTone ? `is-${iconTone}` : ""}`} aria-hidden="true">
          {icon}
        </span>
        <span>{label}</span>
      </div>
      <strong>{value}</strong>
      <p>{description}</p>
    </Card>
  );
}
