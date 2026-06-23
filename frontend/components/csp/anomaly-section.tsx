import type { ReactNode } from "react";
import { FiAlertTriangle, FiFilm, FiGrid, FiActivity } from "react-icons/fi";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import type { AiSummary } from "@/lib/thermal";
import { formatAnomalyScore, formatTime, severityLabel } from "@/lib/thermal";

type AnomalySectionProps = {
  aiSummary: AiSummary | null;
  hasAnomalyAnalysis: boolean;
  predictionCount: number;
  running: boolean;
  onAnalyze: () => void;
};

export function AnomalySection({ aiSummary, hasAnomalyAnalysis, predictionCount, running, onAnalyze }: AnomalySectionProps) {
  const severity = aiSummary?.peakSeverity ?? "none";
  const anomalyLabel = hasAnomalyAnalysis && aiSummary ? severityLabel(aiSummary.peakSeverity) : "En attente";
  const anomalyFrames = hasAnomalyAnalysis && aiSummary ? aiSummary.anomalyFrames.toString().padStart(2, "0") : "--";
  const peakScore = hasAnomalyAnalysis && aiSummary ? formatAnomalyScore(aiSummary.maxAnomalyScore) : "--";
  const peakTime = hasAnomalyAnalysis && aiSummary ? formatTime(aiSummary.peakFrameTime) : "--";
  const segments = hasAnomalyAnalysis && aiSummary ? aiSummary.suspectSegmentCount.toString().padStart(2, "0") : "--";

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
          className={`severity-${severity}`}
          icon={<FiAlertTriangle />}
          iconTone="danger"
          label={anomalyLabel}
          value={anomalyFrames}
          description="Frames suspectes détectées"
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
          label="Segments suspects"
          value={segments}
          description="Zones thermiques retenues"
        />
      </div>
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
