import type { RefObject } from "react";
import { FiCpu, FiDownload } from "react-icons/fi";
import { Badge } from "@/components/ui/badge";
import { Button, ButtonLink } from "@/components/ui/button";
import { videoDecisionLabel, type AiSummary, type BackendConnectionStatus } from "@/lib/thermal";

type SegmentationSectionProps = {
  aiSummary: AiSummary | null;
  backendStatus: BackendConnectionStatus;
  canvasRef: RefObject<HTMLCanvasElement>;
  outUrl: string | null;
  running: boolean;
  videoPreviewUrl: string | null;
};

export function SegmentationSection({
  aiSummary,
  backendStatus,
  canvasRef,
  outUrl,
  running,
  videoPreviewUrl
}: SegmentationSectionProps) {
  const mediaUrl = outUrl || (!running ? videoPreviewUrl : null);
  const backendReady = backendStatus.reachable && backendStatus.modelLoaded;
  const status = running
    ? "En direct"
    : outUrl
      ? "Terminé"
      : backendStatus.checking
        ? "Connexion IA"
        : backendReady
          ? "Backend connecté"
          : backendStatus.reachable
            ? "Modèle absent"
            : "Backend hors ligne";
  const statusVariant: "live" | "danger" = running || outUrl || backendReady ? "live" : "danger";
  return (
    <section className="segmentation-section" id="inspection" aria-labelledby="segmentation-title">
      <div className="section-toolbar">
        <div>
          <h2 id="segmentation-title">Segmentation IA</h2>
          <div className="chip-row" aria-label="Statuts segmentation">
            <Badge variant={statusVariant} title={backendStatus.details ?? backendStatus.backendUrl}>
              <FiCpu aria-hidden="true" />
              {status}
            </Badge>
            <Badge variant={backendStatus.autoencoderLoaded ? "live" : "danger"}>
              AE {backendStatus.autoencoderLoaded ? "chargé" : "absent"}
            </Badge>
            {aiSummary && (
              <Badge variant={aiSummary.decision === "anomaly" ? "danger" : "live"}>
                {videoDecisionLabel(aiSummary.decision)}
              </Badge>
            )}
          </div>
        </div>
        {outUrl ? (
          <ButtonLink className="download-video-action" variant="outline" size="sm" href={outUrl} download="video-annotee-csp.webm">
            <FiDownload aria-hidden="true" />
            Vidéo traitée
          </ButtonLink>
        ) : (
          <Button className="download-video-action" variant="outline" size="sm" type="button" disabled>
            <FiDownload aria-hidden="true" />
            Vidéo traitée
          </Button>
        )}
      </div>

      <div className={`video-stage ${mediaUrl ? "has-media" : "is-empty"} ${running ? "is-running" : ""}`}>
        <canvas ref={canvasRef} className={`processing-canvas ${running ? "is-active" : ""}`} />
        {mediaUrl ? (
          <video
            key={mediaUrl}
            className="source-preview"
            controls
            playsInline
            preload="metadata"
            src={mediaUrl}
          />
        ) : (
          <>
            <div className="design-video-preview" aria-hidden="true" />
            <span className="visually-hidden">Vidéo en attente</span>
          </>
        )}
      </div>
    </section>
  );
}
