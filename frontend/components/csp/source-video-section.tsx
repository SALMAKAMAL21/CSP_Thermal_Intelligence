import type { ChangeEvent, RefObject } from "react";
import { FiCpu, FiUploadCloud } from "react-icons/fi";
import { Button } from "@/components/ui/button";
import { formatFileSize } from "@/lib/thermal";

type SourceVideoSectionProps = {
  error: string | null;
  fileInputRef: RefObject<HTMLInputElement>;
  outUrl: string | null;
  predictionCount: number;
  progress: number;
  running: boolean;
  step: string;
  videoFile: File | null;
  onAnalyze: () => void;
  onFileChange: (event: ChangeEvent<HTMLInputElement>) => void;
};

export function SourceVideoSection({
  error,
  fileInputRef,
  outUrl,
  predictionCount,
  progress,
  running,
  step,
  videoFile,
  onAnalyze,
  onFileChange
}: SourceVideoSectionProps) {
  return (
    <section className="source-section" id="source-video" aria-labelledby="source-title">
      <div className="section-copy is-centered">
        <h1 id="source-title">Vidéo Source</h1>
        <p>Importez vos séquences thermiques pour l'analyse automatisée par IA.</p>
      </div>

      <div className="upload-card">
        <label className={`upload-zone ${videoFile ? "has-file" : ""} ${running ? "is-processing" : ""}`} htmlFor="video">
          <input
            id="video"
            ref={fileInputRef}
            className="file-input"
            type="file"
            accept="video/*"
            onChange={onFileChange}
          />
          <span className="upload-icon" aria-hidden="true">
            <FiUploadCloud />
          </span>
          <strong className="upload-title">{videoFile ? videoFile.name : "Glissez-déposez votre vidéo ici"}</strong>
          <span className="upload-detail">{videoFile ? formatFileSize(videoFile.size) : "Format MP4, MOV ou AVI (Max 500Mo)"}</span>
        </label>

        <Button className="workflow-action segmentation-action" variant="dark" type="button" size="md" onClick={onAnalyze} disabled={running}>
          <span>{running ? "Segmentation..." : "Lancer la segmentation"}</span>
          <FiCpu aria-hidden="true" />
        </Button>

        {(running || progress > 0 || predictionCount > 0 || outUrl) && (
          <div className="progress-stack" aria-label="Progression traitement">
            <div className="progress-copy">
              <span>{step}</span>
              <strong>{progress}%</strong>
            </div>
            <div className="progress-track">
              <span style={{ width: `${progress}%` }} />
            </div>
            <div className="micro-stats">
              <span>{predictionCount} frames IA</span>
              <span>{outUrl ? "Sortie disponible" : "Sortie en attente"}</span>
            </div>
          </div>
        )}

        {error && <p className="error-message">Erreur: {error}</p>}
      </div>
    </section>
  );
}
