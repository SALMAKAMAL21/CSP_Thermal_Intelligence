import { FiDownload, FiFileText } from "react-icons/fi";
import { Button, ButtonLink } from "@/components/ui/button";
import { REPORT_WAITING_STATUS } from "@/lib/thermal";

type ReportSectionProps = {
  canGenerateReport: boolean;
  generatingReport: boolean;
  reportStatus: string;
  reportUrl: string | null;
  onGenerate: () => void;
};

export function ReportSection({
  canGenerateReport,
  generatingReport,
  reportStatus,
  reportUrl,
  onGenerate
}: ReportSectionProps) {
  return (
    <section className="report-section" id="rapport" aria-labelledby="report-title">
      <div className={`report-card ${canGenerateReport ? "is-ready" : ""}`}>
        <div className="report-copy">
          <h2 id="report-title">Rapport d'Inspection</h2>
          <p>Une page avec le niveau prédit, les températures et une image représentative.</p>
          <div className="report-actions">
            <Button className="report-generate-action" variant="dark" size="md" type="button" onClick={onGenerate} disabled={generatingReport || !canGenerateReport}>
              <FiFileText aria-hidden="true" />
              {generatingReport ? "Génération" : "Générer PDF"}
            </Button>
            {reportUrl ? (
              <ButtonLink variant="outline" size="sm" href={reportUrl} download="rapport_thermique_csp.pdf">
                <FiDownload aria-hidden="true" />
                Rapport
              </ButtonLink>
            ) : (
              <Button variant="outline" size="sm" type="button" disabled>
                <FiDownload aria-hidden="true" />
                Rapport
              </Button>
            )}
          </div>
        </div>
        <div className="report-visual" aria-hidden="true">
          <span />
          <span />
          <span />
          <span />
        </div>
      </div>

      {(generatingReport || reportUrl || reportStatus !== REPORT_WAITING_STATUS) && (
        <div className="summary-box report-status" aria-live="polite">
          {reportStatus}
        </div>
      )}

      {reportUrl && (
        <div className="output-panel pdf-preview-panel">
          <div className="pdf-preview-heading">
            <div>
              <h3>Aperçu du rapport PDF</h3>
              <p>Prévisualisation du rapport généré avant téléchargement.</p>
            </div>
            <ButtonLink variant="primary" size="md" href={reportUrl} download="rapport_thermique_csp.pdf">
              <FiDownload aria-hidden="true" />
              Télécharger le rapport
            </ButtonLink>
          </div>
          <div className="pdf-viewer-shell">
            <iframe className="pdf-preview" src={reportUrl} title="Prévisualisation du rapport PDF" />
          </div>
        </div>
      )}
    </section>
  );
}
