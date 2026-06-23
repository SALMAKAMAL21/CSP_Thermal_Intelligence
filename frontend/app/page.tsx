"use client";

import { AnomalySection } from "@/components/csp/anomaly-section";
import { AppHeader } from "@/components/csp/app-header";
import { ReportSection } from "@/components/csp/report-section";
import { SegmentationSection } from "@/components/csp/segmentation-section";
import { SiteFooter } from "@/components/csp/site-footer";
import { SourceVideoSection } from "@/components/csp/source-video-section";
import { TemperatureSection } from "@/components/csp/temperature-section";
import { useCspAnalysis } from "@/hooks/use-csp-analysis";

export default function HomePage() {
  const csp = useCspAnalysis();

  return (
    <main className="app-shell">
      <AppHeader />

      <SourceVideoSection
        error={csp.error}
        fileInputRef={csp.fileInputRef}
        outUrl={csp.outUrl}
        predictionCount={csp.predictionCount}
        progress={csp.progress}
        running={csp.running}
        step={csp.step}
        videoFile={csp.videoFile}
        onAnalyze={csp.handleSegmentationAction}
        onFileChange={csp.handleVideoChange}
      />

      <SegmentationSection
        aiSummary={csp.aiSummary}
        backendStatus={csp.backendStatus}
        canvasRef={csp.canvasRef}
        outUrl={csp.outUrl}
        running={csp.running}
        videoPreviewUrl={csp.videoPreviewUrl}
      />

      <AnomalySection
        aiSummary={csp.aiSummary}
        hasAnomalyAnalysis={csp.hasAnomalyAnalysis}
        predictionCount={csp.predictionCount}
        running={csp.running}
        onAnalyze={csp.handleAnomalyAnalysisAction}
      />

      <TemperatureSection
        refStats={csp.refStats}
        testStats={csp.testStats}
        tubeRefTemps={csp.tubeRefTemps}
        tubeTestTemps={csp.tubeTestTemps}
        onTemperatureChange={csp.updateTemperature}
      />

      <ReportSection
        canGenerateReport={csp.canGenerateReport}
        generatingReport={csp.generatingReport}
        reportStatus={csp.reportStatus}
        reportUrl={csp.reportUrl}
        onGenerate={csp.handleReportAction}
      />

      <SiteFooter />

      <video ref={csp.hiddenVideoRef} className="hidden-video" />
    </main>
  );
}
