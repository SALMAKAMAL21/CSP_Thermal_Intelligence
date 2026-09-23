import type { jsPDF } from "jspdf";
import type { AiSummary } from "./thermal";
import { inspectionComplete, type InspectionDetails } from "./inspection";
import { inspectionResults } from "./inspection-results";

export type ReportCapture = { dataUrl: string; width: number; height: number; time: number };
export type ReportMeasurements = { ref: number[]; test: number[] };

export function thermalTableStats(measurements: ReportMeasurements) {
  const { ref, test } = measurements;
  if (ref.length < 4 || ref.length !== test.length || ![...ref, ...test].every(Number.isFinite)) {
    throw new Error("Les mesures thermiques doivent former au moins quatre paires valides.");
  }
  const deltas = test.map((value, index) => Math.abs(ref[index] - value));
  const mean = deltas.reduce((sum, value) => sum + value / deltas.length, 0);
  if (![...deltas, mean].every(Number.isFinite)) throw new Error("Statistiques thermiques invalides.");
  return { deltas, mean };
}

/** Diagnostic inspiré du modèle 2, avec écarts absolus et capture du milieu de la vidéo. */
export function drawFusionReport(doc: jsPDF, summary: AiSummary, capture: ReportCapture | null, measurements: ReportMeasurements, inspection: InspectionDetails) {
  if (summary.level === null || !Number.isInteger(summary.level) || !inspectionResults[summary.level]) throw new Error("Classification indisponible");
  if (!inspectionComplete(inspection) || !inspection.date || !inspection.time) throw new Error("Informations d’inspection incomplètes.");
  const result = inspectionResults[summary.level];
  const stats = thermalTableStats(measurements);
  const left = 16, right = 194, width = right - left, pageBottom = 272;
  const date = inspection.date.split("-").reverse().join("/");
  const colors: [number, number, number][] = [[53, 151, 71], [190, 151, 0], [225, 143, 39], [201, 91, 16], [201, 41, 41]];
  const color = colors[summary.level];
  const format = (value: number) => value.toLocaleString("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 2, useGrouping: false });
  const lines = (value: string, maxWidth: number, size = 9): string[] => {
    doc.setFontSize(size);
    return doc.splitTextToSize(value, maxWidth);
  };
  const drawHeader = (continuation = false) => {
    doc.setFont("helvetica", "normal"); doc.setTextColor(85, 85, 85); doc.setFontSize(10);
    doc.text("GREEN ENERGY PARK", 105, 14, { align: "center" });
    doc.setFont("helvetica", "bold"); doc.setTextColor(20, 24, 22); doc.setFontSize(16);
    doc.text("RAPPORT DE DIAGNOSTIC THERMIQUE", 105, 22, { align: "center" });
    doc.setFont("helvetica", "normal"); doc.setTextColor(85, 85, 85); doc.setFontSize(9);
    doc.text(continuation ? "Suite du rapport d’inspection" : "Détection d’anomalies sur tube absorbeur CSP", 105, 28, { align: "center" });
    doc.setDrawColor(53, 105, 71); doc.setLineWidth(.5); doc.line(left, 32, right, 32);
    doc.setTextColor(25, 30, 27);
  };
  let y = 35;
  const newPage = () => {
    doc.addPage(); drawHeader(true);
    doc.setFontSize(8);
    const identification = lines(`Tube : ${inspection.tubeSerial} | ${date} à ${inspection.time}`, width, 8);
    doc.text(identification, left, 43);
    y = 47 + identification.length * 3.3;
  };
  const ensureSpace = (height: number) => {
    if (y + height > pageBottom) newPage();
  };

  doc.setProperties({ title: "Rapport de diagnostic thermique CSP", subject: `Tube ${inspection.tubeSerial}`, author: inspection.operator });
  drawHeader();
  // Informations d'inspection : deux colonnes comme le document de référence.
  doc.setFont("helvetica", "normal");
  const operatorLines = lines(`Opérateur : ${inspection.operator.trim()}`, 83);
  const serialLines = lines(`Numéro de série du tube : ${inspection.tubeSerial.trim()}`, 83);
  const identityHeight = Math.max(10, Math.max(operatorLines.length, serialLines.length) * 3.7 + 5);
  doc.setFillColor(247, 249, 247); doc.roundedRect(left, y, width, 9 + identityHeight, 2, 2, "F");
  doc.setDrawColor(219, 226, 220);
  doc.line(105, y, 105, y + 9 + identityHeight);
  doc.line(left, y + 9, right, y + 9);
  doc.setFontSize(9);
  doc.text(`Date : ${date}`, left + 3, y + 5.8);
  doc.text(`Heure : ${inspection.time}`, 108, y + 5.8);
  doc.text(operatorLines, left + 3, y + 14.5);
  doc.text(serialLines, 108, y + 14.5);
  y += 9 + identityHeight + 5;

  ensureSpace(26);
  const tint = color.map(channel => Math.round(255 * .92 + channel * .08)) as [number, number, number];
  doc.setFillColor(...tint); doc.roundedRect(left, y, width, 21, 3, 3, "F");
  doc.setFillColor(...color); doc.roundedRect(left, y + 3, 1, 15, .5, .5, "F");
  doc.setFont("helvetica", "bold"); doc.setTextColor(...color); doc.setFontSize(11.5);
  doc.text(`Résultat du diagnostic : ${result.title}`, left + 6, y + 9);
  doc.setFont("helvetica", "normal"); doc.setFontSize(10); doc.text(result.description, left + 6, y + 17);
  doc.setTextColor(25, 30, 27); y += 26;

  const rowHeight = 6, headerHeight = 11;
  doc.setFont("helvetica", "normal");
  const interpretationLines = lines(result.interpretation, width - 8, 9);
  const interpretationHeight = Math.max(18, 10 + interpretationLines.length * 3.8);
  const hasCapture = capture !== null && Number.isFinite(capture.width) &&
    Number.isFinite(capture.height) && capture.width > 0 && capture.height > 0;
  const aspectRatio = hasCapture ? capture.width / capture.height : width / 60;
  const preferredImageHeight = width / aspectRatio;
  // Donner à l’image tout l’espace restant après réservation du tableau et de l’interprétation.
  // Les rapports usuels tiennent sur une page ; les longues séries de mesures restent paginées.
  const imageSectionSpacing = 15;
  const tableHeight = 4 + headerHeight + measurements.ref.length * rowHeight + 8;
  const availableImageHeight = pageBottom - y - imageSectionSpacing - tableHeight - interpretationHeight - 15;
  const imageHeight = Math.min(preferredImageHeight, Math.max(45, availableImageHeight));
  ensureSpace(imageHeight + imageSectionSpacing);
  const imageWidth = imageHeight * aspectRatio;
  const imageLeft = left + (width - imageWidth) / 2;
  doc.setFont("helvetica", "bold"); doc.setFontSize(10); doc.text("IMAGE THERMIQUE DES TUBES", 105, y, { align: "center" });
  y += 4;
  // Le cadre prend les proportions de la capture : aucun recadrage ni étirement des tubes.
  if (hasCapture) {
    doc.addImage(capture.dataUrl, "JPEG", imageLeft, y, imageWidth, imageHeight);
  } else {
    doc.setFillColor(245, 247, 245); doc.rect(imageLeft, y, imageWidth, imageHeight, "F");
    doc.setFont("helvetica", "normal"); doc.setFontSize(9); doc.text("Capture indisponible", 105, y + imageHeight / 2, { align: "center" });
  }
  y += imageHeight + 4;
  doc.setFont("helvetica", "normal"); doc.setTextColor(85, 95, 89); doc.setFontSize(7.5);
  const instant = capture ? `${Math.floor(capture.time / 60)} min ${(capture.time % 60).toFixed(2)} s` : "indisponible";
  doc.text(`Capture à mi-parcours : ${instant} | Référence : blanc | Tube test : vert`, left, y);
  doc.setTextColor(25, 30, 27); y += 7;

  const columns = [16, 32, 67, 102, 153, 194];
  let start = 0;
  while (start < measurements.ref.length) {
    ensureSpace(7 + headerHeight + Math.min(4, measurements.ref.length - start) * rowHeight + 9);
    doc.setFont("helvetica", "bold"); doc.setFontSize(10);
    doc.text(start === 0 ? "Informations thermiques" : "Informations thermiques - suite", left, y);
    y += 4;
    const count = Math.min(measurements.ref.length - start, Math.floor((pageBottom - y - headerHeight - 9) / rowHeight));
    const bottom = y + headerHeight + count * rowHeight;
    doc.setFillColor(242, 246, 242); doc.rect(left, y, width, headerHeight, "F");
    const labels = ["Point", "Tube réf.\n(°C)", "Tube test\n(°C)", "Écart thermique\n|T_ref - T_test| (°C)", "Moyenne des\nécarts absolus (°C)"];
    doc.setFontSize(8);
    labels.forEach((label, index) => doc.text(label, columns[index] + 2, y + 4.5));
    doc.setFont("helvetica", "normal"); doc.setFontSize(9);
    for (let row = 0; row < count; row++) {
      const index = start + row, rowY = y + headerHeight + row * rowHeight;
      if (row % 2 === 1) { doc.setFillColor(249, 250, 249); doc.rect(left, rowY, 137, rowHeight, "F"); }
      const values = [`T${index + 1}`, format(measurements.ref[index]), format(measurements.test[index]), format(stats.deltas[index])];
      values.forEach((value, col) => {
        // Réduire la police pour conserver une mesure longue dans sa cellule.
        doc.setFontSize(9);
        const available = columns[col + 1] - columns[col] - 4;
        if (doc.getTextWidth(value) > available) doc.setFontSize(9 * available / doc.getTextWidth(value));
        doc.text(value, columns[col] + 2, rowY + 4.6);
      });
    }
    doc.setFontSize(9);
    const centerY = y + headerHeight + count * rowHeight / 2 + 1;
    doc.setFont("helvetica", "bold"); doc.text(format(stats.mean), 173.5, centerY, { align: "center" });
    doc.setFont("helvetica", "normal");
    doc.setDrawColor(191, 201, 193); doc.setLineWidth(.2);
    doc.rect(left, y, width, bottom - y);
    columns.slice(1, -1).forEach(x => doc.line(x, y, x, bottom));
    doc.line(left, y + headerHeight, right, y + headerHeight);
    for (let row = 1; row < count; row++) doc.line(left, y + headerHeight + row * rowHeight, 153, y + headerHeight + row * rowHeight);
    doc.setFontSize(7); doc.setTextColor(85, 95, 89);
    doc.text(`Moyenne calculée sur les ${measurements.ref.length} écarts absolus.`, left, bottom + 4.5);
    doc.setTextColor(25, 30, 27);
    y = bottom + 8;
    start += count;
    if (start < measurements.ref.length) newPage();
  }

  doc.setFont("helvetica", "normal");
  ensureSpace(interpretationHeight + 15);
  doc.setFont("helvetica", "bold"); doc.setFontSize(10); doc.text("INTERPRÉTATION", left, y + 4);
  doc.setDrawColor(190, 203, 194); doc.line(left, y + 7, right, y + 7);
  doc.setFont("helvetica", "normal"); doc.setFontSize(9); doc.text(interpretationLines, left, y + 13);
  y += interpretationHeight + 11;
  doc.text("Validation du technicien :", left, y);
  doc.setDrawColor(120, 130, 123); doc.line(left + 42, y + .5, left + 110, y + .5);

  const pageCount = doc.getNumberOfPages();
  for (let page = 1; page <= pageCount; page++) {
    doc.setPage(page); doc.setDrawColor(200, 205, 202); doc.line(left, 281, right, 281);
    doc.setFont("helvetica", "normal"); doc.setTextColor(85, 95, 89); doc.setFontSize(8);
    doc.text("© Copyright Green Energy Park. Tous droits réservés 2026.", left, 287);
    doc.text(`${page} / ${pageCount}`, right, 287, { align: "right" });
  }
}

