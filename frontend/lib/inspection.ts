export type InspectionDetails = {
  operator: string;
  tubeSerial: string;
  date: string;
  time: string;
  visualAnomaly: "yes" | "no" | "";
};

export const emptyInspection: InspectionDetails = {
  operator: "", tubeSerial: "", date: "", time: "", visualAnomaly: ""
};

export function inspectionMetadataComplete(details: InspectionDetails) {
  const parsedDate = new Date(`${details.date}T12:00:00Z`);
  const validDate = details.date === "" || (/^\d{4}-\d{2}-\d{2}$/.test(details.date) &&
    Number.isFinite(parsedDate.getTime()) && parsedDate.toISOString().slice(0, 10) === details.date);
  const validTime = details.time === "" || /^([01]\d|2[0-3]):[0-5]\d$/.test(details.time);
  return details.operator.trim().length > 0 && details.tubeSerial.trim().length > 0 && validDate && validTime;
}

export function inspectionComplete(details: InspectionDetails) {
  return inspectionMetadataComplete(details) && (details.visualAnomaly === "yes" || details.visualAnomaly === "no");
}

export function completeInspectionTimestamp(details: InspectionDetails, now = new Date()): InspectionDetails {
  const pad = (value: number) => String(value).padStart(2, "0");
  return {
    ...details,
    date: details.date || `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`,
    time: details.time || `${pad(now.getHours())}:${pad(now.getMinutes())}`
  };
}

export function temperaturesComplete(values: string[]) {
  return values.length >= 4 && values.every(value => value.trim() !== "" && Number.isFinite(Number(value))) &&
    Number.isFinite(values.reduce((sum, value) => sum + Number(value) / values.length, 0));
}
