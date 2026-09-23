import { FiInfo } from "react-icons/fi";

import { inspectionResults as results } from "@/lib/inspection-results";


export function InspectionResult({ level }: { level: number | null }) {
  if (level === null || !Number.isInteger(level) || level < 0 || level >= results.length) return null;
  const result = results[level];
  return (
    <div className={`operator-result state-${result.tone}`}>
      <div className="operator-state">
        <span className="state-dot" aria-hidden="true" />
        <div><strong>{result.title}</strong><p>{result.description}</p></div>
      </div>
      <ul className="operator-findings">
        {result.findings.map(finding => <li key={finding}><span className="finding-dot" aria-hidden="true" /><span>{finding}</span></li>)}
      </ul>
      <div className="interpretation"><FiInfo aria-hidden="true" /><div><h4>Interprétation</h4><p>{result.interpretation}</p></div></div>
    </div>
  );
}
