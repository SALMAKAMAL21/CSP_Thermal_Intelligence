import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  formatAverage,
  REF_TEMP_PLACEHOLDERS,
  TEMPERATURE_POINTS,
  TEST_TEMP_PLACEHOLDERS,
  type TemperatureStats
} from "@/lib/thermal";

type TemperatureSectionProps = {
  refStats: TemperatureStats;
  testStats: TemperatureStats;
  tubeRefTemps: string[];
  tubeTestTemps: string[];
  onTemperatureChange: (tube: "ref" | "test", index: number, value: string) => void;
};

export function TemperatureSection({
  refStats,
  testStats,
  tubeRefTemps,
  tubeTestTemps,
  onTemperatureChange
}: TemperatureSectionProps) {
  return (
    <section className="temperature-section" id="temperatures" aria-labelledby="temperature-title">
      <div className="section-copy is-centered">
        <h2 id="temperature-title">Saisie Thermique</h2>
      </div>

      <div className="tube-temperature-list">
        <TubeTemperatureCard
          average={formatAverage(refStats.average)}
          complete={refStats.completed === 4}
          label="Tube Référence"
          tone="ref"
          values={tubeRefTemps}
          placeholders={REF_TEMP_PLACEHOLDERS}
          onChange={(index, value) => onTemperatureChange("ref", index, value)}
        />
        <TubeTemperatureCard
          average={formatAverage(testStats.average)}
          complete={testStats.completed === 4}
          label="Tube Test"
          tone="test"
          values={tubeTestTemps}
          placeholders={TEST_TEMP_PLACEHOLDERS}
          onChange={(index, value) => onTemperatureChange("test", index, value)}
        />
      </div>
    </section>
  );
}

type TubeTemperatureCardProps = {
  average: string;
  complete: boolean;
  label: string;
  onChange: (index: number, value: string) => void;
  placeholders: readonly string[];
  tone: "ref" | "test";
  values: string[];
};

function TubeTemperatureCard({ average, complete, label, onChange, placeholders, tone, values }: TubeTemperatureCardProps) {
  return (
    <Card className={`tube-card ${tone === "test" ? "test-card" : "ref-card"} ${complete ? "is-complete" : ""}`} aria-label={`Températures ${tone}`}>
      <div className="tube-card-header">
        <h3>{label}</h3>
        <div className="tube-average">
          <span>Moyenne</span>
          <strong>{average}</strong>
        </div>
      </div>

      <div className="measure-grid">
        {TEMPERATURE_POINTS.map((point, index) => (
          <label className={`measure-input ${values[index] ? "is-filled" : ""}`} htmlFor={`tube-${tone}-${point}`} key={point}>
            <span>{point} (°C)</span>
            <Input
              id={`tube-${tone}-${point}`}
              type="number"
              inputMode="decimal"
              placeholder={placeholders[index]}
              value={values[index]}
              onChange={(event) => onChange(index, event.target.value)}
            />
          </label>
        ))}
      </div>
    </Card>
  );
}
