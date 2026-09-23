import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FiPlus } from "react-icons/fi";
import {
  getTemperaturePlaceholder,
  getTemperaturePointLabel,
  REQUIRED_TEMPERATURE_COUNT,
  type TemperatureStats
} from "@/lib/thermal";

type TemperatureSectionProps = {
  disabled?: boolean;
  refStats: TemperatureStats;
  testStats: TemperatureStats;
  tubeRefTemps: string[];
  tubeTestTemps: string[];
  onAddTemperatureField: () => void;
  onTemperatureChange: (tube: "ref" | "test", index: number, value: string) => void;
};

export function TemperatureSection({
  disabled,
  refStats,
  testStats,
  tubeRefTemps,
  tubeTestTemps,
  onAddTemperatureField,
  onTemperatureChange
}: TemperatureSectionProps) {
  return (
    <section className="temperature-section" id="temperatures" aria-labelledby="temperature-title">
      <div className="section-copy is-centered">
        <h2 id="temperature-title">Saisie Thermique</h2>
      </div>

      <div className="tube-temperature-list">
        <TubeTemperatureCard
          disabled={disabled}
          complete={refStats.completed >= REQUIRED_TEMPERATURE_COUNT}
          label="Tube référence"
          tone="ref"
          values={tubeRefTemps}
          onChange={(index, value) => onTemperatureChange("ref", index, value)}
        />
        <TubeTemperatureCard
          disabled={disabled}
          complete={testStats.completed >= REQUIRED_TEMPERATURE_COUNT}
          label="Tube test"
          tone="test"
          values={tubeTestTemps}
          onChange={(index, value) => onTemperatureChange("test", index, value)}
        />
      </div>

      <div className="temperature-add-row">
        <Button
          className="temperature-add-shared"
          variant="outline"
          size="sm"
          type="button"
          disabled={disabled}
          onClick={onAddTemperatureField}
          aria-label="Ajouter un champ thermique aux deux tubes"
        >
          <FiPlus aria-hidden="true" />
        </Button>
      </div>
    </section>
  );
}

type TubeTemperatureCardProps = {
  disabled?: boolean;
  complete: boolean;
  label: string;
  onChange: (index: number, value: string) => void;
  tone: "ref" | "test";
  values: string[];
};

function TubeTemperatureCard({ disabled, complete, label, onChange, tone, values }: TubeTemperatureCardProps) {
  return (
    <Card className={`tube-card ${tone === "test" ? "test-card" : "ref-card"} ${complete ? "is-complete" : ""}`} aria-label={`Températures ${tone}`}>
      <div className="tube-card-header">
        <div>
          <h3>{label}</h3>
        </div>

      </div>

      <div className="measure-grid">
        {values.map((value, index) => {
          const point = getTemperaturePointLabel(index);
          return (
          <label className={`measure-input ${value ? "is-filled" : ""}`} htmlFor={`tube-${tone}-${point}`} key={point}>
            <span>{point} (°C)</span>
            <Input
              id={`tube-${tone}-${point}`}
              disabled={disabled}
              type="number"
              required
              step="any"
              inputMode="decimal"
              placeholder={getTemperaturePlaceholder(tone, index)}
              value={value}
              onChange={(event) => onChange(index, event.target.value)}
            />
          </label>
        )})}
      </div>
    </Card>
  );
}
