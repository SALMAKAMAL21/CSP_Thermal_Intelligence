import Image from "next/image";
import logoImage from "../../logo/image.png";
import { ThemeToggle } from "@/components/theme-toggle";

export function AppHeader() {
  return (
    <header className="topbar" aria-label="En-tête application">
      <a className="brand-block" href="#source-video" aria-label="Green Energy Park - CSP Thermal">
        <Image className="brand-logo" src={logoImage} alt="Green Energy Park" priority />
        <div className="brand-copy">
          <strong>CSP Thermal Intelligence</strong>
          <span>Détection d&apos;anomalies par segmentation, comparaison thermique et autoencoder.</span>
        </div>
      </a>
      <ThemeToggle />
    </header>
  );
}
