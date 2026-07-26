import Image from "next/image";
import logoImage from "../../logo/image.png";

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="footer-center">
        <Image className="footer-logo" src={logoImage} alt="Green Energy Park" />
        <span>© Copyright Green Energy Park. Tous droits réservés 2026.</span>
      </div>
    </footer>
  );
}
