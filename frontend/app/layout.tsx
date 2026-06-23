import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Green Energy Park - Analyse Tube CSP",
  description: "Upload video, temperatures, IA segmentation et rapport PDF généré"
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1
};

const themeScript = `
(() => {
  const storageKey = "gep-theme";
  const storedTheme = window.localStorage.getItem(storageKey);
  const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = storedTheme === "dark" || storedTheme === "light"
    ? storedTheme
    : systemPrefersDark
      ? "dark"
      : "light";

  document.documentElement.classList.toggle("dark", theme === "dark");
  document.documentElement.dataset.theme = theme;
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr" suppressHydrationWarning>
      <body>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        {children}
      </body>
    </html>
  );
}
