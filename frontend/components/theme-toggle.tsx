"use client";

import { useEffect, useState } from "react";
import { FiMoon, FiSun } from "react-icons/fi";
import { Button } from "@/components/ui/button";

const THEME_STORAGE_KEY = "gep-theme";

type Theme = "light" | "dark";

function resolveTheme(theme: Theme) {
  const root = document.documentElement;
  root.classList.toggle("dark", theme === "dark");
  root.dataset.theme = theme;
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("light");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
    const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const initialTheme: Theme = storedTheme === "dark" || storedTheme === "light" ? storedTheme : systemPrefersDark ? "dark" : "light";

    resolveTheme(initialTheme);
    setTheme(initialTheme);
    setMounted(true);
  }, []);

  function toggleTheme() {
    const nextTheme: Theme = theme === "dark" ? "light" : "dark";
    resolveTheme(nextTheme);
    window.localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
    setTheme(nextTheme);
  }

  if (!mounted) {
    return (
      <Button className="theme-toggle" variant="outline" size="sm" type="button" disabled aria-hidden="true">
        <FiMoon aria-hidden="true" />
        <span>Thème</span>
      </Button>
    );
  }

  const isDark = theme === "dark";

  return (
    <Button
      className="theme-toggle"
      variant="outline"
      size="sm"
      type="button"
      onClick={toggleTheme}
      aria-label={isDark ? "Activer le mode clair" : "Activer le mode nuit"}
      aria-pressed={isDark}
    >
      {isDark ? <FiSun aria-hidden="true" /> : <FiMoon aria-hidden="true" />}
      <span>{isDark ? "Jour" : "Nuit"}</span>
    </Button>
  );
}
