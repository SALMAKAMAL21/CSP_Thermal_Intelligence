export type BackendJson = Record<string, unknown> | unknown[] | string | number | boolean | null;

export const DEFAULT_BACKEND_URL = "http://127.0.0.1:8002";
export const BACKEND_URL = (process.env.SEGMENTATION_API_URL || DEFAULT_BACKEND_URL).replace(/\/+$/, "");

export function createBackendUrl(path: string) {
  return new URL(path.replace(/^\/+/, ""), `${BACKEND_URL}/`);
}

export async function readBackendJson(response: Response): Promise<BackendJson> {
  const raw = await response.text();

  if (!raw.trim()) {
    if (response.ok) throw new Error("FastAPI renvoie une réponse vide. Vérifiez SEGMENTATION_API_URL.");
    return {};
  }

  try {
    return JSON.parse(raw) as BackendJson;
  } catch {
    if (response.ok) throw new Error("Le serveur configuré ne renvoie pas de JSON. Vérifiez que SEGMENTATION_API_URL pointe vers FastAPI.");
    return { message: `Réponse non JSON du backend (HTTP ${response.status}).` };
  }
}

export async function fetchBackend(input: URL | string, init: RequestInit = {}, timeoutMs = 60_000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(input, {
      ...init,
      cache: init.cache ?? "no-store",
      signal: init.signal ?? controller.signal
    });
  } finally {
    clearTimeout(timeout);
  }
}
