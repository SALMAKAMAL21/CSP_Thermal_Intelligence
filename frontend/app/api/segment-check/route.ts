import { NextResponse } from "next/server";
import { BACKEND_URL, createBackendUrl, fetchBackend, readBackendJson } from "@/lib/backend";

export const dynamic = "force-dynamic";

type HealthPayload = {
  fusion_loaded?: boolean;
  fusion_error?: string;
  autoencoder_loaded?: boolean;
  model_loaded?: boolean;
  siamese_loaded?: boolean;
  status?: string;
};

type ModelInfoPayload = {
  autoencoder_model?: string | null;
  classes?: unknown;
  model?: string;
  siamese_model?: string | null;
  task?: string;
};

export async function GET() {
  try {
    const healthRes = await fetchBackend(createBackendUrl("health"), {}, 5_000);

    if (!healthRes.ok) {
      return NextResponse.json(
        {
          reachable: false,
          modelLoaded: false,
          backendUrl: BACKEND_URL,
          details: `Health endpoint status ${healthRes.status}`
        },
        { status: 200 }
      );
    }

    const health = (await readBackendJson(healthRes)) as HealthPayload;
    if (!health || typeof health !== "object" || typeof health.model_loaded !== "boolean") {
      throw new Error("Réponse /health invalide. Vérifiez que SEGMENTATION_API_URL pointe vers le backend FastAPI du projet.");
    }
    const modelLoaded = Boolean(health.model_loaded);
    const autoencoderLoaded = Boolean(health.autoencoder_loaded);
    const siameseLoaded = Boolean(health.siamese_loaded);
    let modelInfo: ModelInfoPayload = {};

    if (modelLoaded) {
      const modelRes = await fetchBackend(createBackendUrl("model/info"), {}, 5_000);
      if (modelRes.ok) modelInfo = (await readBackendJson(modelRes)) as ModelInfoPayload;
    }

    return NextResponse.json({
      reachable: true,
      fusionLoaded: Boolean(health.fusion_loaded),
      details: health.fusion_error,
      autoencoderLoaded,
      siameseLoaded,
      modelLoaded,
      backendUrl: BACKEND_URL,
      status: health.status ?? (modelLoaded ? "ok" : "no_model"),
      autoencoderModel: modelInfo.autoencoder_model ?? null,
      siameseModel: modelInfo.siamese_model ?? null,
      model: modelInfo.model,
      task: modelInfo.task,
      classes: modelInfo.classes
    });
  } catch (error) {
    const details =
      error instanceof Error && error.name === "AbortError"
        ? "Timeout backend FastAPI"
        : error instanceof Error
          ? error.message === "fetch failed" ? `Connexion à FastAPI impossible (${BACKEND_URL}). Démarrez le backend avec venv\\Scripts\\python.exe -m uvicorn src.inference.api:app --port 8002.` : error.message
          : "Connexion impossible";

    return NextResponse.json(
      {
        reachable: false,
        autoencoderLoaded: false,
        siameseLoaded: false,
        modelLoaded: false,
        backendUrl: BACKEND_URL,
        details
      },
      { status: 200 }
    );
  }
}
