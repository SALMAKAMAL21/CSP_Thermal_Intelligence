import { NextResponse } from "next/server";
import { BACKEND_URL, createBackendUrl, fetchBackend, readBackendJson } from "@/lib/backend";

export const dynamic = "force-dynamic";

type HealthPayload = {
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
          ? error.message
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
