import { NextResponse } from "next/server";
import { BACKEND_URL, createBackendUrl, fetchBackend, readBackendJson } from "@/lib/backend";

export const dynamic = "force-dynamic";

type HealthPayload = {
  model_loaded?: boolean;
  status?: string;
};

type ModelInfoPayload = {
  classes?: unknown;
  model?: string;
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
    let modelInfo: ModelInfoPayload = {};

    if (modelLoaded) {
      const modelRes = await fetchBackend(createBackendUrl("model/info"), {}, 5_000);
      if (modelRes.ok) modelInfo = (await readBackendJson(modelRes)) as ModelInfoPayload;
    }

    return NextResponse.json({
      reachable: true,
      modelLoaded,
      backendUrl: BACKEND_URL,
      status: health.status ?? (modelLoaded ? "ok" : "no_model"),
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
        modelLoaded: false,
        backendUrl: BACKEND_URL,
        details
      },
      { status: 200 }
    );
  }
}
