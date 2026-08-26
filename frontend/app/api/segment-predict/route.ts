import { NextResponse } from "next/server";
import { BACKEND_URL, createBackendUrl, fetchBackend, readBackendJson } from "@/lib/backend";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export async function POST(req: Request) {
  try {
    const incoming = await req.formData();
    const image = incoming.get("image");
    const conf = incoming.get("conf")?.toString() ?? "0.25";
    const iou = incoming.get("iou")?.toString() ?? "0.45";
    const analyzeThermal = incoming.get("analyze_thermal")?.toString() ?? "true";
    const sessionId = incoming.get("session_id")?.toString();
    const resetTracker = incoming.get("reset_tracker")?.toString() ?? "false";
    const detectionsJson = incoming.get("detections_json")?.toString();

    if (!(image instanceof File)) {
      return NextResponse.json({ error: "Image manquante" }, { status: 400 });
    }

    const forward = new FormData();
    forward.append("image", image);
    if (detectionsJson) forward.append("detections_json", detectionsJson);

    const target = createBackendUrl(detectionsJson ? "analyze-anomaly" : "predict");
    if (!detectionsJson) {
      target.searchParams.set("conf", conf);
      target.searchParams.set("iou", iou);
      target.searchParams.set("analyze_thermal", analyzeThermal);
      if (sessionId) target.searchParams.set("session_id", sessionId);
      target.searchParams.set("reset_tracker", resetTracker);
    }

    const res = await fetchBackend(target, {
      method: "POST",
      body: forward
    });

    const data = await readBackendJson(res);

    if (!res.ok) {
      return NextResponse.json(
        {
          error: "Segmentation backend error",
          backendUrl: BACKEND_URL,
          backendStatus: res.status,
          backendResponse: data
        },
        { status: res.status }
      );
    }

    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    const message =
      error instanceof Error && error.name === "AbortError"
        ? "Timeout backend FastAPI pendant la prédiction"
        : error instanceof Error
          ? error.message
          : "Erreur inconnue";

    return NextResponse.json(
      {
        error: "Backend IA indisponible",
        details: message,
        backendUrl: BACKEND_URL
      },
      { status: 503 }
    );
  }
}
