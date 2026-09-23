/** Attend le décodage de l'image demandée avant toute capture du média source. */
export async function seekVideoFrame(video: HTMLVideoElement, time: number): Promise<void> {
  if (!Number.isFinite(time) || time < 0 || !Number.isFinite(video.duration) || time > video.duration) {
    throw new Error("Instant de capture vidéo invalide.");
  }
  if (!video.seeking && Math.abs(video.currentTime - time) < 0.001 && video.readyState >= 2) return;
  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
    };
    const onSeeked = () => {
      cleanup();
      if (video.readyState < 2) reject(new Error("Image vidéo non décodée."));
      else resolve();
    };
    const onError = () => { cleanup(); reject(new Error("Lecture de l’image vidéo impossible.")); };
    const timeout = window.setTimeout(() => { cleanup(); reject(new Error("Délai de capture vidéo dépassé.")); }, 15000);
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("error", onError);
    try { video.currentTime = time; } catch (error) { cleanup(); reject(error); }
  });
}

export async function captureVideoFrame(video: HTMLVideoElement, time: number): Promise<HTMLCanvasElement> {
  await seekVideoFrame(video, time);
  if (!video.videoWidth || !video.videoHeight) throw new Error("Dimensions de la vidéo indisponibles.");
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Capture de l’image indisponible.");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}
