/** Une erreur HTML de Next.js ne doit pas être confondue avec un backend IA arrêté. */
export async function readApiJson<T>(response: Response, route: string): Promise<T> {
  const raw = await response.text();
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    throw new Error(`L’interface renvoie une réponse non JSON sur ${route} (HTTP ${response.status}). Vérifiez le terminal Next.js et redémarrez l’interface.`);
  }
  if (data === null || typeof data !== "object" || Array.isArray(data)) {
    throw new Error(`Réponse invalide de l’interface sur ${route} (HTTP ${response.status}).`);
  }
  return data as T;
}
