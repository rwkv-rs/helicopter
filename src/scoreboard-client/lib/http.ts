const SERVER_API_BASE = process.env.SCOREBOARD_API_BASE_URL || "http://127.0.0.1:7860";
const BROWSER_API_BASE = process.env.NEXT_PUBLIC_SCOREBOARD_API_BASE_URL || "";

function apiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  const normalized = path.startsWith("/") ? path : `/${path}`;
  if (typeof window === "undefined") return `${SERVER_API_BASE}${normalized}`;
  return BROWSER_API_BASE ? `${BROWSER_API_BASE}${normalized}` : normalized;
}

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path), { headers: { Accept: "application/json" }, cache: "no-store" });
  if (!res.ok) {
    throw new Error(`${res.status}: ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}
