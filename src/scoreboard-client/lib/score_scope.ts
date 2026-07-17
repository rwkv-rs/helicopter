export type ScoreScope = "official" | "non_official";

export function parseScoreScope(raw: string | string[] | undefined): ScoreScope {
  const value = Array.isArray(raw) ? raw[0] : raw;
  return value === "non_official" ? "non_official" : "official";
}
