import { getJson } from "../http";
import type { LeaderboardResponse } from "../dtos/api/leaderboard";

export function leaderboard(model: string | null, view: string): Promise<LeaderboardResponse> {
  const params = new URLSearchParams({ view });
  if (model) params.set("model", model);
  return getJson<LeaderboardResponse>(`/api/leaderboard?${params.toString()}`);
}
