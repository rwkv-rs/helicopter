import { getJson } from "../http";
import type { LeaderboardResponse } from "../dtos/api/leaderboard";
import type { ScoreScope } from "../score_scope";

export function leaderboard(
  model: string | null,
  view: string,
  scope: ScoreScope,
): Promise<LeaderboardResponse> {
  const params = new URLSearchParams({ view, scope });
  if (model) params.set("model", model);
  return getJson<LeaderboardResponse>(`/api/leaderboard?${params.toString()}`);
}
