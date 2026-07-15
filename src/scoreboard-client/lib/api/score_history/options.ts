import { getJson } from "../../http";
import type { ScoreHistoryOptionsResponse } from "../../dtos/api/score_history/options";

export function scoreHistoryOptions(): Promise<ScoreHistoryOptionsResponse> {
  return getJson<ScoreHistoryOptionsResponse>("/api/score-history/options");
}
