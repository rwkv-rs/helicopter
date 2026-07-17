import { getJson } from "../../http";
import type { ScoreHistoryOptionsResponse } from "../../dtos/api/score_history/options";
import type { ScoreScope } from "../../score_scope";

export function scoreHistoryOptions(scope: ScoreScope): Promise<ScoreHistoryOptionsResponse> {
  return getJson<ScoreHistoryOptionsResponse>(
    `/api/score-history/options?scope=${scope}`,
  );
}
