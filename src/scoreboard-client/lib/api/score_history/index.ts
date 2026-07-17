import { getJson } from "../../http";
import type { ScoreHistoryResponse } from "../../dtos/api/score_history";
import type { ScoreHistoryScope } from "../../dtos/api/score_history/options";

export function scoreHistory(
  model: string,
  benchmark: string,
  scope: ScoreHistoryScope,
): Promise<ScoreHistoryResponse> {
  return getJson<ScoreHistoryResponse>(
    `/api/score-history?model=${encodeURIComponent(model)}&benchmark=${encodeURIComponent(benchmark)}&scope=${scope}`,
  );
}
