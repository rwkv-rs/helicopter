import { getJson } from "../../http";
import type { ScoreHistoryResponse } from "../../dtos/api/score_history";
import type { ScoreScope } from "../../score_scope";

export function scoreHistory(
  model: string,
  benchmark: string,
  scope: ScoreScope,
): Promise<ScoreHistoryResponse> {
  return getJson<ScoreHistoryResponse>(
    `/api/score-history?model=${encodeURIComponent(model)}&benchmark=${encodeURIComponent(benchmark)}&scope=${scope}`,
  );
}
