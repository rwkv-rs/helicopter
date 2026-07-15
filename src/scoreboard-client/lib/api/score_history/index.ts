import { getJson } from "../../http";
import type { ScoreHistoryResponse } from "../../dtos/api/score_history";

export function scoreHistory(model: string, benchmark: string): Promise<ScoreHistoryResponse> {
  return getJson<ScoreHistoryResponse>(
    `/api/score-history?model=${encodeURIComponent(model)}&benchmark=${encodeURIComponent(benchmark)}`,
  );
}
