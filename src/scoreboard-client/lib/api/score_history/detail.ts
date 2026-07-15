import { getJson } from "../../http";
import type { ScoreHistoryDetailResponse } from "../../dtos/api/score_history/detail";

export function scoreHistoryDetail(taskId: number): Promise<ScoreHistoryDetailResponse> {
  return getJson<ScoreHistoryDetailResponse>(`/api/score-history/detail?task_id=${taskId}`);
}
