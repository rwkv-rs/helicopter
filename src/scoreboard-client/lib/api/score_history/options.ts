import { getJson } from "../../http";
import type {
  ScoreHistoryOptionsResponse,
  ScoreHistoryScope,
} from "../../dtos/api/score_history/options";

export function scoreHistoryOptions(scope: ScoreHistoryScope): Promise<ScoreHistoryOptionsResponse> {
  return getJson<ScoreHistoryOptionsResponse>(
    `/api/score-history/options?scope=${scope}`,
  );
}
