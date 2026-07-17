import type { ScoreScope } from "../../../score_scope";

export interface ScoreHistoryOptionsResponse {
  scope: ScoreScope;
  models: string[];
  benchmarks: string[];
  pairs: { model: string; dataset: string }[];
}
