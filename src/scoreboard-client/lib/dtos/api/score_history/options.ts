export type ScoreHistoryScope = "official" | "non_official";

export interface ScoreHistoryOptionsResponse {
  scope: ScoreHistoryScope;
  models: string[];
  benchmarks: string[];
  pairs: { model: string; dataset: string }[];
}
