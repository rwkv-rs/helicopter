export interface ScoreHistoryOptionsResponse {
  models: string[];
  benchmarks: string[];
  pairs: { model: string; dataset: string }[];
}
