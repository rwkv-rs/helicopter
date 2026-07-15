export interface ScoreHistoryPoint {
  score_id: number;
  task_id: number;
  cot_mode: string;
  evaluator: string | null;
  board: "normal" | "naive";
  percent: number | null;
  metric: string | null;
  created_at: string | null;
  sampling_summary: string;
  model: string | null;
  benchmark: string | null;
}

export interface ScoreHistoryGroup {
  cot_mode: string;
  points: ScoreHistoryPoint[];
}

export interface ScoreHistoryResponse {
  model: string;
  benchmark: string;
  total: number;
  groups: ScoreHistoryGroup[];
}
