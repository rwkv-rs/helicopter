export interface StopTokenRow {
  id: number;
  token: string;
}

export interface StageSampling {
  temperature: number | null;
  top_k: number | null;
  top_p: number | null;
  max_tokens: number | null;
  stop_tokens: StopTokenRow[];
  penalties: {
    presence_penalty: number | null;
    repetition_penalty: number | null;
    penalty_decay: number | null;
  };
}

export interface ScoreHistoryDetailResponse {
  found: boolean;
  task_id: number;
  model: string | null;
  benchmark: string | null;
  cot_mode: string | null;
  evaluator: string | null;
  board: string;
  visibility: "official" | "non_official";
  eligibility: "official" | "sanity" | "proxy" | "temporary";
  comparable: boolean | null;
  dirty: boolean | null;
  run_id: string | null;
  metric: string | null;
  percent: number | null;
  metrics: Record<string, unknown>;
  accounting: Record<string, unknown>;
  performance: Record<string, unknown>;
  generated_samples: number;
  truncated_samples: number;
  truncation_rate: number;
  sampling: {
    stages: Record<string, StageSampling>;
    effective_sample_count: number | null;
    avg_k: unknown;
    pass_ks: unknown;
    n_shot: unknown;
    sample_limit: unknown;
    prompt_profile: string | null;
  };
  stages: { prompt: string; completion: string; stop_reason: unknown }[];
}
