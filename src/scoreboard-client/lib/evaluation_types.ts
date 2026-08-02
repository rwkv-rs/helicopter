export type WkvMode = "fp16" | "fp32io16";
export type PromptTemplate = "bot" | "assistant" | "function_calling" | "none";
export type AnswerOutcome =
  | "correct"
  | "incorrect"
  | "unanswered"
  | "undetermined";

export interface TaskIdentity {
  identity: string;
  weight_sha256: string;
  weight_display_name: string;
  wkv_mode: WkvMode;
  selector: string;
  task_name: string;
  task_version: string;
  module_family: string;
  module: string;
  dataset: string;
  subset: string;
  evaluation_splits: string[];
  languages: string[];
  upstream_tags: string[];
}

export interface ModelExecution {
  weight_sha256: string;
  weight_display_name: string;
  wkv_mode: WkvMode;
  prompt_template: PromptTemplate;
  gemm_policy: string;
  gpu: string;
  max_num_seqs: number;
  max_num_batched_tokens: number;
  dependency_versions: Record<string, string>;
  evaluator?: "lighteval" | "lm-eval";
}

export interface Diagnostics {
  samples: number;
  completions: number;
  truncated: number;
  non_truncated: number;
  truncation_rate: number;
  turn_boundary_violations: number;
  turn_boundary_violation_rate: number;
}

export interface EvaluationSummary {
  evaluation_id: string;
  campaign_id: string;
  task_identity: string;
  created_at: string;
  completed_at: string;
  task: TaskIdentity;
  artifact: {
    lighteval_version?: string | null;
    evaluator?: { name: "lighteval" | "lm-eval"; version: string } | null;
    results_path: string;
    details_paths: string[];
  };
  task_config: Record<string, unknown>;
  model: ModelExecution;
  sampling_config: Record<string, unknown>;
  primary_metric: string;
  aggregates: Record<string, number>;
  diagnostics: Diagnostics;
  provenance: {
    config_digest: string;
    registry_digest: string;
    eval_contract_digest: string;
    lighteval_version?: string | null;
    evaluator?: { name: "lighteval" | "lm-eval"; version: string } | null;
    configured_selectors: string[];
    resolved_selectors: string[];
    skipped_selectors: string[];
    publisher_principal: string;
  };
}

export interface EvaluationDataset {
  evaluations: EvaluationSummary[];
  generated_at: string;
  total: number;
  offset: number;
  limit: number;
  next_offset: number | null;
}

export interface SampleDetail {
  id: string;
  sample_index: number;
  document_index: number;
  outcome: AnswerOutcome;
  doc: Record<string, unknown>;
  metric: Record<string, unknown>;
  model_response: Record<string, unknown>;
}

export interface SamplePage {
  evaluation_id: string;
  primary_metric: string;
  total: number;
  offset: number;
  limit: number;
  next_offset: number | null;
  items: SampleDetail[];
}

export interface EvaluationDataSource {
  load(): Promise<EvaluationDataset>;
  loadSamples(
    evaluationId: string,
    offset: number,
    limit: number,
    outcome?: AnswerOutcome,
  ): Promise<SamplePage>;
}
