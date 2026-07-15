export interface CellMeta {
  cell_id: string;
  task_id: number | null;
  benchmark_name: string;
  eval_method: string;
  k_metric: string;
  column_label: string;
  model: string | null;
  tooltip: string | null;
  clickable: boolean;
}

export interface DetailCell {
  percent: number | null;
  meta: CellMeta | null;
}

export interface DeltaCell {
  prev: number | null;
  latest: number | null;
  delta: number | null;
  prev_meta: CellMeta | null;
  latest_meta: CellMeta | null;
}

export interface ParamColumn {
  param: string;
  param_label: string;
  latest_model: string;
  latest_label: string;
  prev_model: string | null;
  prev_label: string | null;
}

export interface LeaderboardRow {
  benchmark_name: string;
  num_samples: number | null;
  eval_method: string;
  k_metric: string;
  cells: (DetailCell | DeltaCell)[];
}

export interface DomainTable {
  key: string;
  title: string;
  label: string;
  param_columns: ParamColumn[];
  rows: LeaderboardRow[];
}

export interface NaiveBoard {
  key: string;
  title: string;
  label: string;
  is_delta: boolean;
  param_columns: ParamColumn[];
  rows: LeaderboardRow[];
}

export interface OverviewRow {
  domain_key: string;
  domain_title: string;
  cells: ({ percent: number | null } | DeltaCell)[];
}

export interface KnowledgeChart {
  type: "knowledge_bar";
  subjects: string[];
  models: string[];
  data: { model: string; subject: string; score: number }[];
}

export interface AimeChart {
  type: "aime_line";
  ks: number[];
  series: { name: string; points: { k: number; acc: number }[] }[];
}

export interface InstructionChart {
  type: "instruction_bar";
  domains: string[];
  models: string[];
  data: { domain: string; model: string; score: number }[];
}

export interface CodingChart {
  type: "coding_bar";
  datasets: string[];
  models: string[];
  data: { dataset: string; model: string; score: number; metric: string }[];
}

export interface AgentChart {
  type: "agent_bar";
  datasets: string[];
  models: string[];
  data: { dataset: string; model: string; score: number; metric: string }[];
}

export interface ChartPayload {
  knowledge: KnowledgeChart | null;
  math: AimeChart | null;
  instruction_following: InstructionChart | null;
  coding: CodingChart | null;
  agent: AgentChart | null;
}

export interface LeaderboardResponse {
  view: string;
  view_label: string;
  is_delta: boolean;
  is_field_avg: boolean;
  param_columns: ParamColumn[];
  interaction_meta: Record<string, CellMeta>;
  domains: DomainTable[];
  naive_board?: NaiveBoard;
  overview?: OverviewRow[] | null;
  selection: {
    dropdown_value: string;
    selected_label: string;
    auto_selected: boolean;
    model_sequence: string[];
    skipped_small_params: number;
    auto_label: string;
  };
  charts: ChartPayload;
  errors: string[];
}
