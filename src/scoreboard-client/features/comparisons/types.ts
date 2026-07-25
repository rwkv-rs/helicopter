export type ComparisonId =
  | "generation"
  | "prompt_template"
  | "fake_cot"
  | "precision"
  | "architecture";

export type DomainId =
  | "regular"
  | "math"
  | "knowledge"
  | "instruction"
  | "coding"
  | "agent";

export interface ComparisonOption {
  id: ComparisonId;
  label: string;
  shortLabel: string;
  aLabel: string;
  bLabel: string;
  contract: string;
}

export interface ParameterGroup {
  id: string;
  label: string;
  aModel: string;
  bModel: string;
  parameterDeltaPercent: number;
  comparable: boolean;
}

export interface BenchmarkScore {
  benchmark: string;
  samples: number;
  evalMethod: string;
  metric: string;
  domain: Exclude<DomainId, "regular">;
  scores: Record<ComparisonId, Record<string, { a: number; b: number }>>;
}

export interface HistoryPoint {
  id: string;
  runLabel: string;
  createdAt: string;
  score: number;
  parameterGroupId: string;
  benchmark: string;
  comparisonId: ComparisonId;
  arm: "a" | "b";
  model: string;
  promptProfile: string;
  precision: string;
  samples: number;
  runId: string;
}

export interface ComparisonDataset {
  comparisons: ComparisonOption[];
  parameterGroups: Record<ComparisonId, ParameterGroup[]>;
  benchmarks: BenchmarkScore[];
  history: HistoryPoint[];
  generatedAt: string;
  source: "mock" | "api";
}

export interface ComparisonDataSource {
  load(): Promise<ComparisonDataset>;
}
