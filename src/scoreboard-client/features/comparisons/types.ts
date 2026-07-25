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

export interface ModelVariant {
  label: string;
  architecture: "RWKV" | "QWEN";
  generation: string;
  parameters: string;
}

export interface ParameterGroup {
  id: string;
  label: string;
  aModel: ModelVariant;
  bModel: ModelVariant;
  parameterDeltaPercent: number;
  comparable: boolean;
}

export interface ComparisonScore {
  a: number;
  b: number;
  aTruncationRate: number;
  bTruncationRate: number;
}

export interface BenchmarkScore {
  benchmark: string;
  samples: number;
  evalMethod: string;
  metric: string;
  domain: Exclude<DomainId, "regular">;
  scores: Record<ComparisonId, Record<string, ComparisonScore>>;
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

export type ScoreArm = "a" | "b";
export type AnswerOutcome = "correct" | "incorrect" | "unanswered";

export interface ScoreCellSelection {
  comparisonId: ComparisonId;
  parameterGroupId: string;
  benchmark: string;
  metric: string;
  samples: number;
  arm: ScoreArm;
  architecture: ModelVariant["architecture"];
  generation: string;
  parameterCount: string;
  score: number;
  truncationRate: number;
  promptTemplate: string;
  samplingConfig: {
    temperature: number;
    topP: number;
    topK: number;
    maxTokens: number;
    seed: number;
  };
}

export interface AnswerSample {
  id: string;
  problemId: string;
  repeatId: number;
  groundTruth: string;
  extractedAnswer: string;
  isPassed: boolean | null;
  context: {
    assembledPrompt: string;
    rawCompletion: string;
    failReason: string | null;
    generatedTokens: number;
    latencyMs: number;
    runId: string;
  };
}

export interface AnswerSampleGroup {
  outcome: AnswerOutcome;
  total: number;
  items: AnswerSample[];
}

export type AnswerSampleGroups = Record<AnswerOutcome, AnswerSampleGroup>;

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
  loadAnswerSamples(
    selection: ScoreCellSelection,
    limit: number,
  ): Promise<AnswerSampleGroups>;
}
