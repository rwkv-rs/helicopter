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
  architecture: string;
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
  aEvaluationId?: string;
  bEvaluationId?: string;
  aRunId?: string;
  bRunId?: string;
  aPromptTemplate?: string;
  bPromptTemplate?: string;
  aSamplingConfig?: SamplingConfig;
  bSamplingConfig?: SamplingConfig;
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
export type AnswerOutcome =
  | "correct"
  | "incorrect"
  | "unanswered"
  | "undetermined";

export interface SamplingConfig {
  temperature: number | null;
  topP: number | null;
  topK: number | null;
  maxTokens: number | null;
  seed: number | null;
}

export interface ScoreCellSelection {
  evaluationId: string;
  runId: string;
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
  samplingConfig: SamplingConfig;
}

export interface AnswerSample {
  id: string;
  problemId: string;
  repeatId: number;
  groundTruth: string;
  extractedAnswer: string;
  isPassed: boolean | null;
  sampleMetric: Record<string, unknown>;
  context: {
    assembledPrompt: string;
    rawCompletion: string;
    failReason: string | null;
    generatedTokens: number;
    latencyMs: number | null;
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
