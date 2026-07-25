import { getJson } from "./http";
import type {
  AnswerOutcome,
  AnswerSample,
  AnswerSampleGroups,
  BenchmarkScore,
  ComparisonDataSource,
  ComparisonDataset,
  ComparisonId,
  ComparisonOption,
  ComparisonScore,
  HistoryPoint,
  ModelVariant,
  ParameterGroup,
  SamplingConfig,
  ScoreArm,
  ScoreCellSelection,
} from "./comparison_types";

interface ApiComparisonOption {
  id: ComparisonId;
  label: string;
  short_label: string;
  a_label: string;
  b_label: string;
  contract: string;
}

interface ApiParameterGroup {
  id: string;
  label: string;
  a_model: ModelVariant;
  b_model: ModelVariant;
  parameter_delta_percent: number;
  comparable: boolean;
}

interface ApiComparisonCoordinate {
  comparison: ApiComparisonOption;
  parameter_group: ApiParameterGroup;
  arm: ScoreArm;
}

interface ApiEvaluation {
  evaluation_id: string;
  source_run_id: string;
  created_at: string;
  task_name: string;
  model: ModelVariant;
  benchmark: {
    label: string;
    domain: string;
    evaluation_method: string;
    score_multiplier: number;
  };
  evaluation: {
    prompt_profile: string;
    prompt_template: string;
    precision: string;
  };
  comparisons: ApiComparisonCoordinate[];
  sampling_config: Record<string, unknown>;
  primary_metric: string;
  aggregates: Record<string, number>;
  diagnostics: {
    samples: number;
    truncation_rate: number;
  };
}

interface ApiEvaluationList {
  evaluations: ApiEvaluation[];
  generated_at: string;
}

interface ApiSample {
  id: string;
  sample_index: number;
  outcome: AnswerOutcome;
  doc: Record<string, unknown>;
  metric: Record<string, unknown>;
  model_response: Record<string, unknown>;
}

interface ApiSampleGroup {
  outcome: AnswerOutcome;
  total: number;
  items: ApiSample[];
}

interface ApiSampleGroups {
  groups: Record<AnswerOutcome, ApiSampleGroup>;
}

interface Pair {
  a?: ApiEvaluation;
  b?: ApiEvaluation;
}

const OUTCOMES: AnswerOutcome[] = [
  "correct",
  "incorrect",
  "unanswered",
  "undetermined",
];

function option(value: ApiComparisonOption): ComparisonOption {
  return {
    id: value.id,
    label: value.label,
    shortLabel: value.short_label,
    aLabel: value.a_label,
    bLabel: value.b_label,
    contract: value.contract,
  };
}

function parameterGroup(value: ApiParameterGroup): ParameterGroup {
  return {
    id: value.id,
    label: value.label,
    aModel: value.a_model,
    bModel: value.b_model,
    parameterDeltaPercent: value.parameter_delta_percent,
    comparable: value.comparable,
  };
}

function numeric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function sampling(value: Record<string, unknown>): SamplingConfig {
  return {
    temperature: numeric(value.temperature),
    topP: numeric(value.top_p),
    topK: numeric(value.top_k),
    maxTokens: numeric(value.max_tokens ?? value.max_new_tokens),
    seed: numeric(value.seed),
  };
}

function pairScore(pair: Required<Pair>): ComparisonScore {
  return {
    a: pair.a.aggregates[pair.a.primary_metric] * pair.a.benchmark.score_multiplier,
    b: pair.b.aggregates[pair.b.primary_metric] * pair.b.benchmark.score_multiplier,
    aTruncationRate: pair.a.diagnostics.truncation_rate * 100,
    bTruncationRate: pair.b.diagnostics.truncation_rate * 100,
    aEvaluationId: pair.a.evaluation_id,
    bEvaluationId: pair.b.evaluation_id,
    aRunId: pair.a.source_run_id,
    bRunId: pair.b.source_run_id,
    aPromptTemplate: pair.a.evaluation.prompt_template,
    bPromptTemplate: pair.b.evaluation.prompt_template,
    aSamplingConfig: sampling(pair.a.sampling_config),
    bSamplingConfig: sampling(pair.b.sampling_config),
  };
}

function dataset(payload: ApiEvaluationList): ComparisonDataset {
  const comparisons = new Map<ComparisonId, ComparisonOption>();
  const groups = new Map<ComparisonId, Map<string, ParameterGroup>>();
  const pairs = new Map<string, Pair>();
  const evaluations = new Map<string, ApiEvaluation>();
  const coordinates = new Map<string, ApiComparisonCoordinate>();
  const history: HistoryPoint[] = [];

  for (const evaluation of payload.evaluations) {
    for (const coordinate of evaluation.comparisons) {
      const comparison = option(coordinate.comparison);
      const group = parameterGroup(coordinate.parameter_group);
      const existingComparison = comparisons.get(comparison.id);
      if (
        existingComparison &&
        JSON.stringify(existingComparison) !== JSON.stringify(comparison)
      ) {
        throw new Error(`comparison metadata conflict: ${comparison.id}`);
      }
      comparisons.set(comparison.id, comparison);
      const byGroup = groups.get(comparison.id) ?? new Map<string, ParameterGroup>();
      const existingGroup = byGroup.get(group.id);
      if (existingGroup && JSON.stringify(existingGroup) !== JSON.stringify(group)) {
        throw new Error(`parameter group metadata conflict: ${comparison.id}/${group.id}`);
      }
      byGroup.set(group.id, group);
      groups.set(comparison.id, byGroup);

      const key = [
        comparison.id,
        group.id,
        evaluation.benchmark.label,
        evaluation.primary_metric,
      ].join("\u0000");
      const pair = pairs.get(key) ?? {};
      if (!pair[coordinate.arm]) pair[coordinate.arm] = evaluation;
      pairs.set(key, pair);
      evaluations.set(key, evaluation);
      coordinates.set(key, coordinate);

      history.push({
        id: `${evaluation.evaluation_id}:${comparison.id}:${group.id}:${coordinate.arm}`,
        runLabel: evaluation.source_run_id,
        createdAt: evaluation.created_at,
        score:
          evaluation.aggregates[evaluation.primary_metric] *
          evaluation.benchmark.score_multiplier,
        parameterGroupId: group.id,
        benchmark: evaluation.benchmark.label,
        comparisonId: comparison.id,
        arm: coordinate.arm,
        model: evaluation.model.label,
        promptProfile: evaluation.evaluation.prompt_profile,
        precision: evaluation.evaluation.precision,
        samples: evaluation.diagnostics.samples,
        runId: evaluation.source_run_id,
      });
    }
  }

  const rows = new Map<string, BenchmarkScore>();
  for (const [key, pair] of pairs) {
    if (!pair.a || !pair.b) continue;
    const evaluation = evaluations.get(key)!;
    const coordinate = coordinates.get(key)!;
    const rowKey = [
      evaluation.benchmark.label,
      evaluation.primary_metric,
      evaluation.benchmark.evaluation_method,
    ].join("\u0000");
    const row =
      rows.get(rowKey) ??
      ({
        benchmark: evaluation.benchmark.label,
        samples: Math.min(pair.a.diagnostics.samples, pair.b.diagnostics.samples),
        evalMethod: evaluation.benchmark.evaluation_method,
        metric: evaluation.primary_metric,
        domain: evaluation.benchmark.domain as BenchmarkScore["domain"],
        scores: {},
      } as BenchmarkScore);
    const byParameter = row.scores[coordinate.comparison.id] ?? {};
    byParameter[coordinate.parameter_group.id] = pairScore({
      a: pair.a,
      b: pair.b,
    });
    row.scores[coordinate.comparison.id] = byParameter;
    rows.set(rowKey, row);
  }

  return {
    comparisons: [...comparisons.values()],
    parameterGroups: Object.fromEntries(
      [...groups].map(([id, values]) => [id, [...values.values()]])
    ) as ComparisonDataset["parameterGroups"],
    benchmarks: [...rows.values()],
    history,
    generatedAt: payload.generated_at,
    source: "api",
  };
}

function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function reference(doc: Record<string, unknown>): string {
  const choices = Array.isArray(doc.choices) ? doc.choices : [];
  const indices = Array.isArray(doc.gold_index) ? doc.gold_index : [doc.gold_index];
  const selected = indices
    .filter((index): index is number => Number.isInteger(index))
    .map((index) => choices[index])
    .filter((value) => value !== undefined);
  if (selected.length) return selected.map(String).join("\n");
  for (const key of ["reference", "target", "gold", "answer"]) {
    const value = doc[key];
    if (value !== undefined && value !== null) {
      return typeof value === "string" ? value : JSON.stringify(value);
    }
  }
  return "—";
}

function prompt(sample: ApiSample): string {
  const input = sample.model_response.input;
  if (typeof input === "string") return input;
  const query = sample.doc.query;
  if (typeof query === "string") return query;
  return JSON.stringify(sample.doc, null, 2);
}

function joined(values: string[], label: string): string {
  if (values.length <= 1) return values[0] ?? "";
  return values
    .map((value, index) => `--- ${label} ${index + 1} ---\n${value}`)
    .join("\n\n");
}

function generatedTokens(response: Record<string, unknown>): number {
  if (!Array.isArray(response.output_tokens)) return 0;
  return response.output_tokens.reduce(
    (total, value) => total + (Array.isArray(value) ? value.length : 0),
    0,
  );
}

function answerSample(sample: ApiSample, runId: string): AnswerSample {
  const raw = strings(sample.model_response.text);
  const processed = strings(sample.model_response.text_post_processed);
  return {
    id: sample.id,
    problemId:
      typeof sample.doc.id === "string"
        ? sample.doc.id
        : String(sample.sample_index),
    repeatId: sample.sample_index,
    groundTruth: reference(sample.doc),
    extractedAnswer: joined(processed, "extracted answer"),
    isPassed:
      sample.outcome === "correct"
        ? true
        : sample.outcome === "incorrect"
          ? false
          : null,
    sampleMetric: sample.metric,
    context: {
      assembledPrompt: prompt(sample),
      rawCompletion: joined(raw, "completion"),
      failReason: null,
      generatedTokens: generatedTokens(sample.model_response),
      latencyMs: null,
      runId,
    },
  };
}

export class ApiComparisonDataSource implements ComparisonDataSource {
  async load(): Promise<ComparisonDataset> {
    return dataset(await getJson<ApiEvaluationList>("/api/evaluations"));
  }

  async loadAnswerSamples(
    selection: ScoreCellSelection,
    limit: number,
  ): Promise<AnswerSampleGroups> {
    const payload = await getJson<ApiSampleGroups>(
      `/api/evaluations/${encodeURIComponent(selection.evaluationId)}/samples?limit=${limit}`,
    );
    return Object.fromEntries(
      OUTCOMES.map((outcome) => {
        const group = payload.groups[outcome] ?? {
          outcome,
          total: 0,
          items: [],
        };
        return [
          outcome,
          {
            outcome,
            total: group.total,
            items: group.items.map((sample) =>
              answerSample(sample, selection.runId),
            ),
          },
        ];
      }),
    ) as AnswerSampleGroups;
  }
}
