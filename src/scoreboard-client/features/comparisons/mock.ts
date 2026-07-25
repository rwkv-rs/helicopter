import type {
  BenchmarkScore,
  ComparisonDataSource,
  ComparisonDataset,
  ComparisonId,
  ComparisonOption,
  HistoryPoint,
  ParameterGroup,
} from "./types";

const COMPARISONS: ComparisonOption[] = [
  {
    id: "generation",
    label: "前代 vs 当代",
    shortLabel: "代际",
    aLabel: "前代",
    bLabel: "当代",
    contract: "仅改变模型代际；prompt、precision、sampling 与输出边界保持一致。",
  },
  {
    id: "prompt_template",
    label: "Prompt template",
    shortLabel: "模板",
    aLabel: "✿\\nBot",
    bLabel: "\\n\\nAssistant",
    contract:
      'A: User✿{task.problem}✿\\nBot✿<think  ·  B: User: {task.problem}\\n\\nAssistant: <think',
  },
  {
    id: "fake_cot",
    label: "Fake CoT vs CoT",
    shortLabel: "推理",
    aLabel: "Fake CoT",
    bLabel: "CoT",
    contract: "A 使用 <think></think>；B 使用真实 <think> 推理，其余 contract 保持一致。",
  },
  {
    id: "precision",
    label: "fp16 vs fp32io16",
    shortLabel: "精度",
    aLabel: "fp16",
    bLabel: "fp32io16",
    contract: "同一 checkpoint 与 generation contract，仅改变 WKV precision。",
  },
  {
    id: "architecture",
    label: "Qwen3.5 vs RWKV",
    shortLabel: "架构",
    aLabel: "Qwen3.5",
    bLabel: "RWKV",
    contract: "选择最接近参数量的 Qwen3.5；参数差超过 10% 时标记为不可比较。",
  },
];

const PARAMETER_LABELS = ["1.5B", "2.9B", "7.2B", "13.3B"];
const BENCHMARKS = [
  ["AIME24", 64, "cot", "avg@32", "math"],
  ["AIME25", 30, "cot", "avg@32", "math"],
  ["MATH-500", 500, "cot", "em", "math"],
  ["GSM8K", 1319, "cot", "em", "math"],
  ["MMLU", 14042, "nocot", "acc", "knowledge"],
  ["IFEval", 541, "nocot", "strict", "instruction"],
  ["LiveCodeBench", 511, "cot", "pass@1", "coding"],
  ["SWE-bench Verified", 500, "agent", "resolved", "agent"],
] as const;

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function clamp(value: number): number {
  return Math.max(0, Math.min(100, value));
}

function parameterGroups(comparisonId: ComparisonId): ParameterGroup[] {
  return PARAMETER_LABELS.map((label, index) => {
    const architectureDelta = comparisonId === "architecture" ? [6.7, 3.4, 4.2, 12.8][index] : 0;
    return {
      id: label.toLowerCase(),
      label,
      aModel:
        comparisonId === "architecture"
          ? `Qwen3.5 ${["1.7B", "3B", "7.5B", "15B"][index]}`
          : `RWKV7 g1g ${label}`,
      bModel:
        comparisonId === "architecture"
          ? `RWKV ${label}`
          : `RWKV7 g1h ${label}`,
      parameterDeltaPercent: architectureDelta,
      comparable: architectureDelta <= 10,
    };
  });
}

function buildBenchmarks(random: () => number): BenchmarkScore[] {
  return BENCHMARKS.map(([benchmark, samples, evalMethod, metric, domain], rowIndex) => {
    const scores = Object.fromEntries(
      COMPARISONS.map((comparison, comparisonIndex) => {
        const byParameter: Record<string, { a: number; b: number }> = {};
        PARAMETER_LABELS.forEach((label, parameterIndex) => {
          const scaleGain = parameterIndex * 9.5;
          const taskBias = rowIndex * 1.8;
          const comparisonBias = comparisonIndex * 0.7;
          const a = clamp(18 + scaleGain + taskBias + comparisonBias + random() * 12);
          const effect = comparison.id === "fake_cot" ? 5.5 : 2.8;
          const b = clamp(a + effect + (random() - 0.5) * 5);
          byParameter[label.toLowerCase()] = { a, b };
        });
        return [comparison.id, byParameter];
      }),
    ) as BenchmarkScore["scores"];
    return { benchmark, samples, evalMethod, metric, domain, scores };
  });
}

function buildHistory(
  random: () => number,
  comparisons: ComparisonOption[],
  groups: Record<ComparisonId, ParameterGroup[]>,
): HistoryPoint[] {
  const points: HistoryPoint[] = [];
  comparisons.forEach((comparison, comparisonIndex) => {
    groups[comparison.id].forEach((group, groupIndex) => {
      for (let run = 0; run < 8; run += 1) {
        const base = 30 + groupIndex * 9 + comparisonIndex * 1.5 + run * 0.8;
        for (const arm of ["a", "b"] as const) {
          const score = clamp(base + (arm === "b" ? 2.8 : 0) + (random() - 0.5) * 5);
          points.push({
            id: `${comparison.id}-${group.id}-${run}-${arm}`,
            runLabel: `r${String(run + 1).padStart(2, "0")}`,
            createdAt: new Date(Date.UTC(2026, 6, 1 + run * 3, 12)).toISOString(),
            score,
            parameterGroupId: group.id,
            benchmark: run % 2 === 0 ? "MATH-500" : "AIME24",
            comparisonId: comparison.id,
            arm,
            model: arm === "a" ? group.aModel : group.bModel,
            promptProfile: comparison.id === "prompt_template" ? (arm === "a" ? "rwkv" : "assistant") : "unified",
            precision: comparison.id === "precision" ? (arm === "a" ? "fp16" : "fp32io16") : "fp32io16",
            samples: run % 2 === 0 ? 500 : 64,
            runId: `mock-${comparison.id}-${group.id}-${run}-${arm}`,
          });
        }
      }
    });
  });
  return points;
}

export class MockComparisonDataSource implements ComparisonDataSource {
  async load(): Promise<ComparisonDataset> {
    const random = seededRandom(0x7a11ce);
    const groups = Object.fromEntries(
      COMPARISONS.map((comparison) => [comparison.id, parameterGroups(comparison.id)]),
    ) as Record<ComparisonId, ParameterGroup[]>;
    return {
      comparisons: COMPARISONS,
      parameterGroups: groups,
      benchmarks: buildBenchmarks(random),
      history: buildHistory(random, COMPARISONS, groups),
      generatedAt: "2026-07-24T12:00:00.000Z",
      source: "mock",
    };
  }
}
