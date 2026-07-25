import type {
  AnswerOutcome,
  AnswerSample,
  AnswerSampleGroups,
  BenchmarkScore,
  ComparisonDataSource,
  ComparisonDataset,
  ComparisonId,
  ComparisonOption,
  HistoryPoint,
  ParameterGroup,
  ScoreCellSelection,
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

function stringSeed(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function clamp(value: number): number {
  return Math.max(0, Math.min(100, value));
}

function sampleProblem(benchmark: string, sampleIndex: number): string {
  const ordinal = sampleIndex + 1;
  if (benchmark.startsWith("AIME")) {
    return `设正整数 x、y 满足 x² + y² = ${125 + ordinal}，求题目要求的余数。`;
  }
  if (benchmark === "MATH-500") {
    return `求方程 3x² - ${ordinal + 2}x - 2 = 0 的全部实数解，并给出化简过程。`;
  }
  if (benchmark === "GSM8K") {
    return `一家商店上午售出 ${18 + ordinal} 件商品，下午比上午多售出 40%，全天共售出多少件？`;
  }
  if (benchmark === "MMLU") {
    return `关于样本 ${ordinal} 所涉及的基础概念，下列四个选项中哪一项最准确？`;
  }
  if (benchmark === "IFEval") {
    return `用恰好三句话解释评测样本 ${ordinal}，每句话以同一个动词开头。`;
  }
  if (benchmark === "LiveCodeBench") {
    return `实现 solve()：读取长度为 n 的数组，返回第 ${ordinal} 个测试条件下的最长递增子序列长度。`;
  }
  return `分析 ${benchmark} 的第 ${ordinal} 个任务，并给出可验证的最终结论。`;
}

function referenceAnswer(benchmark: string, sampleIndex: number): string {
  if (benchmark === "MMLU") return ["A", "B", "C", "D"][sampleIndex % 4];
  if (benchmark === "IFEval") return "满足三句话、相同动词开头及内容约束。";
  if (benchmark === "LiveCodeBench") return "通过动态规划或 patience sorting 得到正确长度。";
  return String((sampleIndex * 17 + benchmark.length * 11) % 997);
}

function buildAnswerSample(
  selection: ScoreCellSelection,
  outcome: AnswerOutcome,
  index: number,
  random: () => number,
): AnswerSample {
  const problem = sampleProblem(selection.benchmark, index);
  const reference = referenceAnswer(selection.benchmark, index);
  const incorrectAnswer = String((Number.parseInt(reference, 10) || index + 7) + 1);
  const unansweredReasons = [
    "empty_completion",
    "max_tokens_before_final_answer",
    "generation_timeout",
  ];
  const extractedAnswer =
    outcome === "correct"
      ? reference
      : outcome === "incorrect"
        ? incorrectAnswer
        : "";
  const rawCompletion =
    outcome === "correct"
      ? `<think>已完成逐步推理并校验结果。</think>\n${reference}`
      : outcome === "incorrect"
        ? `<think>推理中采用了错误假设，未能通过校验。</think>\n${incorrectAnswer}`
        : index % 2 === 0
          ? "<think>推理尚未完成，生成在最终答案前停止。"
          : "";
  const failReason =
    outcome === "correct"
      ? null
      : outcome === "incorrect"
        ? "answer_mismatch"
        : unansweredReasons[index % unansweredReasons.length];
  return {
    id: `${selection.comparisonId}-${selection.parameterGroupId}-${selection.arm}-${outcome}-${index}`,
    problemId: `${selection.benchmark.toLowerCase().replaceAll(" ", "-")}-${String(index + 1).padStart(4, "0")}`,
    repeatId: index % 4,
    groundTruth: reference,
    extractedAnswer,
    isPassed:
      outcome === "correct" ? true : outcome === "incorrect" ? false : null,
    context: {
      problem,
      assembledPrompt: `User: ${problem}\n\nAssistant: <think>`,
      rawCompletion,
      failReason,
      generatedTokens:
        outcome === "unanswered"
          ? Math.round(4 + random() * 28)
          : Math.round(80 + random() * 420),
      latencyMs: Math.round(450 + random() * 3800),
      runId: `mock-${selection.comparisonId}-${selection.parameterGroupId}-${selection.arm}`,
      model: selection.model,
      metric: selection.metric,
    },
  };
}

function buildAnswerSampleGroups(
  selection: ScoreCellSelection,
  limit: number,
): AnswerSampleGroups {
  const outcomes: AnswerOutcome[] = ["correct", "incorrect", "unanswered"];
  return Object.fromEntries(
    outcomes.map((outcome, outcomeIndex) => {
      const random = seededRandom(
        stringSeed(
          `${selection.benchmark}:${selection.comparisonId}:${selection.parameterGroupId}:${selection.arm}:${outcome}`,
        ),
      );
      const total = 18 + outcomeIndex * 7 + Math.floor(random() * 23);
      const items = Array.from({ length: Math.min(limit, total) }, (_, index) =>
        buildAnswerSample(selection, outcome, index, random),
      );
      return [outcome, { outcome, total, items }];
    }),
  ) as AnswerSampleGroups;
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

  async loadAnswerSamples(
    selection: ScoreCellSelection,
    limit: number,
  ): Promise<AnswerSampleGroups> {
    return buildAnswerSampleGroups(selection, limit);
  }
}
