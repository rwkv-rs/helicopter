"use client";

import { useMemo } from "react";

import { useComparisonStore } from "./store";
import type {
  BenchmarkScore,
  ComparisonId,
  ComparisonOption,
  ComparisonScore,
  DomainId,
  ParameterGroup,
  ScoreArm,
  ScoreCellSelection,
} from "./types";

const DOMAINS: { id: DomainId; label: string }[] = [
  { id: "regular", label: "常规评估" },
  { id: "math", label: "数学" },
  { id: "knowledge", label: "知识" },
  { id: "instruction", label: "指令遵循" },
  { id: "coding", label: "编程" },
  { id: "agent", label: "Agent" },
];
const REGULAR_BENCHMARKS = new Set([
  "AIME24",
  "AIME25",
  "MATH-500",
  "GSM8K",
  "MMLU",
  "IFEval",
]);

function percent(value: number): string {
  return `${value.toFixed(1)}%`;
}

const DEFAULT_SAMPLING_CONFIG = {
  temperature: 0.6,
  topP: 0.95,
  topK: 40,
  maxTokens: 32768,
  seed: 42,
} as const;

function promptTemplateFor(comparisonId: ComparisonId, arm: ScoreArm): string {
  if (comparisonId === "prompt_template") {
    return arm === "a"
      ? "User✿{task.problem}✿\\nBot✿<think"
      : "User: {task.problem}\\n\\nAssistant: <think";
  }
  if (comparisonId === "fake_cot" && arm === "a") {
    return "User: {task.problem}\\n\\nAssistant: <think></think>";
  }
  if (comparisonId === "fake_cot") {
    return "User: {task.problem}\\n\\nAssistant: <think";
  }
  return "User✿{task.problem}✿\\nBot✿<think";
}

function scoreSelection(
  comparison: ComparisonOption,
  group: ParameterGroup,
  row: BenchmarkScore,
  score: ComparisonScore,
  arm: ScoreArm,
): ScoreCellSelection {
  const model = arm === "a" ? group.aModel : group.bModel;
  return {
    comparisonId: comparison.id,
    parameterGroupId: group.id,
    benchmark: row.benchmark,
    metric: row.metric,
    samples: row.samples,
    arm,
    architecture: model.architecture,
    generation: model.generation,
    parameterCount: model.parameters,
    score: score[arm],
    truncationRate:
      arm === "a" ? score.aTruncationRate : score.bTruncationRate,
    promptTemplate: promptTemplateFor(comparison.id, arm),
    samplingConfig: { ...DEFAULT_SAMPLING_CONFIG },
  };
}

function ArmHeadingLabel({ label }: { label: string }) {
  if (label === "fp32io16") {
    return (
      <>
        fp32
        <br />
        io16
      </>
    );
  }
  return label;
}

export function ScoreMatrix() {
  const { state, dispatch } = useComparisonStore();
  const rows = useMemo(() => {
    if (!state.data) return [];
    if (state.domainId === "regular") {
      return state.data.benchmarks.filter((row) => REGULAR_BENCHMARKS.has(row.benchmark));
    }
    return state.data.benchmarks.filter((row) => row.domain === state.domainId);
  }, [state.data, state.domainId]);

  if (!state.data) return null;
  const comparison = state.data.comparisons.find(
    (item) => item.id === state.comparisonId,
  )!;
  const parameterGroups = state.data.parameterGroups[state.comparisonId];

  return (
    <section className="card matrix-card">
      <nav className="tabs domain-tabs" aria-label="评测领域">
        {DOMAINS.map((domain) => (
          <button
            className={`tab${state.domainId === domain.id ? " active" : ""}`}
            key={domain.id}
            type="button"
            onClick={() => dispatch({ type: "select-domain", domainId: domain.id })}
          >
            {domain.label}
          </button>
        ))}
      </nav>
      <div className="matrix-heading">
        <div>
          <div className="card-title">
            {DOMAINS.find((domain) => domain.id === state.domainId)?.label} ·{" "}
            {comparison.label}
          </div>
          <div className="comparison-contract">
            <strong>{comparison.aLabel}</strong>
            <span className="contract-arrow">→</span>
            <strong>{comparison.bLabel}</strong>
            <span className="contract-divider">·</span>
            {comparison.contract}
          </div>
        </div>
        <span className="mock-badge">临时展示数据</span>
      </div>
      <div className="comparison-matrix-wrap">
        <table className="comparison-matrix">
          <colgroup>
            <col className="benchmark-col" />
            <col className="samples-col" />
            <col className="metric-col" />
            {parameterGroups.flatMap((group) => [
              <col className="score-col" key={`${group.id}-a-col`} />,
              <col className="score-col" key={`${group.id}-b-col`} />,
              <col className="delta-col" key={`${group.id}-delta-col`} />,
            ])}
          </colgroup>
          <thead>
            <tr>
              <th className="axis benchmark-axis" rowSpan={2}>benchmark</th>
              <th className="axis samples-axis" rowSpan={2}>n_<wbr />samples</th>
              <th className="axis metric-axis" rowSpan={2}>k_<wbr />metric</th>
              {parameterGroups.map((group) => (
                <th className="parameter-heading" colSpan={3} key={group.id}>
                  <span>{group.label}</span>
                  {!group.comparable ? <em>参数差 {group.parameterDeltaPercent}%</em> : null}
                </th>
              ))}
            </tr>
            <tr>
              {parameterGroups.flatMap((group) => [
                <th className="arm-heading" key={`${group.id}-a`} title={group.aModel.label}>
                  <ArmHeadingLabel label={comparison.aLabel} />
                </th>,
                <th className="arm-heading" key={`${group.id}-b`} title={group.bModel.label}>
                  <ArmHeadingLabel label={comparison.bLabel} />
                </th>,
                <th className="delta-heading" key={`${group.id}-delta`}>delta</th>,
              ])}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.benchmark}>
                <td className="benchmark-cell">{row.benchmark}</td>
                <td className="dim">{row.samples}</td>
                <td className="dim">{row.metric}</td>
                {parameterGroups.flatMap((group) => {
                  const score = row.scores[state.comparisonId][group.id];
                  if (!group.comparable || !score) {
                    return [
                      <td className="unavailable" key={`${group.id}-a`}>N/A</td>,
                      <td className="unavailable" key={`${group.id}-b`}>N/A</td>,
                      <td className="unavailable delta-cell" key={`${group.id}-d`}>—</td>,
                    ];
                  }
                  const delta = score.b - score.a;
                  return [
                    <td className="score-value score-a" key={`${group.id}-a`}>
                      <button
                        aria-label={`${row.benchmark} ${group.label} ${comparison.aLabel} ${percent(score.a)} 作答详情`}
                        className="score-detail-trigger"
                        onClick={() =>
                          dispatch({
                            type: "select-score-cell",
                            selection: scoreSelection(
                              comparison,
                              group,
                              row,
                              score,
                              "a",
                            ),
                          })
                        }
                        type="button"
                      >
                        {percent(score.a)}
                      </button>
                    </td>,
                    <td className="score-value score-b" key={`${group.id}-b`}>
                      <button
                        aria-label={`${row.benchmark} ${group.label} ${comparison.bLabel} ${percent(score.b)} 作答详情`}
                        className="score-detail-trigger"
                        onClick={() =>
                          dispatch({
                            type: "select-score-cell",
                            selection: scoreSelection(
                              comparison,
                              group,
                              row,
                              score,
                              "b",
                            ),
                          })
                        }
                        type="button"
                      >
                        {percent(score.b)}
                      </button>
                    </td>,
                    <td
                      className={`delta-cell ${delta > 0.05 ? "up" : delta < -0.05 ? "down" : "flat"}`}
                      key={`${group.id}-d`}
                    >
                      {delta > 0 ? "+" : ""}{percent(delta)}
                    </td>,
                  ];
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
