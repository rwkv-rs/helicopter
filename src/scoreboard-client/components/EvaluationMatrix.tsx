"use client";

import { useMemo, useState } from "react";

import type { EvaluationSummary, WkvMode } from "../lib/evaluation_types";
import { useEvaluations } from "./EvaluationProvider";

const MODES: WkvMode[] = ["fp16", "fp32io16"];

function metricEntries(
  evaluation: EvaluationSummary,
): [string, number][] {
  return Object.entries(evaluation.aggregates).sort(([left], [right]) => {
    if (left === evaluation.primary_metric) return -1;
    if (right === evaluation.primary_metric) return 1;
    return left.localeCompare(right);
  });
}

function metricsTitle(evaluation: EvaluationSummary): string {
  return Object.entries(evaluation.aggregates)
    .map(([name, value]) => `${name}=${value}`)
    .join("\n");
}

function keyOf(evaluation: EvaluationSummary): string {
  return [
    evaluation.task.weight_sha256,
    evaluation.task.wkv_mode,
    evaluation.task.task_name,
  ].join("\u0000");
}

export function EvaluationMatrix() {
  const { data, select } = useEvaluations();
  const [tag, setTag] = useState("all");
  const [module, setModule] = useState("all");

  const allEvaluations = data?.evaluations ?? [];
  const latestCampaignId = useMemo(() => {
    const latest = allEvaluations.reduce<EvaluationSummary | null>(
      (current, row) => {
        if (!current) return row;
        const completedOrder = row.completed_at.localeCompare(
          current.completed_at,
        );
        if (completedOrder !== 0) return completedOrder > 0 ? row : current;
        return row.campaign_id.localeCompare(current.campaign_id) > 0
          ? row
          : current;
      },
      null,
    );
    return latest?.campaign_id;
  }, [allEvaluations]);
  const evaluations = useMemo(
    () =>
      allEvaluations.filter((row) => row.campaign_id === latestCampaignId),
    [allEvaluations, latestCampaignId],
  );
  const tags = useMemo(
    () =>
      [
        ...new Set(evaluations.flatMap((row) => row.task.upstream_tags)),
      ].sort(),
    [evaluations],
  );
  const modules = useMemo(
    () => [...new Set(evaluations.map((row) => row.task.module_family))].sort(),
    [evaluations],
  );
  const weights = useMemo(() => {
    const unique = new Map<string, string>();
    for (const row of evaluations) {
      unique.set(row.task.weight_sha256, row.task.weight_display_name);
    }
    return [...unique];
  }, [evaluations]);
  const filtered = evaluations.filter(
    (row) =>
      (tag === "all" || row.task.upstream_tags.includes(tag)) &&
      (module === "all" || row.task.module_family === module),
  );
  const byKey = new Map<string, EvaluationSummary>();
  for (const row of filtered) {
    const key = keyOf(row);
    if (!byKey.has(key)) byKey.set(key, row);
  }
  const tasks = [
    ...new Map(
      filtered.map((row) => [
        row.task.task_name,
        {
          name: row.task.task_name,
          module: row.task.module_family,
          tags: row.task.upstream_tags,
        },
      ]),
    ).values(),
  ].sort((left, right) => left.name.localeCompare(right.name));

  return (
    <section className="card">
      <header className="section-head">
        <div>
          <h2>配置的 LightEval benchmarks</h2>
          <p>显示每个 task 的原生 metric；空单元格表示该 weight/mode 结果缺失。</p>
        </div>
        <div className="filters">
          <label>
            module
            <select value={module} onChange={(event) => setModule(event.target.value)}>
              <option value="all">全部</option>
              {modules.map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
        </div>
      </header>
      <nav aria-label="官方 LightEval tags" className="tag-tabs">
        {["all", ...tags].map((value) => (
          <button
            aria-pressed={tag === value}
            className={tag === value ? "tag-tab active" : "tag-tab"}
            key={value}
            onClick={() => setTag(value)}
            type="button"
          >
            {value === "all" ? "All" : value}
          </button>
        ))}
      </nav>
      <div className="table-scroll">
        <table className="matrix">
          <thead>
            <tr>
              <th rowSpan={2}>task</th>
              <th rowSpan={2}>module</th>
              <th rowSpan={2}>tags</th>
              {weights.map(([sha, name]) => (
                <th colSpan={MODES.length} key={sha} title={sha}>
                  {name}
                </th>
              ))}
            </tr>
            <tr>
              {weights.flatMap(([sha]) =>
                MODES.map((mode) => <th key={`${sha}:${mode}`}>{mode}</th>),
              )}
            </tr>
          </thead>
          <tbody>
            {tasks.map((task) => (
              <tr key={task.name}>
                <td className="task-name">{task.name}</td>
                <td>{task.module}</td>
                <td>{task.tags.length ? task.tags.join(", ") : "—"}</td>
                {weights.flatMap(([sha]) =>
                  MODES.map((mode) => {
                    const row = byKey.get([sha, mode, task.name].join("\u0000"));
                    return (
                      <td key={`${sha}:${mode}`}>
                        {row ? (
                          <button
                            className="metric-button"
                            onClick={() => select(row)}
                            title={`${metricsTitle(row)}\nn=${row.diagnostics.samples}`}
                            type="button"
                          >
                            {metricEntries(row).map(([name, value]) => (
                              <span className="native-metric" key={name}>
                                <strong>{String(value)}</strong>
                                <small>{name}</small>
                              </span>
                            ))}
                          </button>
                        ) : (
                          <span aria-label={`${task.name} ${mode} 结果缺失`} className="missing">
                            缺失
                          </span>
                        )}
                      </td>
                    );
                  }),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!tasks.length ? <p className="empty">当前筛选条件没有结果。</p> : null}
    </section>
  );
}
