"use client";

import { useEvaluations } from "./EvaluationProvider";
import { EvaluationDetails } from "./EvaluationDetails";

function metricEntries(
  primaryMetric: string,
  aggregates: Record<string, number>,
): [string, number][] {
  return Object.entries(aggregates).sort(([left], [right]) => {
    if (left === primaryMetric) return -1;
    if (right === primaryMetric) return 1;
    return left.localeCompare(right);
  });
}

export function HistoryPage() {
  const { status, data, error, select } = useEvaluations();
  if (status === "loading") return <p>正在读取历史记录…</p>;
  if (status === "error") return <p className="error-bar">加载失败：{error}</p>;
  if (!data?.evaluations.length) return <p className="empty">尚无已完成的历史记录。</p>;
  const rows = [...data.evaluations].sort((left, right) =>
    right.completed_at.localeCompare(left.completed_at),
  );
  return (
    <div className="stack">
      <section className="card">
        <h2>评估历史</h2>
        <div className="table-scroll">
          <table className="history">
            <thead>
              <tr>
                <th>完成时间</th>
                <th>weight</th>
                <th>WKV</th>
                <th>task</th>
                <th>module</th>
                <th>tags</th>
                <th>native metric</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.evaluation_id}>
                  <td>{new Date(row.completed_at).toLocaleString("zh-CN")}</td>
                  <td>{row.task.weight_display_name}</td>
                  <td>{row.task.wkv_mode}</td>
                  <td>
                    <button
                      className="link-button"
                      onClick={() => select(row)}
                      type="button"
                    >
                      {row.task.task_name}
                    </button>
                  </td>
                  <td>{row.task.module_family}</td>
                  <td>
                    {row.task.upstream_tags.length
                      ? row.task.upstream_tags.join(", ")
                      : "—"}
                  </td>
                  <td>
                    <div className="native-metrics">
                      {metricEntries(
                        row.primary_metric,
                        row.aggregates,
                      ).map(([name, value]) => (
                        <span className="native-metric" key={name}>
                          <strong>{String(value)}</strong>
                          <small>{name}</small>
                        </span>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <EvaluationDetails />
    </div>
  );
}
