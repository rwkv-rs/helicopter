"use client";

import { useMemo } from "react";

import { ComparisonTabs } from "./ComparisonTabs";
import { useComparisonStore } from "./ComparisonProvider";

function shortDate(iso: string): string {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
  }).format(new Date(iso));
}

export function HistoryPage() {
  const { state, dispatch, selectedHistoryPoint } = useComparisonStore();
  const series = useMemo(() => {
    if (!state.data) return [];
    return (state.data.parameterGroups[state.comparisonId] ?? []).map((group) => ({
      group,
      points: state.data!.history.filter(
        (point) =>
          point.comparisonId === state.comparisonId &&
          point.parameterGroupId === group.id,
      ),
    }));
  }, [state.comparisonId, state.data]);

  if (state.status === "loading") return <div className="spinner">正在准备历史数据…</div>;
  if (!state.data) return <div className="error-bar">历史数据不可用。</div>;
  if (!state.data.comparisons.length) {
    return <div className="empty">尚无带 comparison metadata 的历史结果。</div>;
  }
  const comparison = state.data.comparisons.find(
    (item) => item.id === state.comparisonId,
  );
  if (!comparison) return <div className="empty">当前对比维度没有结果。</div>;

  return (
    <>
      <ComparisonTabs />
      <div className="history-layout">
        <div className="history-series">
          {series.map(({ group, points }) => {
            const max = Math.max(...points.map((point) => point.score), 1);
            return (
              <section className="card history-card" key={group.id}>
                <div className="history-card-head">
                  <div>
                    <div className="card-title">
                      {group.label} · {comparison.label}
                    </div>
                    <div className="muted">
                      {group.aModel.label} / {group.bModel.label} · {points.length} 条分数
                    </div>
                  </div>
                  {!group.comparable ? (
                    <span className="warning-badge">参数量不可比</span>
                  ) : null}
                </div>
                <div className="history-chart" role="img" aria-label={`${group.label} 分数历史`}>
                  {points.map((point) => (
                    <button
                      aria-label={`${point.runLabel} ${point.arm.toUpperCase()} ${point.score.toFixed(1)}%`}
                      className={`history-bar ${point.arm}${state.selectedHistoryPointId === point.id ? " selected" : ""}`}
                      key={point.id}
                      style={{ height: `${Math.max(8, (point.score / max) * 210)}px` }}
                      title={`${shortDate(point.createdAt)} · ${point.score.toFixed(1)}%`}
                      type="button"
                      onClick={() =>
                        dispatch({ type: "select-history-point", pointId: point.id })
                      }
                    >
                      <span>{point.score.toFixed(1)}</span>
                    </button>
                  ))}
                </div>
                <div className="history-axis">
                  {Array.from({ length: 8 }, (_, index) => (
                    <span key={index}>r{String(index + 1).padStart(2, "0")}</span>
                  ))}
                </div>
              </section>
            );
          })}
        </div>
        <aside className="card history-detail">
          <div className="card-title">分数来源</div>
          {selectedHistoryPoint ? (
            <>
              <div className="history-score">{selectedHistoryPoint.score.toFixed(1)}%</div>
              <div className="detail-pills">
                <span>{selectedHistoryPoint.arm.toUpperCase()}</span>
                <span>{selectedHistoryPoint.parameterGroupId.toUpperCase()}</span>
              </div>
              <dl className="detail-list">
                <dt>run_id</dt><dd>{selectedHistoryPoint.runId}</dd>
                <dt>model</dt><dd>{selectedHistoryPoint.model}</dd>
                <dt>benchmark</dt><dd>{selectedHistoryPoint.benchmark}</dd>
                <dt>created_at</dt><dd>{selectedHistoryPoint.createdAt}</dd>
                <dt>samples</dt><dd>{selectedHistoryPoint.samples}</dd>
                <dt>prompt_profile</dt><dd>{selectedHistoryPoint.promptProfile}</dd>
                <dt>precision</dt><dd>{selectedHistoryPoint.precision}</dd>
              </dl>
              <div className="stage">
                <div className="stage-label">comparison contract</div>
                <pre>{comparison.contract}</pre>
              </div>
            </>
          ) : (
            <div className="empty">点击任意柱子查看分数来源。</div>
          )}
        </aside>
      </div>
    </>
  );
}
