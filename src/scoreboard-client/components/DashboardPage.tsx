"use client";

import { AnswerDetailsPanel } from "./AnswerDetailsPanel";
import { ComparisonTabs } from "./ComparisonTabs";
import { ScoreMatrix } from "./ScoreMatrix";
import { useComparisonStore } from "./ComparisonProvider";

export function DashboardPage() {
  const { state } = useComparisonStore();
  if (state.status === "loading") return <div className="spinner">正在准备展示数据…</div>;
  if (state.status === "error") return <div className="error-bar">加载失败：{state.error}</div>;
  if (!state.data?.comparisons.length) {
    return <div className="empty">尚无带 comparison metadata 的 LightEval 结果。</div>;
  }
  return (
    <div className="scoreboard-page">
      <ComparisonTabs />
      <ScoreMatrix />
      <AnswerDetailsPanel />
    </div>
  );
}
