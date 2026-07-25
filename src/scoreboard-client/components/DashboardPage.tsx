"use client";

import { AnswerDetailsPanel } from "./AnswerDetailsPanel";
import { ComparisonTabs } from "./ComparisonTabs";
import { ScoreMatrix } from "./ScoreMatrix";
import { useComparisonStore } from "./ComparisonProvider";

export function DashboardPage() {
  const { state } = useComparisonStore();
  if (state.status === "loading") return <div className="spinner">正在准备展示数据…</div>;
  if (state.status === "error") return <div className="error-bar">加载失败：{state.error}</div>;
  return (
    <div className="scoreboard-page">
      <ComparisonTabs />
      <ScoreMatrix />
      <AnswerDetailsPanel />
    </div>
  );
}
