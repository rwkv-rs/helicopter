"use client";

import { ComparisonTabs } from "./ComparisonTabs";
import { ScoreMatrix } from "./ScoreMatrix";
import { useComparisonStore } from "./store";

export function ScoreboardPage() {
  const { state } = useComparisonStore();
  if (state.status === "loading") return <div className="spinner">正在准备展示数据…</div>;
  if (state.status === "error") return <div className="error-bar">加载失败：{state.error}</div>;
  return (
    <div className="scoreboard-page">
      <ComparisonTabs />
      <ScoreMatrix />
    </div>
  );
}
