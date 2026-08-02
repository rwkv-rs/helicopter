"use client";

import { EvaluationDetails } from "./EvaluationDetails";
import { EvaluationMatrix } from "./EvaluationMatrix";
import { useEvaluations } from "./EvaluationProvider";

export function DashboardPage() {
  const { status, data, error } = useEvaluations();
  if (status === "loading") return <p>正在读取完整 campaign…</p>;
  if (status === "error") return <p className="error-bar">加载失败：{error}</p>;
  if (!data?.evaluations.length) {
    return <p className="empty">尚无已完成的 LightEval campaign。</p>;
  }
  return (
    <div className="stack">
      <EvaluationMatrix />
      <EvaluationDetails />
    </div>
  );
}
