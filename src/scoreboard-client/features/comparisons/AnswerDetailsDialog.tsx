"use client";

import { useEffect, useState } from "react";

import { useComparisonStore } from "./store";
import type {
  AnswerOutcome,
  AnswerSampleGroups,
  ScoreCellSelection,
} from "./types";

const OUTCOMES: { id: AnswerOutcome; label: string }[] = [
  { id: "correct", label: "正确作答" },
  { id: "incorrect", label: "错误作答" },
  { id: "unanswered", label: "未能作答" },
];

function SelectionSummary({ selection }: { selection: ScoreCellSelection }) {
  return (
    <div className="answer-selection-summary">
      <strong>{selection.benchmark}</strong>
      <span>{selection.parameterLabel}</span>
      <span>{selection.armLabel}</span>
      <span>{selection.model}</span>
      <b>{selection.score.toFixed(1)}%</b>
    </div>
  );
}

export function AnswerDetailsDialog() {
  const { state, dispatch, loadAnswerSamples } = useComparisonStore();
  const selection = state.selectedScoreCell;
  const [activeOutcome, setActiveOutcome] = useState<AnswerOutcome>("correct");
  const [data, setData] = useState<AnswerSampleGroups | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selection) return;
    let cancelled = false;
    setActiveOutcome("correct");
    setData(null);
    setError(null);
    setLoading(true);
    loadAnswerSamples(selection, 10)
      .then((payload) => {
        if (!cancelled) setData(payload);
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [loadAnswerSamples, selection]);

  useEffect(() => {
    if (!selection) return;
    const previousOverflow = document.body.style.overflow;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        dispatch({ type: "select-score-cell", selection: null });
      }
    };
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [dispatch, selection]);

  if (!selection) return null;
  const close = () => dispatch({ type: "select-score-cell", selection: null });
  const activeGroup = data?.[activeOutcome] ?? null;

  return (
    <div className="modal-backdrop answer-detail-backdrop" onClick={close}>
      <section
        aria-label={`${selection.benchmark} 作答详情`}
        aria-modal="true"
        className="modal answer-detail-modal"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
      >
        <header className="modal-head answer-detail-head">
          <div>
            <div className="card-title">作答详情</div>
            <SelectionSummary selection={selection} />
          </div>
          <button className="btn answer-close" type="button" onClick={close}>
            关闭
          </button>
        </header>

        <nav className="answer-tabs" aria-label="作答结果" role="tablist">
          {OUTCOMES.map((outcome) => {
            const group = data?.[outcome.id];
            return (
              <button
                aria-selected={activeOutcome === outcome.id}
                className={`answer-tab ${outcome.id}${activeOutcome === outcome.id ? " active" : ""}`}
                key={outcome.id}
                onClick={() => setActiveOutcome(outcome.id)}
                role="tab"
                type="button"
              >
                <span>{outcome.label}</span>
                <b>{group ? group.items.length : "—"}</b>
              </button>
            );
          })}
        </nav>

        <div className="answer-detail-content">
          {loading ? <div className="spinner">正在抽取作答样本…</div> : null}
          {error ? <div className="error-bar">加载失败：{error}</div> : null}
          {!loading && !error && activeGroup ? (
            <>
              <div className="answer-sample-note">
                从该结果类别的 {activeGroup.total} 条记录中随机抽取{" "}
                <strong>{activeGroup.items.length}</strong> 条
              </div>
              <div className="answer-sample-list">
                {activeGroup.items.map((sample) => (
                  <article
                    className={`answer-sample-card ${activeOutcome}`}
                    key={sample.id}
                  >
                    <header>
                      <strong>sample #{sample.sampleIndex + 1}</strong>
                      <span>{sample.generatedTokens} tokens</span>
                      <span>{sample.latencyMs} ms</span>
                      {sample.failReason ? (
                        <code>{sample.failReason}</code>
                      ) : (
                        <code>passed</code>
                      )}
                    </header>
                    <div className="answer-sample-grid">
                      <section>
                        <h3>题目</h3>
                        <p>{sample.problem}</p>
                      </section>
                      <section>
                        <h3>模型作答</h3>
                        <p className={!sample.answer ? "empty-answer" : ""}>
                          {sample.answer || "无有效输出"}
                        </p>
                      </section>
                      <section>
                        <h3>参考答案</h3>
                        <p>{sample.referenceAnswer}</p>
                      </section>
                    </div>
                    <footer>
                      <span>{sample.runId}</span>
                      <span>{selection.metric}</span>
                    </footer>
                  </article>
                ))}
              </div>
            </>
          ) : null}
        </div>
      </section>
    </div>
  );
}
