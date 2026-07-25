"use client";

import { useEffect, useState } from "react";

import { useComparisonStore } from "./store";
import type {
  AnswerOutcome,
  AnswerSample,
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

function PassedBadge({ value }: { value: boolean | null }) {
  return (
    <span
      className={`answer-pass-badge ${value === null ? "unanswered" : value ? "passed" : "failed"}`}
    >
      {value === null ? "n/a" : value ? "true" : "false"}
    </span>
  );
}

function ContextDetailModal({
  sample,
  onClose,
}: {
  sample: AnswerSample;
  onClose: () => void;
}) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <div className="modal-backdrop answer-context-backdrop" onClick={onClose}>
      <section
        aria-label={`${sample.problemId} 完整上下文`}
        aria-modal="true"
        className="modal answer-context-modal"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
      >
        <header className="modal-head">
          <div>
            <div className="card-title">完整模型上下文</div>
            <div className="context-outcome">
              <span>problem_id={sample.problemId}</span>
              <span>repeat_id={sample.repeatId}</span>
              <PassedBadge value={sample.isPassed} />
            </div>
          </div>
          <button className="btn" onClick={onClose} type="button">
            关闭
          </button>
        </header>
        <div className="modal-body answer-context-body">
          <div className="modal-col">
            <div className="stage">
              <div className="stage-label">assembled prompt</div>
              <pre>{sample.context.assembledPrompt}</pre>
            </div>
            <div className="stage">
              <div className="stage-label">raw completion</div>
              <pre>{sample.context.rawCompletion || "无原始输出"}</pre>
            </div>
            <div className="stage">
              <div className="stage-label">problem</div>
              <pre>{sample.context.problem}</pre>
            </div>
          </div>
          <div className="modal-col right">
            <div className="card-title">scoring result</div>
            <dl className="answer-context-meta">
              <dt>ground_truth</dt>
              <dd>{sample.groundTruth}</dd>
              <dt>extracted_answer</dt>
              <dd>{sample.extractedAnswer || "—"}</dd>
              <dt>is_passed</dt>
              <dd>
                <PassedBadge value={sample.isPassed} />
              </dd>
              <dt>fail_reason</dt>
              <dd>{sample.context.failReason || "—"}</dd>
            </dl>
            <div className="card-title token-title">generation metadata</div>
            <dl className="answer-context-meta">
              <dt>model</dt>
              <dd>{sample.context.model}</dd>
              <dt>run_id</dt>
              <dd>{sample.context.runId}</dd>
              <dt>metric</dt>
              <dd>{sample.context.metric}</dd>
              <dt>generated_tokens</dt>
              <dd>{sample.context.generatedTokens}</dd>
              <dt>latency_ms</dt>
              <dd>{sample.context.latencyMs}</dd>
            </dl>
          </div>
        </div>
      </section>
    </div>
  );
}

export function AnswerDetailsPanel() {
  const { state, dispatch, loadAnswerSamples } = useComparisonStore();
  const selection = state.selectedScoreCell;
  const [activeOutcome, setActiveOutcome] = useState<AnswerOutcome>("correct");
  const [data, setData] = useState<AnswerSampleGroups | null>(null);
  const [detailSample, setDetailSample] = useState<AnswerSample | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDetailSample(null);
    if (!selection) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }
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

  const activeGroup = data?.[activeOutcome] ?? null;

  return (
    <section aria-label="作答详情" className="card answer-detail-panel">
      <header className="panel-head answer-detail-head">
        <div>
          <div className="card-title">作答详情</div>
          {selection ? (
            <SelectionSummary selection={selection} />
          ) : (
            <div className="answer-unselected-label">未选择 benchmark</div>
          )}
        </div>
        {selection ? (
          <button
            className="btn answer-close"
            onClick={() =>
              dispatch({ type: "select-score-cell", selection: null })
            }
            type="button"
          >
            清除选择
          </button>
        ) : null}
      </header>

      <nav className="answer-tabs" aria-label="作答结果" role="tablist">
        {OUTCOMES.map((outcome) => {
          const group = data?.[outcome.id];
          return (
            <button
              aria-selected={activeOutcome === outcome.id}
              className={`answer-tab ${outcome.id}${activeOutcome === outcome.id ? " active" : ""}`}
              disabled={!selection}
              key={outcome.id}
              onClick={() => {
                setActiveOutcome(outcome.id);
                setDetailSample(null);
              }}
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
        {!selection ? (
          <div className="answer-unselected">
            <strong>未选择</strong>
            <span>点击上方任意分数，查看对应 benchmark 的抽样作答。</span>
          </div>
        ) : null}
        {selection && loading ? (
          <div className="spinner">正在抽取作答样本…</div>
        ) : null}
        {selection && error ? (
          <div className="error-bar">加载失败：{error}</div>
        ) : null}
        {selection && !loading && !error && activeGroup ? (
          <>
            <div className="answer-sample-note">
              从该结果类别的 {activeGroup.total} 条记录中随机抽取{" "}
              <strong>{activeGroup.items.length}</strong> 条
            </div>
            <div className="answer-records-wrap">
              <table className="answer-records-table">
                <colgroup>
                  <col className="answer-problem-col" />
                  <col className="answer-repeat-col" />
                  <col className="answer-ground-col" />
                  <col className="answer-model-col" />
                  <col className="answer-passed-col" />
                  <col className="answer-detail-col" />
                </colgroup>
                <thead>
                  <tr>
                    <th>题目 ID</th>
                    <th>repeat_id</th>
                    <th>ground_truth</th>
                    <th>模型作答（判分器提取）</th>
                    <th>is_passed</th>
                    <th>detail</th>
                  </tr>
                </thead>
                <tbody>
                  {activeGroup.items.map((sample) => (
                    <tr key={sample.id}>
                      <td className="answer-problem-id">{sample.problemId}</td>
                      <td>{sample.repeatId}</td>
                      <td className="answer-record-value" title={sample.groundTruth}>
                        {sample.groundTruth}
                      </td>
                      <td
                        className={`answer-record-value${sample.extractedAnswer ? "" : " empty"}`}
                        title={sample.extractedAnswer}
                      >
                        {sample.extractedAnswer || "—"}
                      </td>
                      <td>
                        <PassedBadge value={sample.isPassed} />
                      </td>
                      <td>
                        <button
                          aria-label={`查看 ${sample.problemId} 完整上下文`}
                          className="btn answer-detail-button"
                          onClick={() => setDetailSample(sample)}
                          type="button"
                        >
                          detail
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
      </div>
      {detailSample ? (
        <ContextDetailModal
          onClose={() => setDetailSample(null)}
          sample={detailSample}
        />
      ) : null}
    </section>
  );
}
