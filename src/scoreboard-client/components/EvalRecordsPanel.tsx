"use client";

import { useEffect, useState } from "react";

import { api } from "../lib/api";
import type { EvalContextResponse } from "../lib/dtos/api/eval_context";
import type { EvalRecord, EvalRecordsResponse } from "../lib/dtos/api/eval_records";
import type { CellMeta } from "../lib/dtos/api/leaderboard";

interface Props {
  meta: CellMeta | null;
  onClose: () => void;
}

export function EvalRecordsPanel({ meta, onClose }: Props) {
  const [onlyWrong, setOnlyWrong] = useState(false);
  const [page, setPage] = useState(0);
  const [data, setData] = useState<EvalRecordsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const taskId = meta?.task_id ?? null;
  const limit = 15;

  useEffect(() => {
    setPage(0);
  }, [taskId]);

  useEffect(() => {
    if (taskId === null) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .evalRecords(taskId, onlyWrong, limit, page * limit)
      .then((payload) => {
        if (!cancelled) setData(payload);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [limit, onlyWrong, page, taskId]);

  if (!meta || taskId === null) {
    return null;
  }

  return (
    <section className="card eval-panel">
      <div className="panel-head">
        <div>
          <div className="card-title">评测明细</div>
          <div className="muted">
            {meta.benchmark_name} · {meta.eval_method} · {meta.model ?? "—"} · {onlyWrong ? "仅错题" : "全部"}
          </div>
        </div>
        <button className="btn" type="button" onClick={onClose}>
          关闭
        </button>
      </div>
      <div className="controls panel-controls">
        <label className="toggle">
          <input
            type="checkbox"
            checked={onlyWrong}
            onChange={(event) => {
              setOnlyWrong(event.target.checked);
              setPage(0);
            }}
          />
          仅展示错题
        </label>
        <button className="btn" type="button" disabled={page === 0 || loading} onClick={() => setPage((value) => value - 1)}>
          上一页
        </button>
        <button className="btn" type="button" disabled={!data?.has_more || loading} onClick={() => setPage((value) => value + 1)}>
          下一页
        </button>
        {loading ? <span className="muted">加载中…</span> : null}
      </div>
      {error ? <div className="error-bar">加载失败：{error}</div> : null}
      <EvalTable records={data?.records ?? []} taskId={taskId} />
    </section>
  );
}

function EvalTable({ records, taskId }: { records: EvalRecord[]; taskId: number }) {
  const [contextRecord, setContextRecord] = useState<EvalRecord | null>(null);
  if (!records.length) return <div className="empty">暂无数据。</div>;
  return (
    <>
      <div className="pivot-wrap">
        <table className="eval-table">
          <thead>
            <tr>
              <th>sample</th>
              <th>repeat</th>
              <th>pass_idx</th>
              <th>model_output</th>
              <th>ref_answer</th>
              <th>is_passed</th>
              <th>fail_reason</th>
              <th>context</th>
            </tr>
          </thead>
          <tbody>
            {records.map((record) => (
              <tr key={`${record.sample_index}-${record.repeat_index}-${record.pass_index}`}>
                <td>{record.sample_index}</td>
                <td>{record.repeat_index}</td>
                <td>{record.pass_index}</td>
                <td className="pre">{record.answer?.slice(0, 140) || "—"}</td>
                <td className="pre">{record.ref_answer?.slice(0, 140) || "—"}</td>
                <td>
                  <span className={`badge ${record.is_passed ? "pass" : "fail"}`}>
                    {record.is_passed ? "pass" : "fail"}
                  </span>
                </td>
                <td className="dim">{record.fail_reason?.slice(0, 80) || "—"}</td>
                <td>
                  <button className="context-preview-btn" type="button" onClick={() => setContextRecord(record)}>
                    {record.context_preview || "查看 context"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {contextRecord ? (
        <ContextModal
          record={contextRecord}
          taskId={taskId}
          onClose={() => setContextRecord(null)}
        />
      ) : null}
    </>
  );
}

function ContextModal({
  record,
  taskId,
  onClose,
}: {
  record: EvalRecord;
  taskId: number;
  onClose: () => void;
}) {
  const [data, setData] = useState<EvalContextResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    api
      .evalContext(taskId, record.sample_index, record.repeat_index, record.pass_index)
      .then((payload) => {
        if (!cancelled) setData(payload);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [record.pass_index, record.repeat_index, record.sample_index, taskId]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="card-title">context</div>
            <div className="context-outcome">
              <span>sample={record.sample_index}</span>
              <span>repeat={record.repeat_index}</span>
              <span>pass={record.pass_index}</span>
              <span className={record.is_passed ? "outcome-pass" : "outcome-fail"}>
                {record.is_passed ? "passed" : "failed"}
              </span>
            </div>
          </div>
          <button className="btn" type="button" onClick={onClose}>
            关闭
          </button>
        </div>
        {error ? <div className="error-bar">加载失败：{error}</div> : null}
        {!data && !error ? <div className="spinner">加载中…</div> : null}
        {data?.view === "structured" && data.context ? (
          <StructuredContext context={data.context} stopTokens={data.stop_tokens} errors={data.errors} />
        ) : null}
        {data?.view === "text" ? <pre className="kv modal-text">{data.raw_text}</pre> : null}
      </div>
    </div>
  );
}

function StructuredContext({
  context,
  stopTokens,
  errors,
}: {
  context: Record<string, unknown>;
  stopTokens: EvalContextResponse["stop_tokens"];
  errors: string[];
}) {
  const stages = Array.isArray(context.stages) ? context.stages : [];
  const samplingConfig = typeof context.sampling_config === "object" && context.sampling_config !== null
    ? context.sampling_config
    : {};
  return (
    <div className="modal-body">
      <div className="modal-col">
        {stages.length > 0 ? (
          stages.map((stageValue, index) => {
            const stage = typeof stageValue === "object" && stageValue !== null
              ? (stageValue as Record<string, unknown>)
              : {};
            return (
              <div className="stage" key={index}>
                <div className="stage-label">stage {index + 1} · stop_reason={String(stage.stop_reason ?? "—")}</div>
                <pre>{String(stage.prompt || "")}</pre>
                <pre>{String(stage.completion || "")}</pre>
              </div>
            );
          })
        ) : (
          <pre className="kv">{JSON.stringify(context, null, 2)}</pre>
        )}
        {errors.length > 0 ? <div className="error-bar">{errors.join("; ")}</div> : null}
      </div>
      <div className="modal-col right">
        <div className="card-title">sampling config</div>
        <pre className="kv">{JSON.stringify(samplingConfig, null, 2)}</pre>
        {Object.keys(stopTokens).length > 0 ? (
          <>
            <div className="card-title token-title">stop tokens</div>
            {Object.entries(stopTokens).map(([stageName, tokens]) => (
              <div key={stageName}>
                <div className="muted token-stage">{stageName}</div>
                <pre className="kv">{tokens.map((token) => `${token.id}\t${token.token}`).join("\n")}</pre>
              </div>
            ))}
          </>
        ) : null}
      </div>
    </div>
  );
}
