"use client";

import { useEffect, useMemo, useState } from "react";

import { getAdminToken, setAdminToken } from "../lib/admin_token";
import { api } from "../lib/api";
import type { AdminBackpressureResponse, BackpressureModel } from "../lib/dtos/api/admin/backpressure";
import type { AdminHealthResponse } from "../lib/dtos/api/admin/health";
import type { AdminEvalOptionsResponse } from "../lib/dtos/api/admin/eval/options";
import type { AdminEvalStatusResponse } from "../lib/dtos/api/admin/eval/status";

const TERMINAL = new Set(["idle", "completed", "cancelled", "failed"]);

function statusClass(status: string): string {
  if (status === "running" || status === "starting") return "stat-run";
  if (status === "paused" || status === "pausing") return "stat-warn";
  if (status === "completed") return "stat-good";
  if (status === "failed") return "stat-bad";
  return "stat-idle";
}

function fmtTime(ms: number | null): string {
  if (!ms) return "-";
  try {
    return new Date(ms).toLocaleString();
  } catch {
    return String(ms);
  }
}

export function AdminPage() {
  const [health, setHealth] = useState<AdminHealthResponse | null>(null);
  const [status, setStatus] = useState<AdminEvalStatusResponse | null>(null);
  const [options, setOptions] = useState<AdminEvalOptionsResponse | null>(null);
  const [draft, setDraft] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      api.adminHealth().catch(() => null),
      api.adminStatus(),
      api.adminOptions(),
      api.adminDraft(),
    ])
      .then(([healthPayload, statusPayload, optionsPayload, draftPayload]) => {
        if (cancelled) return;
        setHealth(healthPayload);
        setStatus(statusPayload);
        setOptions(optionsPayload);
        setDraft(draftPayload);
        setError(null);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [refreshToken]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      api.adminStatus()
        .then((next) => {
          setStatus(next);
          setError(null);
        })
        .catch((err) => setError(err instanceof Error ? err.message : String(err)));
    }, 3000);
    return () => window.clearInterval(timer);
  }, []);

  const disabled = useMemo(() => {
    const current = status?.status;
    return !!current && !TERMINAL.has(current);
  }, [status?.status]);

  return (
    <div>
      <section className="card" style={{ marginBottom: 18 }}>
        <div className="controls">
          <TokenField onChange={() => setRefreshToken((value) => value + 1)} />
          {health?.auth_required ? <span className="muted mono-sm">服务端已开启鉴权，请填写 Token。</span> : null}
          {health === null ? <span className="muted mono-sm">admin health 暂无响应。</span> : null}
        </div>
      </section>

      {error ? <div className="error-bar">读取调度器状态失败：{error}</div> : null}
      {status ? (
        <>
          <StatusPanel status={status} onStatus={setStatus} />
          <div className="admin-grid">
            <ConfigPanel disabled={disabled} draft={draft ?? {}} options={options} onStatus={setStatus} />
            <div>
              <QueuePanel status={status} />
              <TelemetryPanel status={status} />
            </div>
          </div>
        </>
      ) : (
        <div className="spinner">加载调度器状态...</div>
      )}
    </div>
  );
}

function StatusPanel({
  status,
  onStatus,
}: {
  status: AdminEvalStatusResponse;
  onStatus: (status: AdminEvalStatusResponse) => void;
}) {
  const [actionError, setActionError] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const isTerminal = TERMINAL.has(status.status);
  const canControl = !isTerminal && status.status !== "idle";
  const pct = Math.round((status.progress_percent ?? 0) * 100);

  const run = async (name: string, fn: () => Promise<AdminEvalStatusResponse>) => {
    setPending(name);
    setActionError(null);
    try {
      onStatus(await fn());
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setPending(null);
    }
  };

  return (
    <section className="card">
      <div className="admin-status-head">
        <span className={`stat-pill ${statusClass(status.status)}`}>{status.status}</span>
        {status.run_id ? <span className="muted mono-sm">{status.run_id}</span> : null}
        {status.desired_state && status.desired_state !== status.status ? (
          <span className="muted mono-sm">-&gt; {status.desired_state}</span>
        ) : null}
        <div className="admin-controls">
          <button className="btn" disabled={!canControl || pending === "pause"} onClick={() => run("pause", api.adminPause)}>
            暂停
          </button>
          <button className="btn" disabled={!canControl || pending === "resume"} onClick={() => run("resume", api.adminResume)}>
            恢复
          </button>
          <button className="btn btn-danger" disabled={!canControl || pending === "cancel"} onClick={() => run("cancel", api.adminCancel)}>
            取消
          </button>
        </div>
      </div>

      {status.error ? <div className="error-bar" style={{ marginTop: 12 }}>{status.error}</div> : null}
      {actionError ? <div className="error-bar" style={{ marginTop: 12 }}>{actionError}</div> : null}

      <div className="progress-bar" title={`${pct}%`}>
        <div className={`progress-fill ${statusClass(status.status)}`} style={{ width: `${pct}%` }} />
      </div>

      <div className="stat-grid">
        <Stat label="总任务" value={status.tasks_total} />
        <Stat label="待处理" value={status.pending_jobs} />
        <Stat label="运行中" value={status.running_jobs} />
        <Stat label="已完成" value={status.completed_jobs} good />
        <Stat label="失败" value={status.failed_jobs} bad={status.failed_jobs > 0} />
        <Stat label="进度" value={`${pct}%`} />
      </div>

      <div className="muted mono-sm" style={{ marginTop: 10 }}>
        开始 {fmtTime(status.started_at_unix_ms)} · 更新 {fmtTime(status.updated_at_unix_ms)}
        {status.finished_at_unix_ms ? ` · 结束 ${fmtTime(status.finished_at_unix_ms)}` : ""}
      </div>
    </section>
  );
}

function Stat({ label, value, good, bad }: { label: string; value: number | string; good?: boolean; bad?: boolean }) {
  return (
    <div className="stat-cell">
      <div className="stat-num" style={{ color: bad ? "var(--bad)" : good ? "var(--good)" : undefined }}>
        {value}
      </div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function QueuePanel({ status }: { status: AdminEvalStatusResponse }) {
  return (
    <section className="card">
      <div className="card-title">任务队列</div>
      <div className="queue-cols">
        <div>
          <div className="queue-head-label">运行中 ({status.active_jobs.length})</div>
          {status.active_jobs.length === 0 ? (
            <div className="muted">无</div>
          ) : (
            <ul className="job-list">
              {status.active_jobs.map((job) => (
                <li key={job} className="job-item active">{job}</li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <div className="queue-head-label">队列头部 ({status.queue_head.length})</div>
          {status.queue_head.length === 0 ? (
            <div className="muted">空</div>
          ) : (
            <ul className="job-list">
              {status.queue_head.map((job) => (
                <li key={job} className="job-item">{job}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}

function TelemetryPanel({ status }: { status: AdminEvalStatusResponse }) {
  const inferUrl = (status.request?.infer_base_url as string) || "";
  const [override, setOverride] = useState("");
  const [backpressure, setBackpressure] = useState<AdminBackpressureResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const url = override || inferUrl;

  const refresh = async () => {
    setError(null);
    try {
      setBackpressure(await api.adminBackpressure(url || undefined));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  return (
    <section className="card">
      <div className="card-title">GPU / 推理 worker 遥测</div>

      <div style={{ marginBottom: 12 }}>
        <span className="muted mono-sm">本地 GPU：</span>
        {status.available_gpus.length === 0 ? (
          <span className="muted"> 无上报</span>
        ) : (
          status.available_gpus.map((gpu) => <span key={gpu} className="chip">{gpu}</span>)
        )}
      </div>

      <div className="controls" style={{ marginBottom: 12 }}>
        <div className="control-group" style={{ flex: 1 }}>
          <label>infer_base_url（留空用当前任务配置）</label>
          <input
            type="text"
            value={override}
            placeholder={inferUrl || "http://127.0.0.1:8000/v1"}
            onChange={(event) => setOverride(event.target.value)}
            style={{ minWidth: 320 }}
          />
        </div>
        <button className="btn" type="button" onClick={() => void refresh()}>刷新</button>
      </div>

      {error ? <div className="error-bar">{error}</div> : null}
      {backpressure?.error ? <div className="error-bar">{backpressure.error}</div> : null}
      {!url ? <div className="muted">未配置远端推理服务，无 worker 遥测。</div> : null}

      {backpressure && backpressure.models.length > 0 ? (
        <div className="pivot-wrap">
          <table className="pivot">
            <thead>
              <tr>
                <th>模型</th>
                <th>状态</th>
                <th>路由 (健康/总)</th>
                <th>排队</th>
                <th>批大小</th>
                <th>失败批</th>
                <th>tok/s</th>
              </tr>
            </thead>
            <tbody>
              {backpressure.models.map((model: BackpressureModel) => (
                <tr key={model.model_slug}>
                  <td className="bench">{model.model}</td>
                  <td className={model.status === "ok" ? "delta up" : "delta down"}>{model.status}</td>
                  <td className="score">{model.ok_route_count}/{model.route_count}</td>
                  <td className="score">{model.pending_queue}</td>
                  <td className="score">{model.max_batch_size ?? "-"}</td>
                  <td className="score">{model.failed_batches}</td>
                  <td className="score">{model.last_total_tok_s != null ? model.last_total_tok_s.toFixed(1) : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}

function ConfigPanel({
  disabled,
  draft,
  options,
  onStatus,
}: {
  disabled: boolean;
  draft: Record<string, unknown>;
  options: AdminEvalOptionsResponse | null;
  onStatus: (status: AdminEvalStatusResponse) => void;
}) {
  const [text, setText] = useState("{}");
  const [parseError, setParseError] = useState<string | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    setText(JSON.stringify(draft, null, 2));
  }, [draft]);

  const patch = (key: string, value: unknown) => {
    try {
      const obj = JSON.parse(text || "{}") as Record<string, unknown>;
      obj[key] = value;
      setText(JSON.stringify(obj, null, 2));
      setParseError(null);
    } catch {
      setParseError("当前 JSON 无法解析，无法快捷修改。");
    }
  };

  const onStart = async () => {
    let payload: Record<string, unknown>;
    try {
      payload = JSON.parse(text || "{}") as Record<string, unknown>;
      setParseError(null);
      setStartError(null);
    } catch (err) {
      setParseError(`JSON 解析失败：${String(err)}`);
      return;
    }
    setStarting(true);
    try {
      onStatus(await api.adminStart(payload));
    } catch (err) {
      setStartError(err instanceof Error ? err.message : String(err));
    } finally {
      setStarting(false);
    }
  };

  return (
    <section className="card">
      <div className="card-title">评测配置 · 启动</div>

      {options ? (
        <div className="controls" style={{ marginBottom: 14 }}>
          <QuickSelect label="model_select" options={options.model_select} onPick={(value) => patch("model_select", value)} />
          <QuickSelect label="run_mode" options={options.run_mode} onPick={(value) => patch("run_mode", value)} />
          <QuickSelect label="domains（追加）" options={options.domains} onPick={(value) => patch("domains", [value])} />
          <QuickSelect label="protocol" options={options.protocol} onPick={(value) => patch("infer_protocol", value)} />
        </div>
      ) : null}

      <textarea className="config-editor" value={text} spellCheck={false} onChange={(event) => setText(event.target.value)} />

      {parseError ? <div className="error-bar" style={{ marginTop: 10 }}>{parseError}</div> : null}
      {startError ? <div className="error-bar" style={{ marginTop: 10 }}>{startError}</div> : null}

      <div style={{ marginTop: 12, display: "flex", gap: 10, alignItems: "center" }}>
        <button className="btn btn-primary" type="button" disabled={disabled || starting} onClick={() => void onStart()}>
          {starting ? "启动中..." : "启动评测"}
        </button>
        {disabled ? <span className="muted mono-sm">已有任务在运行，无法启动新任务。</span> : null}
      </div>
    </section>
  );
}

function QuickSelect({ label, options, onPick }: { label: string; options: string[]; onPick: (value: string) => void }) {
  return (
    <div className="control-group">
      <label>{label}</label>
      <select
        defaultValue=""
        onChange={(event) => {
          if (event.target.value) onPick(event.target.value);
          event.target.value = "";
        }}
        style={{ minWidth: 160 }}
      >
        <option value="" disabled>
          选择...
        </option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );
}

function TokenField({ onChange }: { onChange: () => void }) {
  const [value, setValue] = useState(getAdminToken());
  return (
    <div className="control-group">
      <label>Admin Token（可选）</label>
      <input
        type="password"
        value={value}
        placeholder="Bearer token，可留空"
        onChange={(event) => {
          const next = event.target.value.trim();
          setValue(next);
          setAdminToken(next);
          onChange();
        }}
        style={{ minWidth: 220 }}
      />
    </div>
  );
}
