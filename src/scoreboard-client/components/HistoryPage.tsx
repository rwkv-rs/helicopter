"use client";

import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Tooltip, XAxis, YAxis } from "recharts";

import { api } from "../lib/api";
import type { ScoreHistoryDetailResponse } from "../lib/dtos/api/score_history/detail";
import type {
  ScoreHistoryGroup,
  ScoreHistoryPoint,
  ScoreHistoryResponse,
} from "../lib/dtos/api/score_history";
import type { ScoreHistoryOptionsResponse } from "../lib/dtos/api/score_history/options";

const NORMAL_COLOR = "#5b8cff";
const NAIVE_COLOR = "#f5b14c";
const PER_BAR = 140;

function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function pctText(value: number | null): string {
  return value == null ? "—" : `${value.toFixed(1)}%`;
}

export function HistoryPage() {
  const [options, setOptions] = useState<ScoreHistoryOptionsResponse | null>(null);
  const [history, setHistory] = useState<ScoreHistoryResponse | null>(null);
  const [detail, setDetail] = useState<ScoreHistoryDetailResponse | null>(null);
  const [model, setModel] = useState("");
  const [benchmark, setBenchmark] = useState("");
  const [selectedTask, setSelectedTask] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .scoreHistoryOptions()
      .then((payload) => {
        if (cancelled) return;
        setOptions(payload);
        setModel((current) => current || payload.models[0] || "");
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const benchmarks = useMemo(() => {
    if (!options) return [];
    if (!model) return options.benchmarks;
    return [...new Set(options.pairs.filter((pair) => pair.model === model).map((pair) => pair.dataset))].sort();
  }, [model, options]);

  useEffect(() => {
    if (benchmarks.length && !benchmarks.includes(benchmark)) {
      setBenchmark(benchmarks[0]);
    }
  }, [benchmark, benchmarks]);

  useEffect(() => {
    if (!model || !benchmark) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setSelectedTask(null);
    setDetail(null);
    api
      .scoreHistory(model, benchmark)
      .then((payload) => {
        if (!cancelled) setHistory(payload);
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
  }, [benchmark, model]);

  useEffect(() => {
    if (selectedTask === null) return;
    let cancelled = false;
    setDetail(null);
    api
      .scoreHistoryDetail(selectedTask)
      .then((payload) => {
        if (!cancelled) setDetail(payload);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [selectedTask]);

  const refetch = () => {
    if (!model || !benchmark) return;
    setLoading(true);
    setError(null);
    api
      .scoreHistory(model, benchmark)
      .then(setHistory)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  };

  return (
    <div>
      <section className="card">
        <div className="controls">
          <div className="control-group">
            <label>模型权重</label>
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              {options?.models.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </div>
          <div className="control-group">
            <label>Benchmark</label>
            <select value={benchmark} onChange={(event) => setBenchmark(event.target.value)}>
              {benchmarks.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </div>
          <button className="btn btn-primary" type="button" onClick={refetch}>
            刷新
          </button>
          {history ? <span className="muted">共 {history.total} 条分数 · {history.groups.length} 张图</span> : null}
          {loading ? <span className="muted">加载中…</span> : null}
        </div>
      </section>
      {error ? <div className="error-bar">加载失败：{error}</div> : null}
      <div className="sh-layout">
        <div>
          {history && history.groups.length === 0 ? <div className="empty">该组合下暂无正式分数。</div> : null}
          {history?.groups.map((group) => (
            <HistoryChart key={group.cot_mode} group={group} selectedTask={selectedTask} onSelect={(point) => setSelectedTask(point.task_id)} />
          ))}
        </div>
        <aside className="sh-detail card">
          <div className="card-title">分数来源</div>
          <DetailPanel taskId={selectedTask} detail={detail} />
        </aside>
      </div>
    </div>
  );
}

function HistoryChart({
  group,
  selectedTask,
  onSelect,
}: {
  group: ScoreHistoryGroup;
  selectedTask: number | null;
  onSelect: (point: ScoreHistoryPoint) => void;
}) {
  const data = group.points.map((point, index) => ({ ...point, _label: fmtTime(point.created_at), _idx: index }));
  const width = Math.max(560, data.length * PER_BAR + 80);
  return (
    <section className="card">
      <div className="card-title">{group.cot_mode} · {group.points.length} 条分数</div>
      <div className="history-chart">
        <BarChart width={width} height={340} data={data} margin={{ top: 8, right: 16, bottom: 40, left: 0 }}>
          <CartesianGrid stroke="#252834" vertical={false} />
          <XAxis dataKey="_label" tick={{ fill: "#9aa3b2", fontSize: 11 }} interval={0} height={42} />
          <YAxis tick={{ fill: "#9aa3b2", fontSize: 11 }} domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
          <Tooltip cursor={{ fill: "rgba(91,140,255,0.08)" }} content={<HistoryTooltip />} />
          <Bar
            dataKey="percent"
            radius={[3, 3, 0, 0]}
            onClick={(value: unknown) => {
              const payload = (value as { payload?: ScoreHistoryPoint }).payload;
              if (payload) onSelect(payload);
            }}
          >
            {data.map((point) => (
              <Cell
                key={point.score_id}
                cursor="pointer"
                fill={point.board === "naive" ? NAIVE_COLOR : NORMAL_COLOR}
                opacity={selectedTask === null || selectedTask === point.task_id ? 1 : 0.45}
              />
            ))}
          </Bar>
        </BarChart>
      </div>
      <div className="history-point-list">
        {group.points.map((point) => (
          <button className="score-button" type="button" key={point.score_id} onClick={() => onSelect(point)}>
            task #{point.task_id} · {pctText(point.percent)}
          </button>
        ))}
      </div>
    </section>
  );
}

function HistoryTooltip({ active, payload }: { active?: boolean; payload?: { payload: ScoreHistoryPoint }[] }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div className="tooltip-pop">
      {`时间: ${fmtTime(point.created_at)}
score: ${pctText(point.percent)}
metric: ${point.metric ?? "—"}
evaluator: ${point.evaluator ?? "—"}
board: ${point.board === "naive" ? "朴素榜" : "正式榜"}
task_id: ${point.task_id}`}
    </div>
  );
}

function DetailPanel({ taskId, detail }: { taskId: number | null; detail: ScoreHistoryDetailResponse | null }) {
  if (taskId === null) return <div className="empty muted">点击任意柱子查看该分数来源。</div>;
  if (!detail) return <div className="spinner">加载中…</div>;
  if (!detail.found) return <div className="empty muted">未找到该 task 的详情。</div>;
  return <DetailBody detail={detail} />;
}

function DetailBody({ detail }: { detail: ScoreHistoryDetailResponse }) {
  const stages = Object.entries(detail.sampling.stages);
  return (
    <div>
      <div className="sh-detail-head">
        <span className="stat-pill stat-good">{pctText(detail.percent)}</span>
        <span className={`badge ${detail.board === "naive" ? "fail" : "pass"}`}>
          {detail.board === "naive" ? "朴素榜" : "正式榜"}
        </span>
        <span className="muted">task #{detail.task_id}</span>
      </div>
      <div className="muted detail-summary">
        {detail.model} · {detail.benchmark} · metric={detail.metric ?? "—"} · {detail.evaluator ?? "—"}
      </div>
      <div className="card-title">采样参数</div>
      <pre className="kv modal-text">
        {`effective_sample_count: ${detail.sampling.effective_sample_count ?? "—"}
avg_k: ${String(detail.sampling.avg_k ?? "—")}   pass_ks: ${JSON.stringify(detail.sampling.pass_ks ?? null)}
n_shot: ${String(detail.sampling.n_shot ?? "—")}   sample_limit: ${String(detail.sampling.sample_limit ?? "—")}
prompt_profile: ${detail.sampling.prompt_profile ?? "—"}`}
      </pre>
      {stages.map(([name, sampling]) => (
        <div className="stage" key={name}>
          <div className="stage-label">{name}</div>
          <pre>{`temperature: ${sampling.temperature ?? "—"}   top_k: ${sampling.top_k ?? "—"}   top_p: ${sampling.top_p ?? "—"}
max_tokens: ${sampling.max_tokens ?? "—"}
stop_tokens: ${sampling.stop_tokens.map((token) => `${token.id}(${token.token})`).join(" ") || "—"}`}</pre>
        </div>
      ))}
      <div className="card-title">Prompt（代表样本）</div>
      {detail.stages.length === 0 ? (
        <div className="muted">无 prompt context。</div>
      ) : (
        detail.stages.map((stage, index) => (
          <div className="stage" key={index}>
            <div className="stage-label">stage {index + 1} · stop_reason={String(stage.stop_reason ?? "—")}</div>
            <pre>{stage.prompt}</pre>
            {stage.completion ? <pre>{stage.completion}</pre> : null}
          </div>
        ))
      )}
    </div>
  );
}
