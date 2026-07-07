"use client";

import { useState } from "react";

import { api } from "../lib/api";
import type { CapturePageResponse } from "../lib/dtos/api/capture_page";
import type { CellMeta, ChartPayload, LeaderboardResponse } from "../lib/dtos/api/leaderboard";
import type { MetaResponse } from "../lib/dtos/api/meta";
import { DomainCharts } from "./DomainCharts";
import { EvalRecordsPanel } from "./EvalRecordsPanel";
import { LeaderboardTable } from "./LeaderboardTable";
import { OverviewTable } from "./OverviewTable";

interface Props {
  meta: MetaResponse;
  leaderboard: LeaderboardResponse;
  model: string;
  view: string;
  tab: string;
}

export function DashboardPage({ meta, leaderboard, model, view, tab }: Props) {
  const [selectedMeta, setSelectedMeta] = useState<CellMeta | null>(null);
  const [capture, setCapture] = useState<
    | { status: "idle" }
    | { status: "saving" }
    | { status: "saved"; payload: CapturePageResponse }
    | { status: "error"; message: string }
  >({ status: "idle" });
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const activeTab = meta.domain_groups.some((group) => group.key === tab) ? tab : meta.domain_groups[0]?.key || "math";
  const domain = activeTab === "naive"
    ? null
    : leaderboard.domains.find((item) => item.key === activeTab);
  const naive = leaderboard.naive_board;
  const activeChart = activeTab in leaderboard.charts
    ? leaderboard.charts[activeTab as keyof ChartPayload]
    : null;

  const refresh = async () => {
    setRefreshError(null);
    try {
      await api.refresh();
      window.location.reload();
    } catch (err) {
      setRefreshError(err instanceof Error ? err.message : String(err));
    }
  };

  const capturePage = async () => {
    const target = new URL(window.location.href);
    target.searchParams.set("page", "dashboard");
    target.searchParams.set("model", model);
    target.searchParams.set("view", view);
    target.searchParams.set("tab", activeTab);
    setCapture({ status: "saving" });
    try {
      const payload = await api.capturePage({
        url: target.toString(),
        width: Math.max(window.innerWidth, 1440),
        height: Math.max(window.innerHeight, 1000),
      });
      setCapture({ status: "saved", payload });
    } catch (err) {
      setCapture({ status: "error", message: err instanceof Error ? err.message : String(err) });
    }
  };

  return (
    <>
      {meta.errors.length > 0 ? <div className="error-bar">{meta.errors.join("; ")}</div> : null}
      {leaderboard.errors.length > 0 ? <div className="error-bar">{leaderboard.errors.join("; ")}</div> : null}
      {refreshError ? <div className="error-bar">刷新失败：{refreshError}</div> : null}
      <section className="card">
        <form className="controls" action="/" method="get">
          <input type="hidden" name="page" value="dashboard" />
          <div className="control-group">
            <label>模型选择</label>
            <select name="model" defaultValue={model}>
              {meta.model_choices.map((choice) => (
                <option key={choice} value={choice}>
                  {choice}
                </option>
              ))}
            </select>
          </div>
          <div className="control-group">
            <label>视图模式</label>
            <select name="view" defaultValue={view}>
              {meta.table_views.map((choice) => (
                <option key={choice.key} value={choice.key}>
                  {choice.label}
                </option>
              ))}
            </select>
          </div>
          <input type="hidden" name="tab" value={activeTab} />
          <button className="btn btn-primary" type="submit">应用筛选</button>
          <button className="btn" type="button" onClick={refresh}>刷新数据</button>
          <button className="btn" type="button" onClick={capturePage} disabled={capture.status === "saving"}>
            {capture.status === "saving" ? "截图中..." : "长截图"}
          </button>
          <span className="muted">{leaderboard.selection.model_sequence.length} 个模型</span>
          {capture.status === "saved" ? <span className="capture-status">已保存：{capture.payload.path}</span> : null}
          {capture.status === "error" ? <span className="capture-status error">截图失败：{capture.message}</span> : null}
        </form>
      </section>
      <nav className="tabs">
        {meta.domain_groups.map((group) => (
          <a
            key={group.key}
            className={`tab${activeTab === group.key ? " active" : ""}`}
            href={`/?page=dashboard&model=${encodeURIComponent(model)}&view=${view}&tab=${group.key}`}
          >
            {group.label}
          </a>
        ))}
      </nav>
      {leaderboard.is_field_avg && leaderboard.overview ? (
        <section className="card">
          <div className="card-title">领域均分 · {leaderboard.view_label}</div>
          <OverviewTable rows={leaderboard.overview} columns={leaderboard.param_columns} isDelta={leaderboard.is_delta} />
        </section>
      ) : activeTab === "naive" && naive ? (
        <section className="card">
          <div className="card-title">朴素榜 · {leaderboard.view_label}</div>
          <LeaderboardTable
            paramColumns={naive.param_columns}
            isDelta={naive.is_delta}
            rows={naive.rows}
            onCellClick={setSelectedMeta}
          />
        </section>
      ) : domain ? (
        <>
          <section className="card">
            <div className="card-title">{domain.title} · {leaderboard.view_label}</div>
            <LeaderboardTable
              paramColumns={domain.param_columns}
              isDelta={leaderboard.is_delta}
              rows={domain.rows}
              onCellClick={setSelectedMeta}
            />
          </section>
          {activeChart ? (
            <section className="card chart-panel">
              <div className="card-title">图表</div>
              <DomainCharts chart={activeChart} />
            </section>
          ) : null}
        </>
      ) : (
        <div className="empty">暂无数据。</div>
      )}
      <EvalRecordsPanel meta={selectedMeta} onClose={() => setSelectedMeta(null)} />
    </>
  );
}
