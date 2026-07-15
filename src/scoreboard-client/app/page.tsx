import { AdminPage } from "../components/AdminPage";
import { DashboardPage } from "../components/DashboardPage";
import { HistoryPage } from "../components/HistoryPage";
import { api } from "../lib/api";

export const dynamic = "force-dynamic";

type SearchParams = Promise<Record<string, string | string[] | undefined>>;
const PAGE_BASE = (process.env.NEXT_PUBLIC_BASE_PATH || "/new-eval").replace(/\/$/, "");

function pageHref(path: string): string {
  return PAGE_BASE ? `${PAGE_BASE}${path}` : path;
}

function value(params: Record<string, string | string[] | undefined>, key: string, fallback: string): string {
  const raw = params[key];
  if (Array.isArray(raw)) return raw[0] || fallback;
  return raw || fallback;
}

export default async function Home({ searchParams }: { searchParams: SearchParams }) {
  const params = await searchParams;
  const page = value(params, "page", "dashboard");
  const view = value(params, "view", "benchmark_detail_delta");
  const model = value(params, "model", "");
  const tab = value(params, "tab", "math");

  const isDashboard = page !== "history" && page !== "admin";
  const meta = isDashboard ? await api.meta() : null;
  const selectedModel = meta ? model || meta.auto_label : model;
  const leaderboard = meta ? await api.leaderboard(selectedModel, view) : null;

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <h1>RWKV Skills</h1>
          <div className="subtitle">
            {page === "history" ? "分数历史" : page === "admin" ? "调度器管理" : `评测看板 · ${leaderboard?.view_label ?? view}`}
          </div>
        </div>
        <nav className="page-nav">
          <a className={page === "dashboard" ? "active" : ""} href={pageHref(`/?page=dashboard&view=${view}&model=${encodeURIComponent(selectedModel)}&tab=${tab}`)}>
            评测看板
          </a>
          <a className={page === "history" ? "active" : ""} href={pageHref("/?page=history")}>
            分数历史
          </a>
          <a className={page === "admin" ? "active" : ""} href={pageHref("/?page=admin")}>
            管理面板
          </a>
        </nav>
      </header>
      {page === "history" ? (
        <HistoryPage />
      ) : page === "admin" ? (
        <AdminPage />
      ) : meta && leaderboard ? (
        <DashboardPage meta={meta} leaderboard={leaderboard} model={selectedModel} view={view} tab={tab} />
      ) : (
        <div className="empty">暂无数据。</div>
      )}
    </main>
  );
}
