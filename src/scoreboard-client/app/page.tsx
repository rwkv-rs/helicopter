import { EvaluationProvider } from "../components/EvaluationProvider";
import { DashboardPage } from "../components/DashboardPage";
import { HistoryPage } from "../components/HistoryPage";

export const dynamic = "force-dynamic";

type SearchParams = Promise<Record<string, string | string[] | undefined>>;
const PAGE_BASE = (process.env.NEXT_PUBLIC_BASE_PATH ?? "").replace(/\/$/, "");

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
  const isHistory = page === "history";

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <h1>RWKV Skills</h1>
          <div className="subtitle">
            {isHistory
              ? "评估历史 · 原生指标与来源"
              : "评测看板 · 配置的 LightEval 评估集"}
          </div>
        </div>
        <nav className="page-nav">
          <a className={!isHistory ? "active" : ""} href={pageHref("/?page=dashboard")}>
            评测看板
          </a>
          <a className={isHistory ? "active" : ""} href={pageHref("/?page=history")}>
            分数历史
          </a>
        </nav>
      </header>
      <EvaluationProvider>
        {isHistory ? <HistoryPage /> : <DashboardPage />}
      </EvaluationProvider>
    </main>
  );
}
