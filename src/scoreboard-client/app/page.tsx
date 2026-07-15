import { scoreboard } from "../lib/api";


export const dynamic = "force-dynamic";

export default async function Home() {
  const [{ data: metadata, error: metadataError }, { data: leaderboard, error: leaderboardError }] =
    await Promise.all([
      scoreboard.GET("/api/v1/meta"),
      scoreboard.GET("/api/v1/leaderboard", { params: { query: { limit: 100 } } }),
    ]);
  const error = metadataError ?? leaderboardError;

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <h1>Helicopter Scoreboard</h1>
          <div className="subtitle">可验证、可比较的 LightEval 结果</div>
        </div>
      </header>
      {error ? (
        <div className="error-bar">Scoreboard API 请求失败：{JSON.stringify(error)}</div>
      ) : (
        <>
          <section className="card">
            <div className="card-title">概览</div>
            <p>{metadata?.run_count ?? 0} 个 official run · {metadata?.model_count ?? 0} 个模型</p>
          </section>
          <section className="card">
            <div className="card-title">Official leaderboard</div>
            <table>
              <thead>
                <tr>
                  <th>模型</th><th>任务</th><th>split</th><th>CoT</th><th>修补</th><th>指标</th><th>分数</th>
                </tr>
              </thead>
              <tbody>
                {(leaderboard?.items ?? []).map((row) => (
                  <tr key={row.run_id}>
                    <td>{row.model_name}</td>
                    <td>{row.suite}/{row.task_name}@{row.task_version}</td>
                    <td>{row.split_name} · {row.fewshot}-shot</td>
                    <td>{row.cot_mode}</td>
                    <td>{row.repair_strategy}</td>
                    <td>{row.metric_name}</td>
                    <td>{row.metric_value.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {(leaderboard?.items.length ?? 0) === 0 ? <div className="empty">暂无 official completed 结果。</div> : null}
          </section>
        </>
      )}
    </main>
  );
}
