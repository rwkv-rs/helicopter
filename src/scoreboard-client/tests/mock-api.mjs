import { createServer } from "node:http";

const server = createServer((request, response) => {
  response.setHeader("Content-Type", "application/json");
  if (request.url === "/api/v1/meta") {
    response.end(JSON.stringify({ api_version: "v1", schema_state: "ready", run_count: 0, model_count: 0 }));
    return;
  }
  if (request.url?.startsWith("/api/v1/leaderboard")) {
    response.end(JSON.stringify({ items: [], next_cursor: null }));
    return;
  }
  response.statusCode = 404;
  response.end(JSON.stringify({ error: { code: "not_found", message: "not found" } }));
});

server.listen(32148, "127.0.0.1");
