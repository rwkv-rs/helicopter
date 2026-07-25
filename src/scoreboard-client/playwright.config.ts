import { defineConfig, devices } from "@playwright/test";

const testPort = process.env.SCOREBOARD_CLIENT_TEST_PORT || "3011";
const apiPort = process.env.SCOREBOARD_E2E_PORT || "7862";
const apiBase = `http://127.0.0.1:${apiPort}`;

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  webServer: [
    {
      command:
        "../../.venv/bin/python ../scoreboard-server/tests/live_e2e_server.py",
      env: { SCOREBOARD_E2E_PORT: apiPort },
      url: `${apiBase}/api/evaluations`,
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `bun run build && bun run start -- -p ${testPort}`,
      env: { SCOREBOARD_API_BASE_URL: apiBase },
      url: `http://127.0.0.1:${testPort}/?page=dashboard`,
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
  use: {
    baseURL:
      process.env.SCOREBOARD_CLIENT_BASE_URL ||
      `http://127.0.0.1:${testPort}`,
    trace: "retain-on-failure"
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] }
    }
  ]
});
