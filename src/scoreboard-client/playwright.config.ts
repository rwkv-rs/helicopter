import { defineConfig, devices } from "@playwright/test";

const testPort = process.env.SCOREBOARD_CLIENT_TEST_PORT || "3011";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  webServer: {
    command: `bun run build && bun run start -- -p ${testPort}`,
    url: `http://127.0.0.1:${testPort}/?page=dashboard`,
    reuseExistingServer: false,
    timeout: 120_000,
  },
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
