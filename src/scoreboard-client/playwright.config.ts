import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  use: {
    baseURL: process.env.SCOREBOARD_CLIENT_BASE_URL || "http://127.0.0.1:32147",
    trace: "retain-on-failure"
  },
  webServer: [
    {
      command: "bun tests/mock-api.mjs",
      port: 32148,
      reuseExistingServer: false
    },
    {
      command: "bun run build && bun run start --hostname 127.0.0.1 --port 32147",
      port: 32147,
      reuseExistingServer: false,
      env: { SCOREBOARD_API_BASE_URL: "http://127.0.0.1:32148" }
    }
  ],
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] }
    }
  ]
});
