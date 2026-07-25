import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  webServer: {
    command: "bun run build && bun run start",
    url: "http://127.0.0.1:3010/?page=dashboard",
    reuseExistingServer: false,
    timeout: 120_000,
  },
  use: {
    baseURL: process.env.SCOREBOARD_CLIENT_BASE_URL || "http://127.0.0.1:3010",
    trace: "retain-on-failure"
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] }
    }
  ]
});
