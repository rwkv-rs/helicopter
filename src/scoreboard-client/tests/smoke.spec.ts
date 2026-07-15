import { test, expect } from "@playwright/test";

test("renders the SSR scoreboard table", async ({ page }) => {
  await page.goto("/?page=dashboard&view=benchmark_detail_latest&tab=math");
  await expect(page.getByRole("heading", { name: "RWKV Skills" })).toBeVisible();
  await expect(page.getByText("gsm8k_test")).toBeVisible();
  await expect(page.getByText("50.0%")).toBeVisible();
  await page.getByRole("button", { name: "50.0%" }).click();
  await expect(page.getByText("评测明细")).toBeVisible();
  await expect(page.getByText("wrong arithmetic")).toBeVisible();
  await page.getByRole("button", { name: /查看 context|What is/ }).first().click();
  await expect(page.locator(".modal pre").filter({ hasText: "What is 1+1?" })).toBeVisible();
  await page.locator(".modal").getByRole("button", { name: "关闭" }).click();
  await page.getByRole("button", { name: "长截图" }).click();
  await expect(page.getByText(/已保存：.*scoreboard-screenshots/)).toBeVisible({ timeout: 120_000 });

  await page.goto("/?page=dashboard&view=field_avg_latest&tab=math");
  await expect(page.getByText("领域均分 · 领域均分（最新）")).toBeVisible();
  await expect(page.getByText(/数学推理/)).toBeVisible();

  await page.goto("/?page=dashboard&view=benchmark_detail_latest&tab=coding");
  await expect(page.getByText("图表", { exact: true })).toBeVisible();
  await expect(page.locator(".chart-panel").getByText("HUMANEVAL", { exact: true })).toBeVisible();

  await page.goto("/?page=history");
  await expect(page.getByText("分数来源", { exact: true })).toBeVisible();
  await expect(page.getByText("共 1 条分数")).toBeVisible();
  await page.getByRole("button", { name: /task #1/ }).click();
  await expect(page.getByText(/metric=avg@1/)).toBeVisible();

  await page.goto("/?page=admin");
  await expect(page.getByText("评测配置 · 启动")).toBeVisible();
  await expect(page.getByText("GPU / 推理 worker 遥测")).toBeVisible();
  await expect(page.getByText("Scheduler control is not part of the migrated scoreboard server.").first()).toBeVisible();
  await page.getByRole("button", { name: "启动评测" }).click();
  await expect(page.getByText(/501: Scheduler control is not part/).first()).toBeVisible();
});
