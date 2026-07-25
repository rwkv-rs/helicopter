import { expect, test } from "@playwright/test";

test("shows every parameter scale for each comparison option", async ({ page }) => {
  await page.goto("/?page=dashboard");

  await expect(page.getByRole("heading", { name: "RWKV Skills" })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "对比维度" })).toBeVisible();
  await expect(page.getByText("临时展示数据")).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "1.5B" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "2.9B" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "7.2B" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "13.3B" })).toBeVisible();
  await expect(page.getByText("分数范围")).toHaveCount(0);
  await expect(page.getByRole("link", { name: "管理面板" })).toHaveCount(0);

  await page.getByRole("button", { name: "Qwen3.5 vs RWKV" }).click();
  await expect(page.getByText(/选择最接近参数量的 Qwen3.5/)).toBeVisible();
  await expect(page.getByText("N/A").first()).toBeVisible();

  await page.getByRole("button", { name: "Prompt template" }).click();
  await expect(page.getByText(/User✿\{task\.problem\}✿/)).toBeVisible();
});

test("shows a fixed answer panel with ten samples for every outcome", async ({ page }) => {
  await page.goto("/?page=dashboard");

  const panel = page.getByRole("region", { name: "作答详情" });
  await expect(panel).toBeVisible();
  await expect(panel.getByText("未选择 benchmark")).toBeVisible();
  await expect(panel.getByText("未选择", { exact: true })).toBeVisible();

  await page
    .getByRole("button", {
      name: /AIME24 1\.5B 前代 .* 作答详情/,
    })
    .click();

  await expect(panel.getByText("AIME24", { exact: true })).toBeVisible();
  await expect(panel.getByRole("tab", { name: /正确作答 10/ })).toBeVisible();
  await expect(panel.getByRole("tab", { name: /错误作答 10/ })).toBeVisible();
  await expect(panel.getByRole("tab", { name: /未能作答 10/ })).toBeVisible();
  await expect(panel.locator(".answer-sample-card.correct")).toHaveCount(10);

  await panel.getByRole("tab", { name: /错误作答/ }).click();
  await expect(panel.locator(".answer-sample-card.incorrect")).toHaveCount(10);
  await expect(panel.getByText("answer_mismatch").first()).toBeVisible();

  await panel.getByRole("tab", { name: /未能作答/ }).click();
  await expect(panel.locator(".answer-sample-card.unanswered")).toHaveCount(10);
  await expect(panel.getByText("无有效输出").first()).toBeVisible();

  await panel.getByRole("button", { name: "清除选择" }).click();
  await expect(panel.getByText("未选择 benchmark")).toBeVisible();
});

test("renders score history and opens point provenance", async ({ page }) => {
  await page.goto("/?page=history");

  await expect(page.getByText("分数来源", { exact: true })).toBeVisible();
  await expect(page.locator(".history-card")).toHaveCount(4);
  await expect(page.getByText("分数范围")).toHaveCount(0);
  await expect(page.getByRole("link", { name: "管理面板" })).toHaveCount(0);

  await page.locator(".history-bar").first().click();
  await expect(page.getByText("run_id", { exact: true })).toBeVisible();
  await expect(page.getByText(/^mock-generation-/)).toBeVisible();

  await page.getByRole("button", { name: "fp16 vs fp32io16" }).click();
  await expect(page.getByText("1.5B · fp16 vs fp32io16")).toBeVisible();
  await expect(page.locator(".history-card")).toHaveCount(4);
  await expect(page.getByText("点击任意柱子查看分数来源。")).toBeVisible();
});

test("legacy admin URL falls back to the scoreboard", async ({ page }) => {
  await page.goto("/?page=admin");
  await expect(page.getByText("评测看板 · 全参数规模对比")).toBeVisible();
  await expect(page.getByText("管理面板")).toHaveCount(0);
});
