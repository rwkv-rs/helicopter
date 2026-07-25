import { expect, test } from "@playwright/test";

test("keeps every parameter scale visible across comparison contracts", async ({
  page,
}) => {
  await page.goto("/?page=dashboard");

  for (const scale of ["1.5B", "2.9B", "7.2B", "13.3B"]) {
    await expect(page.getByRole("columnheader", { name: scale })).toBeVisible();
  }
  await expect(page.getByRole("columnheader", { name: "G1G" }).first()).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "G1H" }).first()).toBeVisible();

  await page.getByRole("button", { name: "Qwen3.5 vs RWKV" }).click();
  await expect(page.getByText("参数差 12.8%")).toBeVisible();
  await expect(
    page.getByRole("button", { name: /AIME24 13\.3B .* 作答详情/ }),
  ).toHaveCount(0);

  await page.getByRole("button", { name: "Prompt template" }).click();
  await expect(page.getByText("✿\\nBot", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("\\n\\nAssistant", { exact: true }).first()).toBeVisible();
});

test("traces one score through sampled outcomes and full context", async ({
  page,
}) => {
  await page.goto("/?page=dashboard");

  const panel = page.getByRole("region", { name: "作答详情" });
  await expect(panel.getByText("未选择", { exact: true })).toBeVisible();

  await page
    .getByRole("button", {
      name: /AIME24 1\.5B G1G .* 作答详情/,
    })
    .click();

  await expect(panel.getByLabel("模型架构 RWKV")).toBeVisible();
  await expect(panel.getByLabel("模型代际 G1G")).toBeVisible();
  await expect(panel.getByLabel("参数量 1.5B")).toBeVisible();
  await expect(panel.getByLabel("Benchmark AIME24")).toBeVisible();
  await expect(
    panel.getByText("User✿{task.problem}✿\nBot✿<think", { exact: true }),
  ).toBeVisible();
  await expect(panel.getByText("32768", { exact: true })).toBeVisible();

  await expect(panel.getByRole("tab", { name: /正确作答 10/ })).toBeVisible();
  await expect(panel.getByRole("table").getByRole("row")).toHaveCount(11);
  await expect(panel.getByText("true", { exact: true })).toHaveCount(10);

  await panel.getByRole("button", { name: /查看 aime24-0001 完整上下文/ }).click();
  let dialog = page.getByRole("dialog", { name: "aime24-0001 完整上下文" });
  await expect(dialog.getByText(/设正整数 x、y/)).toBeVisible();
  await expect(dialog.getByText(/已完成逐步推理并校验结果/)).toBeVisible();
  await expect(dialog.getByLabel("模型代际 G1G")).toBeVisible();
  await expect(dialog.getByText("true", { exact: true })).toBeVisible();
  await dialog.getByRole("button", { name: "关闭" }).click();

  await panel.getByRole("tab", { name: /错误作答/ }).click();
  await expect(panel.getByRole("table").getByRole("row")).toHaveCount(11);
  await expect(panel.getByText("false", { exact: true })).toHaveCount(10);
  await panel.getByRole("button", { name: /查看 aime24-0001 完整上下文/ }).click();
  dialog = page.getByRole("dialog", { name: "aime24-0001 完整上下文" });
  await expect(dialog.getByText("answer_mismatch")).toBeVisible();
  await dialog.getByRole("button", { name: "关闭" }).click();

  await panel.getByRole("tab", { name: /未能作答/ }).click();
  await expect(panel.getByRole("table").getByRole("row")).toHaveCount(11);
  await expect(panel.getByText("n/a", { exact: true })).toHaveCount(10);
  await expect(panel.getByText("—", { exact: true })).toHaveCount(10);

  await panel.getByRole("button", { name: "清除选择" }).click();
  await expect(panel.getByText("未选择", { exact: true })).toBeVisible();
});

test("switches history series and exposes selected run provenance", async ({
  page,
}) => {
  await page.goto("/?page=history");

  await expect(page.getByRole("img", { name: /分数历史/ })).toHaveCount(4);
  await page.getByRole("button", { name: /r01 A/ }).first().click();
  await expect(page.getByText(/^mock-generation-/)).toBeVisible();

  await page.getByRole("button", { name: "fp16 vs fp32io16" }).click();
  await expect(page.getByText("1.5B · fp16 vs fp32io16")).toBeVisible();
  await expect(page.getByRole("img", { name: /分数历史/ })).toHaveCount(4);
  await expect(page.getByText("点击任意柱子查看分数来源。")).toBeVisible();
});

test("falls back from the retired admin URL to the scoreboard", async ({ page }) => {
  await page.goto("/?page=admin");
  await expect(page.getByText("评测看板 · 全参数规模对比")).toBeVisible();
});
