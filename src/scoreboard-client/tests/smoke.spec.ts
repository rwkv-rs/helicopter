import { expect, test } from "@playwright/test";

test("generated client renders the versioned scoreboard payload", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Helicopter Scoreboard" })).toBeVisible();
  await expect(page.getByText("0 个 official run · 0 个模型")).toBeVisible();
  await expect(page.getByText("暂无 official completed 结果。")).toBeVisible();
});
