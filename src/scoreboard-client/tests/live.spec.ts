import { expect, test } from "@playwright/test";


test("renders the live parser, gzip, PostgreSQL and query API chain", async ({
  page,
}) => {
  await page.goto("/?page=dashboard");

  await expect(page.getByRole("columnheader", { name: "small.pth" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "large.pth" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "fp16" }).first()).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "fp32io16" }).first(),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: /1 exact_match/ })).toHaveCount(4);

  await page.getByRole("button", { name: /1 exact_match/ }).first().click();
  const details = page.getByRole("region", { name: "评估详情" });
  await expect(details.getByText("publisher audit: playwright-live")).toBeVisible();
  await expect(details.getByText("completion 1")).toBeVisible();
  await expect(details.getByText("completion 2")).toBeVisible();
  await expect(details.getByText("first", { exact: true })).toBeVisible();
  await expect(details.getByText("second", { exact: true })).toBeVisible();
  await expect(details.getByText("Doc 0 · correct")).toBeVisible();
  const firstSample = details.getByRole("article", {
    name: "Doc 0 details",
  });
  await expect(firstSample.getByText('"exact_match": 1')).toBeVisible();
  await expect(firstSample.getByText(/shared-0/)).toBeVisible();
});
