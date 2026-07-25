import { expect, type Page, test } from "@playwright/test";

const g1g = {
  label: "RWKV G1G 1.5B",
  architecture: "RWKV",
  generation: "G1G",
  parameters: "1.5B",
};
const g1h = { ...g1g, label: "RWKV G1H 1.5B", generation: "G1H" };
const comparison = {
  id: "generation",
  label: "G1G vs G1H",
  short_label: "代际",
  a_label: "G1G",
  b_label: "G1H",
  contract: "相同 prompt、precision、sampling 与输出边界。",
};
const parameterGroup = {
  id: "1.5b",
  label: "1.5B",
  a_model: g1g,
  b_model: g1h,
  parameter_delta_percent: 0,
  comparable: true,
};

function evaluation(arm: "a" | "b") {
  const isA = arm === "a";
  return {
    evaluation_id: `evaluation-${arm}`,
    publication_id: `run:aime24:${arm}`,
    source_run_id: `run-${arm}`,
    source: "lighteval-ci",
    visibility: "non_official",
    created_at: isA ? "2026-07-24T12:00:00Z" : "2026-07-25T12:00:00Z",
    task_name: "aime24|0",
    model: isA ? g1g : g1h,
    benchmark: {
      label: "AIME24",
      domain: "math",
      evaluation_method: "cot",
      score_multiplier: 100,
    },
    evaluation: {
      prompt_profile: "unified",
      prompt_template: "User: {task.problem}\\n\\nAssistant: <think",
      precision: "fp32io16",
    },
    comparisons: [{ comparison, parameter_group: parameterGroup, arm }],
    sampling_config: {
      temperature: 0.96,
      top_p: 0.76,
      top_k: 32,
      max_tokens: 8192,
      seed: 42,
    },
    primary_metric: "exact_match",
    aggregates: { exact_match: isA ? 0.25 : 0.5 },
    diagnostics: {
      samples: 2,
      completions: 3,
      truncated: isA ? 1 : 0,
      non_truncated: isA ? 2 : 3,
      truncation_rate: isA ? 1 / 3 : 0,
      turn_boundary_violations: 0,
      turn_boundary_violation_rate: 0,
    },
  };
}

function sample(
  outcome: "correct" | "incorrect" | "unanswered" | "undetermined",
  index: number,
) {
  const correct = outcome === "correct";
  const incorrect = outcome === "incorrect";
  const unanswered = outcome === "unanswered";
  return {
    id: `evaluation-a:${index}`,
    sample_index: index,
    outcome,
    doc: {
      id: `aime24-${index}`,
      query: `problem ${index}`,
      choices: ["42"],
      gold_index: 0,
    },
    metric:
      outcome === "undetermined"
        ? { judge_score: 0.6 }
        : outcome === "unanswered"
          ? {}
          : { exact_match: correct ? 1 : 0 },
    model_response: {
      input: `assembled prompt ${index}`,
      text: unanswered
        ? []
        : correct
          ? ["<think>first</think>42", "<think>second</think>42"]
          : [incorrect ? "<think>wrong</think>41" : "free-form response"],
      text_post_processed: unanswered
        ? []
        : correct
          ? ["42", "42"]
          : [incorrect ? "41" : "free-form response"],
      output_tokens: unanswered ? [] : correct ? [[1, 2], [3, 4]] : [[5]],
    },
  };
}

async function serveApi(page: Page): Promise<void> {
  await page.route("**/api/evaluations", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        evaluations: [evaluation("b"), evaluation("a")],
        generated_at: "2026-07-25T13:00:00Z",
      }),
    }),
  );
  await page.route("**/api/evaluations/evaluation-a/samples?limit=10", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        evaluation_id: "evaluation-a",
        primary_metric: "exact_match",
        groups: {
          correct: { outcome: "correct", total: 1, items: [sample("correct", 0)] },
          incorrect: {
            outcome: "incorrect",
            total: 1,
            items: [sample("incorrect", 1)],
          },
          unanswered: {
            outcome: "unanswered",
            total: 1,
            items: [sample("unanswered", 2)],
          },
          undetermined: {
            outcome: "undetermined",
            total: 1,
            items: [sample("undetermined", 3)],
          },
        },
      }),
    }),
  );
}

test("pivots real evaluation summaries and loads details by evaluation id", async ({
  page,
}) => {
  await serveApi(page);
  await page.goto("/?page=dashboard");

  await expect(page.getByText("LightEval reported")).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "1.5B" })).toBeVisible();
  await expect(page.getByRole("button", { name: /AIME24 1\.5B G1G 25\.0%/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /AIME24 1\.5B G1H 50\.0%/ })).toBeVisible();

  await page.getByRole("button", { name: /AIME24 1\.5B G1G 25\.0%/ }).click();
  const panel = page.getByRole("region", { name: "作答详情" });
  await expect(panel.getByLabel("模型代际 G1G")).toBeVisible();
  await expect(panel.getByText("8192", { exact: true })).toBeVisible();
  await expect(panel.getByRole("tab", { name: /正确作答 1/ })).toBeVisible();
  await panel.getByRole("button", { name: /查看 aime24-0 完整上下文/ }).click();

  const dialog = page.getByRole("dialog", { name: "aime24-0 完整上下文" });
  await expect(dialog.getByText(/completion 1/)).toBeVisible();
  await expect(dialog.getByText(/completion 2/)).toBeVisible();
  await expect(dialog.getByText('{"exact_match":1}')).toBeVisible();
  await expect(dialog.getByText("—", { exact: true })).toHaveCount(2);
});

test("shows all faithful sample outcomes including undetermined", async ({ page }) => {
  await serveApi(page);
  await page.goto("/?page=dashboard");
  await page.getByRole("button", { name: /AIME24 1\.5B G1G 25\.0%/ }).click();
  const panel = page.getByRole("region", { name: "作答详情" });

  await expect(panel.getByRole("tab", { name: /正确作答 1/ })).toBeVisible();
  await expect(panel.getByRole("tab", { name: /错误作答 1/ })).toBeVisible();
  await expect(panel.getByRole("tab", { name: /未能作答 1/ })).toBeVisible();
  await expect(panel.getByRole("tab", { name: /无法判定 1/ })).toBeVisible();
});

test("uses API history provenance and handles empty datasets", async ({ page }) => {
  await serveApi(page);
  await page.goto("/?page=history");
  await expect(page.getByRole("img", { name: /分数历史/ })).toHaveCount(1);
  await page.getByRole("button", { name: /run-a A 25\.0%/ }).click();
  await expect(page.getByText("run-a", { exact: true })).toBeVisible();

  await page.unrouteAll();
  await page.route("**/api/evaluations", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ evaluations: [], generated_at: "2026-07-25T13:00:00Z" }),
    }),
  );
  await page.goto("/?page=dashboard");
  await expect(
    page.getByText("尚无带 comparison metadata 的 LightEval 结果。"),
  ).toBeVisible();
});

test("reports API failures instead of falling back to mock data", async ({ page }) => {
  await page.route("**/api/evaluations", (route) =>
    route.fulfill({ status: 503, body: "scoreboard unavailable" }),
  );
  await page.goto("/?page=dashboard");
  await expect(page.getByText(/加载失败：503: scoreboard unavailable/)).toBeVisible();
});
