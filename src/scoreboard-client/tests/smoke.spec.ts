import { expect, type Page, test } from "@playwright/test";

import sharedFixture from "../test_data/evaluation.json" with {
  type: "json",
};
import type { EvaluationSummary } from "../lib/evaluation_types";

type WkvMode = "fp16" | "fp32io16";

function evaluation(
  weight: "small.pth" | "large.pth",
  mode: WkvMode,
  task = sharedFixture.registry_task.identity,
  tags: string[] = sharedFixture.registry_task.upstream_tags,
  evaluator: "lighteval" | "lm-eval" = "lighteval",
): EvaluationSummary {
  const sha = weight === "small.pth" ? "a".repeat(64) : "b".repeat(64);
  const score = weight === "small.pth" ? 0.25 : 0.5;
  return {
    evaluation_id: `${weight}-${mode}-${task}`,
    campaign_id: "campaign-complete",
    task_identity: `${sha}:${mode}:${task}`,
    created_at: "2026-07-25T12:00:00Z",
    completed_at: "2026-07-25T13:00:00Z",
    task: {
      identity: `${sha}:${mode}:${task}`,
      weight_sha256: sha,
      weight_display_name: weight,
      wkv_mode: mode,
      selector: task.split("|", 1)[0].split(":", 1)[0],
      task_name: task,
      task_version: "0",
      module_family: sharedFixture.registry_task.module_family,
      module: sharedFixture.registry_task.module,
      dataset: sharedFixture.registry_task.dataset,
      subset: sharedFixture.registry_task.subset,
      evaluation_splits: sharedFixture.registry_task.evaluation_splits,
      languages: sharedFixture.registry_task.languages,
      upstream_tags: tags,
    },
    artifact: {
      ...(evaluator === "lighteval"
        ? { lighteval_version: "0.13.0" }
        : { evaluator: { name: "lm-eval" as const, version: "0.4.12" } }),
      results_path:
        evaluator === "lighteval"
          ? "results/model/results_stamp.json"
          : "results.json",
      details_paths:
        evaluator === "lighteval"
          ? ["details/model/stamp/details_task_stamp.parquet"]
          : ["samples/0000.json"],
    },
    task_config: {
      original_num_docs: 2,
      effective_num_docs: 2,
      skipped_multiselect_docs: 0,
    },
    model: {
      weight_sha256: sha,
      weight_display_name: weight,
      wkv_mode: mode,
      prompt_template: evaluator === "lighteval" ? "assistant" : "none",
      gemm_policy:
        mode === "fp16" ? "fp16-accumulation" : "fp32-accumulation",
      gpu: "NVIDIA RTX PRO 6000",
      max_num_seqs: 1280,
      max_num_batched_tokens: 8192,
      dependency_versions: {
        [evaluator]: evaluator === "lighteval" ? "0.13.0" : "0.4.12",
        vllm: "fixture",
        torch: "fixture",
      },
      evaluator,
    },
    sampling_config: { max_new_tokens: 8192 },
    primary_metric: "exact_match",
    aggregates: { exact_match: score, stderr: 0.01 },
    diagnostics: {
      samples: 2,
      completions: 3,
      truncated: 1,
      non_truncated: 2,
      truncation_rate: 1 / 3,
      turn_boundary_violations: 0,
      turn_boundary_violation_rate: 0,
    },
    provenance: {
      config_digest: "1".repeat(64),
      registry_digest: "2".repeat(64),
      eval_contract_digest: "4".repeat(64),
      ...(evaluator === "lighteval"
        ? { lighteval_version: "0.13.0" }
        : { evaluator: { name: "lm-eval" as const, version: "0.4.12" } }),
      configured_selectors: ["gsm8k", "unavailable"],
      resolved_selectors: ["gsm8k"],
      skipped_selectors: ["unavailable"],
      publisher_principal: "eval-worker",
    },
  };
}

const sample = {
  id: "sample-0",
  sample_index: 0,
  document_index: 0,
  outcome: "undetermined",
  doc: sharedFixture.standard_rows[0].doc,
  metric: sharedFixture.standard_rows[0].metric,
  model_response: sharedFixture.standard_rows[0].model_response,
};

const logprobSample = {
  id: "sample-1",
  sample_index: 1,
  document_index: 1,
  outcome: "undetermined",
  doc: sharedFixture.standard_rows[1].doc,
  metric: sharedFixture.standard_rows[1].metric,
  model_response: sharedFixture.standard_rows[1].model_response,
};

const lmEvalSample = {
  ...sample,
  id: "sample-lm-eval",
  model_response: {
    arguments: [["prompt", { until: ["\\n"] }]],
    filtered_resps: ["native answer"],
    resps: [["native answer"]],
    target: "reference answer",
  },
};

async function serveApi(page: Page): Promise<void> {
  const evaluations = [
    evaluation("small.pth", "fp16"),
    evaluation("small.pth", "fp32io16"),
    evaluation(
      "large.pth",
      "fp16",
      sharedFixture.registry_task.identity,
      sharedFixture.registry_task.upstream_tags,
      "lm-eval",
    ),
    evaluation("small.pth", "fp16", "multi|0", ["math", "reasoning"]),
    evaluation("small.pth", "fp16", "untagged|0", []),
    // large/fp32io16 is deliberately absent.
  ];
  await page.route("**/api/evaluations?limit=5000&offset=0", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        evaluations,
        generated_at: "2026-07-25T13:00:00Z",
        total: evaluations.length,
        offset: 0,
        limit: 5000,
        next_offset: null,
      }),
    }),
  );
  await page.route("**/api/evaluations/*/samples?*", (route) => {
    const parts = new URL(route.request().url()).pathname.split("/");
    const evaluationId = decodeURIComponent(parts.at(-2) ?? "");
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        evaluation_id: evaluationId,
        primary_metric: "exact_match",
        total: evaluationId.includes("large.pth") ? 1 : 2,
        offset: 0,
        limit: 25,
        next_offset: null,
        items: evaluationId.includes("large.pth")
          ? [lmEvalSample]
          : [sample, logprobSample],
      }),
    });
  });
}

test("shows two weights, both WKV modes, native metrics and missing pairs", async ({
  page,
}) => {
  await serveApi(page);
  await page.goto("/?page=dashboard");

  await expect(page.getByRole("columnheader", { name: "small.pth" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "large.pth" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "fp16" }).first()).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "fp32io16" }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: /0.25 exact_match/ }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: /0.01 stderr/ }).first()).toBeVisible();
  await expect(page.getByLabel(/gsm8k\|0 fp32io16 结果缺失/)).toBeVisible();
});

test("filters tags and pages faithful multi-completion details", async ({ page }) => {
  await serveApi(page);
  await page.goto("/?page=dashboard");
  await expect(page.getByText("untagged|0", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "reasoning", exact: true }).click();
  await expect(page.getByText("multi|0", { exact: true })).toBeVisible();
  await expect(page.getByText("gsm8k|0", { exact: true })).toHaveCount(0);
  await expect(page.getByText("untagged|0", { exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: "math", exact: true }).click();
  await page.getByRole("button", { name: /0.25 exact_match/ }).first().click();

  const details = page.getByRole("region", { name: "评估详情" });
  await expect(details.getByText("publisher audit: eval-worker")).toBeVisible();
  await expect(details.getByText("prompt template: assistant")).toBeVisible();
  await expect(details.getByText("questions: 2 / 2")).toBeVisible();
  await expect(details.getByText("skipped multi-select: 0")).toBeVisible();
  await expect(details.getByText("completion 1")).toBeVisible();
  await expect(details.getByText("completion 2")).toBeVisible();
  await expect(details.getByText("first", { exact: true })).toBeVisible();
  await expect(details.getByText("second", { exact: true })).toBeVisible();
  await expect(details.getByText("turn boundary: true")).toBeVisible();
  await expect(details.getByText("logprob evidence")).toBeVisible();
  await expect(details.getByText("[-0.1,-0.2]")).toBeVisible();
  await expect(
    details.getByText("该样本没有生成 completion。"),
  ).toBeVisible();
  const firstSample = details.getByRole("article", {
    name: "Doc 0 details",
  });
  await expect(firstSample.getByText('"exact_match": 1')).toBeVisible();
  await expect(firstSample.getByText("[10,11]", { exact: true })).toBeVisible();
  await expect(details.getByText("latency", { exact: false })).toHaveCount(0);
});

test("shows lm-eval provenance without a prompt boundary", async ({ page }) => {
  await serveApi(page);
  await page.goto("/?page=dashboard");
  await page.getByRole("button", { name: /0.5 exact_match/ }).click();

  const details = page.getByRole("region", { name: "评估详情" });
  await expect(details.getByText("evaluator: lm-eval")).toBeVisible();
  await expect(details.getByText("prompt template: none")).toBeVisible();
  await details.getByText("lm-eval native response").click();
  await expect(
    details.locator("details").filter({ hasText: "lm-eval native response" }),
  ).toContainText('"native answer"');
});

test("loads every paginated evaluation before rendering the matrix", async ({
  page,
}) => {
  const rows = [
    evaluation("small.pth", "fp16"),
    evaluation("large.pth", "fp32io16"),
  ];
  await page.route("**/api/evaluations?*", (route) => {
    const url = new URL(route.request().url());
    const offset = Number(url.searchParams.get("offset"));
    if (offset === 1) {
      expect(url.searchParams.get("completed_before")).toBe(
        "2026-07-25T13:00:00Z",
      );
    }
    return route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        evaluations: [rows[offset]],
        generated_at: "2026-07-25T13:00:00Z",
        total: rows.length,
        offset,
        limit: 5000,
        next_offset: offset + 1 < rows.length ? offset + 1 : null,
      }),
    });
  });
  await page.goto("/?page=dashboard");
  await expect(page.getByRole("columnheader", { name: "small.pth" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "large.pth" })).toBeVisible();
});

test("dashboard does not mix an older campaign into the latest matrix", async ({
  page,
}) => {
  const latest = evaluation("small.pth", "fp16");
  const olderBase = evaluation("large.pth", "fp16");
  const obsoleteSha = "c".repeat(64);
  const older = {
    ...olderBase,
    evaluation_id: "obsolete-evaluation",
    campaign_id: "campaign-old",
    completed_at: "2026-07-24T13:00:00Z",
    task_identity: `${obsoleteSha}:fp16:${olderBase.task.task_name}`,
    task: {
      ...olderBase.task,
      identity: `${obsoleteSha}:fp16:${olderBase.task.task_name}`,
      weight_sha256: obsoleteSha,
      weight_display_name: "obsolete.pth",
    },
    model: {
      ...olderBase.model,
      weight_sha256: obsoleteSha,
      weight_display_name: "obsolete.pth",
    },
  };
  await page.route("**/api/evaluations?limit=5000&offset=0", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        evaluations: [latest, older],
        generated_at: "2026-07-25T13:00:00Z",
        total: 2,
        offset: 0,
        limit: 5000,
        next_offset: null,
      }),
    }),
  );
  await page.goto("/?page=dashboard");
  await expect(page.getByRole("columnheader", { name: "small.pth" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "obsolete.pth" })).toHaveCount(
    0,
  );
});

test("handles empty complete datasets and API failures", async ({ page }) => {
  await page.route("**/api/evaluations?limit=5000&offset=0", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        evaluations: [],
        generated_at: "2026-07-25T13:00:00Z",
        total: 0,
        offset: 0,
        limit: 5000,
        next_offset: null,
      }),
    }),
  );
  await page.goto("/?page=dashboard");
  await expect(page.getByText("尚无已完成的 LightEval campaign。")).toBeVisible();

  await page.unrouteAll();
  await page.route("**/api/evaluations?limit=5000&offset=0", (route) =>
    route.fulfill({ status: 503, body: "scoreboard unavailable" }),
  );
  await page.goto("/?page=dashboard");
  await expect(page.getByText(/加载失败：503: scoreboard unavailable/)).toBeVisible();
});
