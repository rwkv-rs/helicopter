"use client";

import { useEffect, useMemo, useState } from "react";

import type {
  AnswerOutcome,
  PromptTemplate,
  SampleDetail,
  SamplePage,
} from "../lib/evaluation_types";
import { useEvaluations } from "./EvaluationProvider";

const PAGE_SIZE = 25;
const OUTCOMES: Array<{ value: "" | AnswerOutcome; label: string }> = [
  { value: "", label: "全部结果" },
  { value: "correct", label: "正确" },
  { value: "incorrect", label: "错误" },
  { value: "unanswered", label: "未作答" },
  { value: "undetermined", label: "无法判定" },
];

function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function tokenRows(value: unknown): number[][] {
  return Array.isArray(value)
    ? value.filter(
        (row): row is number[] =>
          Array.isArray(row) && row.every((token) => Number.isInteger(token)),
      )
    : [];
}

function reasoningRows(value: unknown): Array<string | null> {
  return Array.isArray(value)
    ? value.map((item) => (typeof item === "string" ? item : null))
    : [];
}

function reference(doc: Record<string, unknown>): string {
  const choices = Array.isArray(doc.choices) ? doc.choices : [];
  const gold = Array.isArray(doc.gold_index) ? doc.gold_index : [doc.gold_index];
  const selected = gold
    .filter((index): index is number => Number.isInteger(index))
    .map((index) => choices[index])
    .filter((value) => value !== undefined);
  if (selected.length) return selected.map(String).join("\n");
  for (const key of ["reference", "target", "gold", "answer"]) {
    const value = doc[key];
    if (value !== undefined && value !== null) {
      return typeof value === "string" ? value : JSON.stringify(value);
    }
  }
  return "null";
}

const PROMPT_STOPS: Record<PromptTemplate, string> = {
  bot: "✿",
  assistant: "\nUser:",
  function_calling: "\n### User",
};

function completionRows(
  sample: SampleDetail,
  limit: number,
  turnBoundary: string,
) {
  const raw = strings(sample.model_response.text);
  const processed = strings(sample.model_response.text_post_processed);
  const reasonings = reasoningRows(sample.model_response.reasonings);
  const tokens = tokenRows(sample.model_response.output_tokens);
  return raw.map((text, index) => {
    const split = text.split("</think>");
    return {
      index,
      raw: text,
      reasoning:
        reasonings[index] ??
        (split.length === 2
          ? split[0].replace(/^<think>/, "").replace(/^>/, "")
          : null),
      answer: processed[index] ?? (split.length === 2 ? split[1] : null),
      tokens: tokens[index] ?? [],
      truncated: (tokens[index]?.length ?? 0) >= limit,
      boundaryViolation: text.includes(turnBoundary),
    };
  });
}

function SampleCard({
  sample,
  limit,
  turnBoundary,
}: {
  sample: SampleDetail;
  limit: number;
  turnBoundary: string;
}) {
  const completions = completionRows(sample, limit, turnBoundary);
  const logprobs = sample.model_response.logprobs;
  const argmax = sample.model_response.argmax_logits_eq_gold;
  const hasLogprobEvidence =
    (Array.isArray(logprobs) && logprobs.length > 0) ||
    (Array.isArray(argmax) && argmax.length > 0);
  return (
    <article
      aria-label={`Doc ${sample.document_index} details`}
      className="sample-card"
    >
      <header>
        <strong>sample {sample.sample_index}</strong>
        <span>Doc {sample.document_index} · {sample.outcome}</span>
      </header>
      <dl>
        <dt>reference</dt>
        <dd><pre>{reference(sample.doc)}</pre></dd>
        <dt>Doc</dt>
        <dd><pre>{JSON.stringify(sample.doc, null, 2)}</pre></dd>
        <dt>sample metric</dt>
        <dd><pre>{JSON.stringify(sample.metric, null, 2)}</pre></dd>
        <dt>input tokens</dt>
        <dd><pre>{JSON.stringify(sample.model_response.input_tokens ?? null)}</pre></dd>
      </dl>
      {completions.map((completion) => (
        <section className="completion" key={completion.index}>
          <h4>completion {completion.index + 1}</h4>
          <div className="completion-flags">
            <span>tokens: {completion.tokens.length}</span>
            <span>truncated: {String(completion.truncated)}</span>
            <span>turn boundary: {String(completion.boundaryViolation)}</span>
          </div>
          <dl>
            <dt>raw</dt>
            <dd><pre>{completion.raw}</pre></dd>
            <dt>reasoning</dt>
            <dd><pre>{completion.reasoning ?? "null"}</pre></dd>
            <dt>answer</dt>
            <dd><pre>{completion.answer ?? "null"}</pre></dd>
            <dt>output tokens</dt>
            <dd><pre>{JSON.stringify(completion.tokens)}</pre></dd>
          </dl>
        </section>
      ))}
      {hasLogprobEvidence ? (
        <section className="completion logprob-evidence">
          <h4>logprob evidence</h4>
          <dl>
            <dt>output tokens</dt>
            <dd>
              <pre>
                {JSON.stringify(sample.model_response.output_tokens ?? null)}
              </pre>
            </dd>
            <dt>token logprobs</dt>
            <dd><pre>{JSON.stringify(logprobs ?? null)}</pre></dd>
            <dt>argmax equals gold</dt>
            <dd><pre>{JSON.stringify(argmax ?? null)}</pre></dd>
          </dl>
        </section>
      ) : null}
      {!completions.length ? <p>该样本没有生成 completion。</p> : null}
    </article>
  );
}

export function EvaluationDetails() {
  const { selected, select, dataSource } = useEvaluations();
  const [outcome, setOutcome] = useState<"" | AnswerOutcome>("");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<SamplePage | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPage(null);
    setError(null);
    if (!selected) return;
    let cancelled = false;
    dataSource
      .loadSamples(selected.evaluation_id, offset, PAGE_SIZE, outcome || undefined)
      .then((payload) => {
        if (!cancelled) setPage(payload);
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [dataSource, offset, outcome, selected]);

  useEffect(() => {
    setOffset(0);
    setOutcome("");
  }, [selected?.evaluation_id]);

  const outputLimit = useMemo(() => {
    const value = selected?.sampling_config.max_new_tokens;
    return typeof value === "number" && value > 0 ? value : 8192;
  }, [selected]);

  if (!selected) return null;
  return (
    <section aria-label="评估详情" className="card details">
      <header className="section-head">
        <div>
          <h2>{selected.task.task_name}</h2>
          <p>
            {selected.task.weight_display_name} · {selected.task.wkv_mode} ·{" "}
            {selected.primary_metric}
          </p>
        </div>
        <button type="button" onClick={() => select(null)}>关闭</button>
      </header>
      <div className="metadata-grid">
        <span>evaluation: {selected.evaluation_id}</span>
        <span>campaign: {selected.campaign_id}</span>
        <span>selector: {selected.task.selector}</span>
        <span>module: {selected.task.module_family}</span>
        <span>prompt template: {selected.model.prompt_template}</span>
        <span>
          tags:{" "}
          {selected.task.upstream_tags.length
            ? selected.task.upstream_tags.join(", ")
            : "—"}
        </span>
        <span>
          questions: {String(selected.task_config.effective_num_docs)} /{" "}
          {String(selected.task_config.original_num_docs)}
        </span>
        <span>
          skipped multi-select:{" "}
          {String(selected.task_config.skipped_multiselect_docs)}
        </span>
        <span>samples: {selected.diagnostics.samples}</span>
        <span>completions: {selected.diagnostics.completions}</span>
        <span>truncation: {selected.diagnostics.truncation_rate}</span>
        <span>turn boundary: {selected.diagnostics.turn_boundary_violation_rate}</span>
        <span>publisher audit: {selected.provenance.publisher_principal}</span>
      </div>
      <dl>
        <dt>native aggregates</dt>
        <dd><pre>{JSON.stringify(selected.aggregates, null, 2)}</pre></dd>
        <dt>sampling config</dt>
        <dd><pre>{JSON.stringify(selected.sampling_config, null, 2)}</pre></dd>
        <dt>model execution</dt>
        <dd><pre>{JSON.stringify(selected.model, null, 2)}</pre></dd>
        <dt>campaign provenance</dt>
        <dd><pre>{JSON.stringify(selected.provenance, null, 2)}</pre></dd>
      </dl>
      <label className="outcome-filter">
        outcome
        <select
          value={outcome}
          onChange={(event) => {
            setOffset(0);
            setOutcome(event.target.value as "" | AnswerOutcome);
          }}
        >
          {OUTCOMES.map((item) => (
            <option key={item.value} value={item.value}>{item.label}</option>
          ))}
        </select>
      </label>
      {error ? <p className="error-bar">加载失败：{error}</p> : null}
      {!page && !error ? <p>正在加载样本…</p> : null}
      {page?.items.map((sample) => (
        <SampleCard
          key={sample.id}
          limit={outputLimit}
          sample={sample}
          turnBoundary={PROMPT_STOPS[selected.model.prompt_template]}
        />
      ))}
      {page ? (
        <footer className="pager">
          <button
            disabled={page.offset === 0}
            onClick={() => setOffset(Math.max(0, page.offset - PAGE_SIZE))}
            type="button"
          >
            上一页
          </button>
          <span>
            {page.total === 0 ? 0 : page.offset + 1}–
            {page.offset + page.items.length} / {page.total}
          </span>
          <button
            disabled={page.next_offset === null}
            onClick={() => setOffset(page.next_offset ?? page.offset)}
            type="button"
          >
            下一页
          </button>
        </footer>
      ) : null}
    </section>
  );
}
