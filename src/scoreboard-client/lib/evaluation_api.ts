import { getJson } from "./http";
import type {
  AnswerOutcome,
  EvaluationDataSource,
  EvaluationDataset,
  SamplePage,
} from "./evaluation_types";

export class ApiEvaluationDataSource implements EvaluationDataSource {
  async load(): Promise<EvaluationDataset> {
    const pageLimit = 5000;
    const evaluations: EvaluationDataset["evaluations"] = [];
    const evaluationIds = new Set<string>();
    let offset = 0;
    let generatedAt = "";
    let total: number | null = null;
    while (true) {
      const snapshot = generatedAt
        ? `&completed_before=${encodeURIComponent(generatedAt)}`
        : "";
      const page = await getJson<EvaluationDataset>(
        `/api/evaluations?limit=${pageLimit}&offset=${offset}${snapshot}`,
      );
      if (page.offset !== offset || page.limit !== pageLimit) {
        throw new Error("Scoreboard evaluation pagination is inconsistent");
      }
      if (!page.generated_at) {
        throw new Error("Scoreboard evaluation snapshot is missing");
      }
      if (generatedAt && page.generated_at !== generatedAt) {
        throw new Error("Scoreboard evaluation snapshot changed during pagination");
      }
      generatedAt ||= page.generated_at;
      if (!Number.isSafeInteger(page.total) || page.total < 0) {
        throw new Error("Scoreboard evaluation total is invalid");
      }
      if (total !== null && page.total !== total) {
        throw new Error("Scoreboard evaluation total changed during pagination");
      }
      const pageTotal: number = total ?? page.total;
      total = pageTotal;
      if (
        !Array.isArray(page.evaluations) ||
        page.evaluations.length > pageLimit ||
        offset + page.evaluations.length > pageTotal
      ) {
        throw new Error("Scoreboard evaluation page exceeds its declared bounds");
      }
      for (const evaluation of page.evaluations) {
        if (evaluationIds.has(evaluation.evaluation_id)) {
          throw new Error("Scoreboard evaluation pagination returned a duplicate");
        }
        evaluationIds.add(evaluation.evaluation_id);
        evaluations.push(evaluation);
      }
      if (page.next_offset === null) break;
      if (
        !Number.isSafeInteger(page.next_offset) ||
        page.next_offset <= offset ||
        page.next_offset > pageTotal ||
        page.next_offset !== offset + page.evaluations.length
      ) {
        throw new Error("Scoreboard evaluation pagination did not advance");
      }
      offset = page.next_offset;
    }
    if (total === null || evaluations.length !== total) {
      throw new Error("Scoreboard evaluation pagination is incomplete");
    }
    return {
      evaluations,
      generated_at: generatedAt,
      total: total ?? 0,
      offset: 0,
      limit: evaluations.length || 1,
      next_offset: null,
    };
  }

  loadSamples(
    evaluationId: string,
    offset: number,
    limit: number,
    outcome?: AnswerOutcome,
  ): Promise<SamplePage> {
    const query = new URLSearchParams({
      offset: String(offset),
      limit: String(limit),
    });
    if (outcome) query.set("outcome", outcome);
    return getJson<SamplePage>(
      `/api/evaluations/${encodeURIComponent(evaluationId)}/samples?${query}`,
    ).then((page) => {
      if (
        page.evaluation_id !== evaluationId ||
        page.offset !== offset ||
        page.limit !== limit ||
        !Number.isSafeInteger(page.total) ||
        page.total < 0
      ) {
        throw new Error("Scoreboard sample pagination is inconsistent");
      }
      if (
        page.next_offset !== null &&
        (!Number.isSafeInteger(page.next_offset) ||
          page.next_offset <= offset ||
          page.next_offset > page.total)
      ) {
        throw new Error("Scoreboard sample pagination did not advance");
      }
      if (page.items.length > limit || offset + page.items.length > page.total) {
        throw new Error("Scoreboard sample page exceeds its declared bounds");
      }
      const itemIds = new Set<string>();
      for (const item of page.items) {
        if (
          itemIds.has(item.id) ||
          !Number.isSafeInteger(item.sample_index) ||
          item.sample_index < 0 ||
          !Number.isSafeInteger(item.document_index) ||
          item.document_index < 0
        ) {
          throw new Error("Scoreboard sample page contains invalid items");
        }
        itemIds.add(item.id);
      }
      if (
        page.next_offset !== null &&
        page.next_offset !== offset + page.items.length
      ) {
        throw new Error("Scoreboard sample next offset is inconsistent");
      }
      if (
        page.next_offset === null &&
        offset + page.items.length !== page.total
      ) {
        throw new Error("Scoreboard sample pagination is incomplete");
      }
      return page;
    });
  }
}
