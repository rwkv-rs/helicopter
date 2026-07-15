import { getJson } from "../http";
import type { EvalRecordsResponse } from "../dtos/api/eval_records";

export function evalRecords(
  taskId: number,
  onlyWrong: boolean,
  limit: number,
  offset: number,
): Promise<EvalRecordsResponse> {
  const params = new URLSearchParams({
    task_id: String(taskId),
    only_wrong: String(onlyWrong),
    limit: String(limit),
    offset: String(offset),
  });
  return getJson<EvalRecordsResponse>(`/api/eval-records?${params.toString()}`);
}
