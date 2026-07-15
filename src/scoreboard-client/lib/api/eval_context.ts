import { getJson } from "../http";
import type { EvalContextResponse } from "../dtos/api/eval_context";

export function evalContext(
  taskId: number,
  sampleIndex: number,
  repeatIndex: number,
  passIndex: number,
): Promise<EvalContextResponse> {
  const params = new URLSearchParams({
    task_id: String(taskId),
    sample_index: String(sampleIndex),
    repeat_index: String(repeatIndex),
    pass_index: String(passIndex),
  });
  return getJson<EvalContextResponse>(`/api/eval-context?${params.toString()}`);
}
