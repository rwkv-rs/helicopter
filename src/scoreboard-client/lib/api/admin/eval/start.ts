import { postJsonAuth } from "../../../http";
import type { AdminEvalStartRequest, AdminEvalStartResponse } from "../../../dtos/api/admin/eval/start";

export function adminStart(payload: AdminEvalStartRequest): Promise<AdminEvalStartResponse> {
  return postJsonAuth<AdminEvalStartResponse>("/api/admin/eval/start", payload);
}
