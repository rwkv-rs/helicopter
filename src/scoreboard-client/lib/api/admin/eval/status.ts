import { getJsonAuth } from "../../../http";
import type { AdminEvalStatusResponse } from "../../../dtos/api/admin/eval/status";

export function adminStatus(): Promise<AdminEvalStatusResponse> {
  return getJsonAuth<AdminEvalStatusResponse>("/api/admin/eval/status");
}
