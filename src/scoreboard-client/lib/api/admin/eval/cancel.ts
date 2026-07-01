import { postJsonAuth } from "../../../http";
import type { AdminEvalCancelResponse } from "../../../dtos/api/admin/eval/cancel";

export function adminCancel(): Promise<AdminEvalCancelResponse> {
  return postJsonAuth<AdminEvalCancelResponse>("/api/admin/eval/cancel");
}
