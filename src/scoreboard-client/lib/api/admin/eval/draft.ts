import { getJsonAuth } from "../../../http";
import type { AdminEvalDraftResponse } from "../../../dtos/api/admin/eval/draft";

export function adminDraft(): Promise<AdminEvalDraftResponse> {
  return getJsonAuth<AdminEvalDraftResponse>("/api/admin/eval/draft");
}
