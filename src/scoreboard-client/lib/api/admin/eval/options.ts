import { getJsonAuth } from "../../../http";
import type { AdminEvalOptionsResponse } from "../../../dtos/api/admin/eval/options";

export function adminOptions(): Promise<AdminEvalOptionsResponse> {
  return getJsonAuth<AdminEvalOptionsResponse>("/api/admin/eval/options");
}
