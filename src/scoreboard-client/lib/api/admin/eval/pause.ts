import { postJsonAuth } from "../../../http";
import type { AdminEvalPauseResponse } from "../../../dtos/api/admin/eval/pause";

export function adminPause(): Promise<AdminEvalPauseResponse> {
  return postJsonAuth<AdminEvalPauseResponse>("/api/admin/eval/pause");
}
