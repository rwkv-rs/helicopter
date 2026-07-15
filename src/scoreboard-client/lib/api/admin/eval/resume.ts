import { postJsonAuth } from "../../../http";
import type { AdminEvalResumeResponse } from "../../../dtos/api/admin/eval/resume";

export function adminResume(): Promise<AdminEvalResumeResponse> {
  return postJsonAuth<AdminEvalResumeResponse>("/api/admin/eval/resume");
}
