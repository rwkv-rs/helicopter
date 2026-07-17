import { postJson } from "../http";
import type { RefreshResponse } from "../dtos/api/refresh";
import type { ScoreScope } from "../score_scope";

export function refresh(scope: ScoreScope): Promise<RefreshResponse> {
  return postJson<RefreshResponse>(`/api/refresh?scope=${scope}`);
}
