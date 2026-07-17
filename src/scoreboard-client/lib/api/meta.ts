import { getJson } from "../http";
import type { MetaResponse } from "../dtos/api/meta";
import type { ScoreScope } from "../score_scope";

export function meta(scope: ScoreScope): Promise<MetaResponse> {
  return getJson<MetaResponse>(`/api/meta?scope=${scope}`);
}
