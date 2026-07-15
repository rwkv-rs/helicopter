import { postJson } from "../http";
import type { RefreshResponse } from "../dtos/api/refresh";

export function refresh(): Promise<RefreshResponse> {
  return postJson<RefreshResponse>("/api/refresh");
}
