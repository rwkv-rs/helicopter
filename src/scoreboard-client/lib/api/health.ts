import { getJson } from "../http";
import type { HealthResponse } from "../dtos/api/health";

export function health(): Promise<HealthResponse> {
  return getJson<HealthResponse>("/api/health");
}
