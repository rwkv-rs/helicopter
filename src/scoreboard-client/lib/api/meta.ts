import { getJson } from "../http";
import type { MetaResponse } from "../dtos/api/meta";

export function meta(): Promise<MetaResponse> {
  return getJson<MetaResponse>("/api/meta");
}
