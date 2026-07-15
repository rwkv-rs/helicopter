import { getJsonAuth } from "../../http";
import type { AdminBackpressureResponse } from "../../dtos/api/admin/backpressure";

export function adminBackpressure(inferBaseUrl?: string): Promise<AdminBackpressureResponse> {
  const params = new URLSearchParams();
  if (inferBaseUrl) params.set("infer_base_url", inferBaseUrl);
  const query = params.toString();
  return getJsonAuth<AdminBackpressureResponse>(`/api/admin/backpressure${query ? `?${query}` : ""}`);
}
