import { getJsonAuth } from "../../http";
import type { AdminHealthResponse } from "../../dtos/api/admin/health";

export function adminHealth(): Promise<AdminHealthResponse> {
  return getJsonAuth<AdminHealthResponse>("/api/admin/health");
}
