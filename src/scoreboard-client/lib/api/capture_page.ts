import { postJson } from "../http";
import type { CapturePageRequest, CapturePageResponse } from "../dtos/api/capture_page";

export function capturePage(payload: CapturePageRequest): Promise<CapturePageResponse> {
  return postJson<CapturePageResponse>("/api/capture-page", payload);
}
