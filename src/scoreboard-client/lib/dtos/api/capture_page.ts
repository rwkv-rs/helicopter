export interface CapturePageRequest {
  url: string;
  width?: number;
  height?: number;
}

export interface CapturePageResponse {
  path: string;
  url: string;
  width: number;
  height: number;
  page_height: number | null;
  full_page: boolean;
}
