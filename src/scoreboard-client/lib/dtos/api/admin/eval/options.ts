export interface AdminEvalOptionsResponse {
  jobs: { name: string; domain: string }[];
  domains: string[];
  model_select: string[];
  worker_profile: string[];
  protocol: string[];
  run_mode: string[];
}
