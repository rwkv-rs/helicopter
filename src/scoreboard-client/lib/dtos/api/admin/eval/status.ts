export interface AdminEvalStatusResponse {
  status: string;
  desired_state: string | null;
  run_id: string | null;
  error: string | null;
  started_at_unix_ms: number | null;
  updated_at_unix_ms: number | null;
  finished_at_unix_ms: number | null;
  pending_jobs: number;
  running_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  tasks_total: number;
  progress_percent: number;
  queue_head: string[];
  active_jobs: string[];
  available_gpus: string[];
  request: Record<string, unknown> | null;
}
