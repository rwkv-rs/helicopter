export interface BackpressureModel {
  model: string;
  model_slug: string;
  status: string;
  route_count: number;
  ok_route_count: number;
  pending_queue: number;
  max_batch_size: number | null;
  failed_batches: number;
  last_total_tok_s: number | null;
  error: string | null;
}

export interface AdminBackpressureResponse {
  infer_base_url: string;
  available_gpus: string[];
  models: BackpressureModel[];
  error: string | null;
}
