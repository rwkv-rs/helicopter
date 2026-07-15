export interface EvalRecord {
  sample_index: number;
  repeat_index: number;
  pass_index: number;
  is_passed: boolean;
  answer: string;
  ref_answer: string;
  fail_reason: string;
  context_preview?: string;
}

export interface EvalRecordsResponse {
  task_id: number;
  records: EvalRecord[];
  offset: number;
  limit: number;
  next_offset: number;
  has_more: boolean;
}
