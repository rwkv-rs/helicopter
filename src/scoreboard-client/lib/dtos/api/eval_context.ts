export interface StopToken {
  id: number;
  token: string;
}

export interface EvalContextResponse {
  view: "text" | "structured";
  raw_text: string;
  context: Record<string, unknown> | null;
  stop_tokens: Record<string, StopToken[]>;
  errors: string[];
}
