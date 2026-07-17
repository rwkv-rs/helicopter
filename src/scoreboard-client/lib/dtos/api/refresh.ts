import type { ScoreScope } from "../../score_scope";

export interface RefreshResponse {
  scope: ScoreScope;
  entry_count: number;
  errors: string[];
}
