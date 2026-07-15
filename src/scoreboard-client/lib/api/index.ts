import { adminBackpressure } from "./admin/backpressure";
import { adminHealth } from "./admin/health";
import { adminCancel } from "./admin/eval/cancel";
import { adminDraft } from "./admin/eval/draft";
import { adminOptions } from "./admin/eval/options";
import { adminPause } from "./admin/eval/pause";
import { adminResume } from "./admin/eval/resume";
import { adminStart } from "./admin/eval/start";
import { adminStatus } from "./admin/eval/status";
import { capturePage } from "./capture_page";
import { evalContext } from "./eval_context";
import { evalRecords } from "./eval_records";
import { health } from "./health";
import { leaderboard } from "./leaderboard";
import { meta } from "./meta";
import { refresh } from "./refresh";
import { scoreHistory } from "./score_history";
import { scoreHistoryDetail } from "./score_history/detail";
import { scoreHistoryOptions } from "./score_history/options";

export const api = {
  health,
  meta,
  refresh,
  capturePage,
  leaderboard,
  evalRecords,
  evalContext,
  scoreHistoryOptions,
  scoreHistory,
  scoreHistoryDetail,
  adminHealth,
  adminOptions,
  adminDraft,
  adminStatus,
  adminStart,
  adminPause,
  adminResume,
  adminCancel,
  adminBackpressure,
};
