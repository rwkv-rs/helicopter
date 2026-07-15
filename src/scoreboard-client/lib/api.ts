import createClient from "openapi-fetch";

import type { paths } from "./generated/schema";


const baseUrl = process.env.SCOREBOARD_API_BASE_URL || process.env.NEXT_PUBLIC_SCOREBOARD_API_BASE_URL || "http://127.0.0.1:7860";

export const scoreboard = createClient<paths>({ baseUrl });
