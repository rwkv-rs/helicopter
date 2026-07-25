from __future__ import annotations

import gzip
import json
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import EvaluationEnvironment


MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class ScoreboardError(RuntimeError):
    pass


class ScoreboardConflict(ScoreboardError):
    pass


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, new_url):
        return None


class ScoreboardClient:
    def __init__(self, environment: EvaluationEnvironment) -> None:
        self.base_url = environment.scoreboard_url.rstrip("/")
        self._token = environment.scoreboard_token
        self._opener = build_opener(_RejectRedirects())

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        try:
            body = None
            headers = {
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/json",
            }
            if payload is not None:
                raw = json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                body = gzip.compress(raw)
                headers.update(
                    {
                        "Content-Type": "application/json",
                        "Content-Encoding": "gzip",
                    }
                )
            if idempotency_key is not None:
                headers["Idempotency-Key"] = idempotency_key
            request = Request(
                f"{self.base_url}{path}",
                data=body,
                method=method,
                headers=headers,
            )
            with self._opener.open(request, timeout=60) as response:
                content_type = response.headers.get_content_type()
                if content_type != "application/json":
                    raise ScoreboardError(
                        "Scoreboard response Content-Type must be application/json"
                    )
                raw_response = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw_response) > MAX_RESPONSE_BYTES:
                    raise ScoreboardError("Scoreboard response exceeds size limit")
                decoded = json.loads(
                    raw_response,
                    parse_constant=_reject_json_constant,
                )
        except HTTPError as error:
            raw_detail = error.read(MAX_RESPONSE_BYTES + 1)
            if len(raw_detail) > MAX_RESPONSE_BYTES:
                raise ScoreboardError(
                    f"Scoreboard HTTP {error.code} response exceeds size limit"
                ) from error
            if error.code == 409:
                raise ScoreboardConflict(
                    f"Scoreboard HTTP {error.code} conflict"
                ) from error
            raise ScoreboardError(f"Scoreboard HTTP {error.code}") from error
        except ScoreboardError:
            raise
        except (
            OSError,
            TimeoutError,
            TypeError,
            UnicodeDecodeError,
            ValueError,
        ) as error:
            raise ScoreboardError(
                f"Scoreboard request failed: "
                f"{type(error).__module__}.{type(error).__qualname__}"
            ) from error
        if not isinstance(decoded, dict):
            raise ScoreboardError("Scoreboard response must be a JSON object")
        return decoded

    def create_campaign(
        self, payload: dict[str, Any], resume_key: str
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/evaluation-campaigns",
            payload=payload,
            idempotency_key=f"campaign:{resume_key}",
        )

    def preflight(self) -> dict[str, Any]:
        response = self._request(
            "GET",
            "/api/v1/evaluation-publication-preflight",
        )
        if (
            response.get("status") != "ready"
            or response.get("schema_version") != "lighteval-campaign-v1"
            or response.get("lighteval_version") != "0.13.0"
            or not isinstance(response.get("publisher_principal"), str)
            or not response["publisher_principal"]
        ):
            raise ScoreboardError("Scoreboard publication preflight is incompatible")
        return response

    def campaign_status(self, campaign_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/v1/evaluation-campaigns/{quote(campaign_id, safe='')}",
        )

    def publish_task(
        self,
        *,
        campaign_id: str,
        task_identity: str,
        payload: dict[str, Any],
        digest: str,
    ) -> dict[str, Any]:
        return self._request(
            "PUT",
            (
                f"/api/v1/evaluation-campaigns/{quote(campaign_id, safe='')}"
                f"/tasks/{quote(task_identity, safe='')}"
            ),
            payload=payload,
            idempotency_key=f"publish:{digest}",
        )

    def finalize(self, campaign_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            (f"/api/v1/evaluation-campaigns/{quote(campaign_id, safe='')}/finalize"),
            idempotency_key=f"finalize:{campaign_id}",
        )
