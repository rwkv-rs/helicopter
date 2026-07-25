from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest

from helicopter_eval.config import EvaluationEnvironment
from helicopter_eval.http_client import (
    ScoreboardClient,
    ScoreboardConflict,
    ScoreboardError,
)


def _client(tmp_path: Path) -> ScoreboardClient:
    return ScoreboardClient(
        EvaluationEnvironment(
            weight_root=tmp_path,
            scoreboard_url="https://scoreboard.example.test",
            scoreboard_token="private-token",
            staging_root=tmp_path / "staging",
        )
    )


@pytest.mark.parametrize("invalid", [float("nan"), object()])
def test_request_wraps_non_json_publication_payload(
    tmp_path: Path,
    invalid: object,
) -> None:
    with pytest.raises(ScoreboardError, match="Scoreboard request failed"):
        _client(tmp_path)._request(
            "POST",
            "/api/v1/evaluation-campaigns",
            payload={"invalid": invalid},
        )


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (400, ScoreboardError),
        (409, ScoreboardConflict),
        (500, ScoreboardError),
    ],
)
def test_http_error_never_exposes_backend_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    error_type: type[ScoreboardError],
) -> None:
    client = _client(tmp_path)
    transformed_secret = "private\\u002dtoken"

    def fail(*_args, **_kwargs):
        raise HTTPError(
            client.base_url,
            status,
            "failed",
            {},
            BytesIO(f'{{"detail":"credential={transformed_secret}"}}'.encode()),
        )

    monkeypatch.setattr(client._opener, "open", fail)
    with pytest.raises(error_type) as raised:
        client._request("GET", "/api/v1/evaluation-publication-preflight")

    assert "credential" not in str(raised.value)
    assert transformed_secret not in str(raised.value)
