from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from lighteval_runner.application import (
    retry_scoreboard_publication,
    run_evaluation,
)
from lighteval_runner.contracts import EvaluationRequest


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "src/scoreboard-server"


class _ProviderHandler(BaseHTTPRequestHandler):
    def log_message(self, _format, *_args):
        pass

    def do_GET(self):
        if self.path != "/v1/helicopter/attestation":
            self._json(404, {})
            return
        self._json(
            200,
            {
                "model": {
                    "served_name": "rwkv-test",
                    "checkpoint_sha256": "b" * 64,
                    "tokenizer_revision": "tok-v1",
                    "chat_template_revision": "chat-v1",
                },
                "provider": {
                    "server_revision": "server-v1",
                    "wkv_mode": "fp32io16",
                    "precision": "fp16-io-fp32-state",
                    "gemm_policy": "fp32-accumulation",
                    "launch_contract": "launch-v1",
                },
                "capabilities": [
                    "openai-chat",
                    "output-token-ids",
                    "terminal-reason",
                    "prompt-evidence",
                ],
            },
        )

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length))
        if self.path == "/tokenize":
            self._json(200, {"count": 2, "max_model_len": 1024, "tokens": [1, 2]})
            return
        if self.path == "/v1/chat/completions":
            self._json(
                200,
                {
                    "id": "provider-request-1",
                    "choices": [
                        {
                            "message": {
                                "content": '{"name":"search","arguments":{"query":"rwkv"}}'
                            },
                            "finish_reason": "stop",
                            "stop_reason": 0,
                            "token_ids": [7, 0],
                        }
                    ],
                    "prompt_token_ids": [1, 2],
                    "prompt_text": "signed proxy prompt",
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 2,
                        "total_tokens": 4,
                    },
                },
            )
            return
        self._json(404, {})

    def _json(self, status: int, payload: dict):
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@contextmanager
def _provider_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProviderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@contextmanager
def _scoreboard_server(tmp_path: Path):
    port = _unused_port()
    env = {
        **os.environ,
        "SCOREBOARD_DATABASE_URL": f"sqlite:///{tmp_path / 'scoreboard.db'}",
        "SCOREBOARD_CORS_ORIGINS": "https://scoreboard.example",
        "SCOREBOARD_AUTH_TOKENS": json.dumps(
            {
                "publisher-token": {"subject": "e2e", "roles": ["publisher"]},
                "reader-token": {"subject": "reader", "roles": ["evidence_reader"]},
                "admin-token": {"subject": "admin", "roles": ["admin"]},
            }
        ),
    }
    process = subprocess.Popen(
        [
            str(SERVER / ".venv/bin/python"),
            "-m",
            "uvicorn",
            "scoreboard_server.application:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=SERVER,
        env=env,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        for _attempt in range(100):
            if process.poll() is not None:
                raise RuntimeError("scoreboard exited before becoming ready")
            try:
                if (
                    httpx.get(f"{base_url}/api/v1/health", timeout=0.2).status_code
                    == 200
                ):
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:
            raise RuntimeError("scoreboard did not become ready")
        response = httpx.post(
            f"{base_url}/api/v1/admin/migrations",
            headers={"Authorization": "Bearer admin-token"},
        )
        response.raise_for_status()
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_evaluator_artifact_http_publish_and_idempotent_retry(tmp_path: Path) -> None:
    with (
        _provider_server() as provider_port,
        _scoreboard_server(tmp_path) as scoreboard_url,
    ):
        request = EvaluationRequest(
            model="rwkv-test",
            task="helicopter-proxy/function-calling/exact-json@1",
            output_root=tmp_path / "runs",
            snapshot_path=None,
            snapshot_manifest_path=None,
            snapshot_sha256=None,
            endpoint_url=f"http://127.0.0.1:{provider_port}/v1",
            checkpoint_sha256="b" * 64,
            tokenizer_revision="tok-v1",
            chat_template_revision="chat-v1",
            expected_server_revision="server-v1",
            wkv_mode="fp32io16",
            precision="fp16-io-fp32-state",
            gemm_policy="fp32-accumulation",
            launch_contract="launch-v1",
            allow_non_comparable=True,
            publish_to_scoreboard=True,
            scoreboard_url=scoreboard_url,
            scoreboard_token="publisher-token",
        )
        outcome = run_evaluation(request)
        assert outcome.is_success
        assert outcome.publication_status == "published"
        retry = retry_scoreboard_publication(
            manifest_path=outcome.manifest_path,
            scoreboard_url=scoreboard_url,
            scoreboard_token="publisher-token",
        )
        assert retry.is_success
        detail = httpx.get(
            f"{scoreboard_url}/api/v1/runs/{outcome.run_id}",
            headers={"Authorization": "Bearer reader-token"},
        )
        detail.raise_for_status()
        assert detail.json()["status"] == "completed"
        receipt = json.loads(
            (outcome.manifest_path.parent / "publication.json").read_text()
        )
        assert [attempt["status"] for attempt in receipt["attempts"]] == [
            "published",
            "published",
        ]
