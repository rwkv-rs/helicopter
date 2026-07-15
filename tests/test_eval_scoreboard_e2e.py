from __future__ import annotations

import json
import getpass
import os
import socket
import subprocess
import threading
import time
import uuid
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
_DATABASE_SCRIPT = '''
import asyncio
import os
import sys

import asyncpg


async def main():
    kwargs = {
        "host": os.environ.get("PGHOST", "/var/run/postgresql"),
        "port": int(os.environ.get("PGPORT", "5432")),
        "user": os.environ["PGUSER"],
        "database": os.environ.get("PGDATABASE", "postgres"),
    }
    password = os.environ.get("PGPASSWORD")
    if password:
        kwargs["password"] = password
    connection = await asyncpg.connect(**kwargs)
    database = sys.argv[2]
    try:
        if sys.argv[1] == "create":
            await connection.execute(f'CREATE DATABASE "{database}"')
        else:
            await connection.execute(
                """
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = $1 AND pid <> pg_backend_pid()
                """,
                database,
            )
            await connection.execute(f'DROP DATABASE IF EXISTS "{database}"')
    finally:
        await connection.close()


asyncio.run(main())
'''


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
def _temporary_scoreboard_database():
    database = f"helicopter_eval_e2e_{uuid.uuid4().hex[:12]}"
    maintenance_env = {
        **os.environ,
        "PGHOST": os.environ.get("PGHOST", "/var/run/postgresql"),
        "PGUSER": os.environ.get("PGUSER", getpass.getuser()),
        "PGDATABASE": os.environ.get("PGDATABASE", "postgres"),
    }
    command = [str(SERVER / ".venv/bin/python"), "-c", _DATABASE_SCRIPT]
    subprocess.run([*command, "create", database], env=maintenance_env, check=True)
    try:
        yield {
            "SCOREBOARD_DB_HOST": maintenance_env["PGHOST"],
            "SCOREBOARD_DB_PORT": maintenance_env.get("PGPORT", "5432"),
            "SCOREBOARD_DB_USER": maintenance_env["PGUSER"],
            "SCOREBOARD_DB_PASSWORD": maintenance_env.get("PGPASSWORD", ""),
            "SCOREBOARD_DB_NAME": database,
        }
    finally:
        subprocess.run([*command, "drop", database], env=maintenance_env, check=True)


@contextmanager
def _scoreboard_server():
    port = _unused_port()
    with _temporary_scoreboard_database() as database_env:
        env = {
            **os.environ,
            **database_env,
            "SCOREBOARD_AUTH_TOKENS": json.dumps(
                {
                    "publisher-token": {"subject": "e2e", "roles": ["publisher"]},
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
                        httpx.get(f"{base_url}/api/health", timeout=0.2).status_code
                        == 200
                    ):
                        break
                except httpx.HTTPError:
                    time.sleep(0.05)
            else:
                raise RuntimeError("scoreboard did not become ready")
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
        _scoreboard_server() as scoreboard_url,
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
            product_revision="d" * 40,
            allow_non_comparable=True,
            publish_to_scoreboard=True,
            scoreboard_url=scoreboard_url,
            scoreboard_token="publisher-token",
        )
        outcome = run_evaluation(request)
        assert outcome.is_success
        assert outcome.publication_status == "published"
        assert outcome.publication_task_id is not None
        retry = retry_scoreboard_publication(
            manifest_path=outcome.manifest_path,
            scoreboard_url=scoreboard_url,
            scoreboard_token="publisher-token",
        )
        assert retry.is_success
        records = httpx.get(
            f"{scoreboard_url}/api/eval-records",
            params={"task_id": outcome.publication_task_id},
        )
        records.raise_for_status()
        assert json.loads(records.json()["records"][0]["answer"]) == {
            "name": "search",
            "arguments": {"query": "rwkv"},
        }
        receipt = json.loads(
            (outcome.manifest_path.parent / "publication.json").read_text()
        )
        assert [attempt["status"] for attempt in receipt["attempts"]] == [
            "published",
            "published",
        ]
        assert {attempt["task_id"] for attempt in receipt["attempts"]} == {
            outcome.publication_task_id
        }
