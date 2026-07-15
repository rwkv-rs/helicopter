from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from lighteval_runner.results.acceptance import verify_acceptance_runs


MODEL = "rwkv7-g1h-7.2b-eval"
BASE_URL = "http://127.0.0.1:8000/v1"
CHECKPOINT_SHA256 = "1fe61e5c4b9037ffd4723a11c4de146d99c26bcd89e00a61afa67ef653d215e8"
TOKENIZER_REVISION = (
    "sha256:e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89"
)
CHAT_TEMPLATE_REVISION = (
    "sha256:87a5c16f5f052451b2daa261baed47a08a4d15e7e0e3888761beda2a836cecfe"
)


@dataclass(frozen=True, slots=True)
class AcceptanceConfig:
    output: Path
    server_revision: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the signed remote LightEval acceptance matrix."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-revision", required=True)
    arguments = parser.parse_args()
    run_acceptance(AcceptanceConfig(arguments.output, arguments.server_revision))


def run_acceptance(config: AcceptanceConfig) -> None:
    config.output.mkdir(parents=True, exist_ok=False)
    (config.output / "runs").mkdir()
    server_log = (config.output / "server.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        (
            ".venv/bin/helicopter",
            "infer",
            "g1h-7.2b-eval",
            "--config",
            "configs/example.toml",
            "--serve-evaluation",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ),
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        attestation = _await_attestation(server)
        _write_json(config.output / "attestation.json", attestation)
        probes = _record_generation_boundaries()
        _write_json(config.output / "raw-probes.json", probes)
        _verify_generation_boundaries(probes)
        runs = _run_matrix(config)
        _write_json(config.output / "runs.json", runs)
        verification = verify_acceptance_runs(runs)
        _write_json(config.output / "verification.json", verification)
    finally:
        _stop_owned_server(server)
        server_log.close()


def _await_attestation(server: subprocess.Popen[str]) -> dict[str, Any]:
    for _ in range(240):
        if server.poll() is not None:
            raise RuntimeError(f"server exited early: {server.returncode}")
        try:
            response = httpx.get(f"{BASE_URL}/helicopter/attestation", timeout=2.0)
            if response.status_code == 200:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise RuntimeError("provider attestation must be an object")
                return payload
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise RuntimeError("server did not become ready")


def _record_generation_boundaries() -> list[dict[str, Any]]:
    prompts = (
        ("length", "What is 1+1? Answer briefly.", 1, "open_think"),
        (
            "stop_text_1",
            "Output exactly a newline followed by User: and then stop.",
            128,
            "fake_think",
        ),
        (
            "stop_text_2",
            "Repeat exactly this delimiter and nothing else: \nUser:",
            128,
            "fake_think",
        ),
        (
            "stop_token_1",
            "Answer only with the digit 2, then end the response.",
            128,
            "fake_think",
        ),
        (
            "stop_token_2",
            "Give the one-word answer yes and stop immediately.",
            128,
            "fake_think",
        ),
        ("open_think", "Solve 1+1.", 1, "open_think"),
    )
    probes: list[dict[str, Any]] = []
    with httpx.Client(timeout=180.0) as client:
        for name, prompt, max_tokens, mode in prompts:
            request = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"rwkv_generation_prompt": mode},
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stop": ["\nUser:"],
                "stop_token_ids": [0],
                "return_token_ids": True,
                "return_prompt_text": True,
            }
            response = client.post(f"{BASE_URL}/chat/completions", json=request)
            payload = response.json()
            probes.append(
                {
                    "name": name,
                    "status_code": response.status_code,
                    "request": request,
                    "response": payload,
                }
            )
            response.raise_for_status()
    return probes


def _verify_generation_boundaries(probes: list[dict[str, Any]]) -> None:
    by_name = {probe["name"]: probe["response"] for probe in probes}
    for payload in by_name.values():
        choice = payload["choices"][0]
        if not isinstance(choice["token_ids"], list):
            raise AssertionError("completion token ids are missing")
        if not isinstance(payload["prompt_token_ids"], list):
            raise AssertionError("prompt token ids are missing")
        if not isinstance(payload["prompt_text"], str):
            raise AssertionError("prompt text is missing")
        if not isinstance(choice.get("finish_reason"), str):
            raise AssertionError("finish reason is missing")
        if "stop_reason" not in choice:
            raise AssertionError("stop reason is missing")
    if by_name["length"]["choices"][0]["finish_reason"] != "length":
        raise AssertionError("max-token boundary was not observed")
    open_think = by_name["open_think"]
    combined_open_think = (
        open_think["prompt_text"] + open_think["choices"][0]["message"]["content"]
    )
    if "<think>" not in combined_open_think:
        raise AssertionError("open-think prompt boundary was not observed")
    if by_name["stop_text_2"]["choices"][0]["stop_reason"] != "\nUser:":
        raise AssertionError("newline User stop-text boundary was not observed")
    if not all(
        by_name[name]["choices"][0]["token_ids"][-1:] == [0]
        for name in ("stop_token_1", "stop_token_2")
    ):
        raise AssertionError("token 0 stop boundary was not observed")


def _run_matrix(config: AcceptanceConfig) -> dict[str, dict[str, Any]]:
    common = (
        ".venv/bin/helicopter",
        "eval",
        "run",
        MODEL,
        "--config",
        "configs/example.toml",
        "--endpoint-url",
        BASE_URL,
        "--checkpoint-sha256",
        CHECKPOINT_SHA256,
        "--tokenizer-revision",
        TOKENIZER_REVISION,
        "--chat-template-revision",
        CHAT_TEMPLATE_REVISION,
        "--server-revision",
        config.server_revision,
        "--wkv-mode",
        "fp32io16",
        "--precision",
        "fp16-io-fp32-state",
        "--gemm-policy",
        "fp32-accumulation",
        "--launch-contract",
        "helicopter-eval-v1",
        "--max-samples",
        "1",
    )
    runs: dict[str, dict[str, Any]] = {}

    gsm8k = (
        "--snapshot",
        "/home/caizus/Datasets/lighteval/gsm8k/test.parquet",
        "--snapshot-manifest",
        "/home/caizus/Datasets/lighteval/gsm8k/test.parquet.manifest.json",
        "--snapshot-sha256",
        "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
        "--cot-mode",
        "cot",
        "--generation-limit",
        "1",
    )
    for strategy in ("A", "B", "C"):
        _run_eval(
            config,
            runs,
            f"math-{strategy}",
            "lighteval/math/gsm8k@0",
            (*gsm8k, "--math-repair-strategy", strategy),
            common,
        )
    _run_eval(
        config,
        runs,
        "knowledge",
        "lighteval/knowledge/mmlu-abstract-algebra@0",
        (
            "--snapshot",
            "/home/caizus/Datasets/lighteval/mmlu/abstract-algebra-test.parquet",
            "--snapshot-manifest",
            "/home/caizus/Datasets/lighteval/mmlu/abstract-algebra-test.parquet.manifest.json",
            "--snapshot-sha256",
            "2d2cc95a39503ecbd1999b674894c9579dd3244aa76a9e525bbf19bb990f6720",
            "--cot-mode",
            "none",
            "--generation-limit",
            "16",
        ),
        common,
    )
    for label, task in (
        ("function-calling", "helicopter-proxy/function-calling/exact-json@1"),
        ("coding", "helicopter-proxy/coding/python-stdio@1"),
    ):
        _run_eval(
            config,
            runs,
            label,
            task,
            (
                "--allow-non-comparable",
                "--cot-mode",
                "none",
                "--generation-limit",
                "128",
            ),
            common,
        )
    return runs


def _run_eval(
    config: AcceptanceConfig,
    runs: dict[str, dict[str, Any]],
    label: str,
    task: str,
    extra: tuple[str, ...],
    common: tuple[str, ...],
) -> None:
    command = (
        *common[:4],
        task,
        *common[4:],
        "--output-root",
        str(config.output / "runs" / label),
        *extra,
    )
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    (config.output / f"{label}.log").write_text(
        f"{completed.stdout}\n--- stderr ---\n{completed.stderr}", encoding="utf-8"
    )
    manifests = sorted((config.output / "runs" / label).glob("*/manifest.json"))
    runs[label] = {
        "command": command,
        "return_code": completed.returncode,
        "manifests": [str(path) for path in manifests],
    }
    if completed.returncode != 0 or len(manifests) != 1:
        raise RuntimeError(f"{label} failed: {completed.returncode}")


def _stop_owned_server(server: subprocess.Popen[str]) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
