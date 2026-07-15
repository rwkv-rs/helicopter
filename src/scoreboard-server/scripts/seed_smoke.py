from __future__ import annotations

import asyncio

from scoreboard_server.db.connection import close_db, init_db
from scoreboard_server.db.settings import DatabaseSettings
from scoreboard_server.db.repository import ScoreboardStore


async def main() -> None:
    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=True)
    service = ScoreboardStore(settings=settings)
    await service.ensure_benchmark_num_samples(dataset="gsm8k_test", num_samples=2)
    task_id = await service.get_or_create_task(
        job_name="free_response",
        job_id="smoke",
        dataset="gsm8k_test",
        model="rwkv7-g1g-1.5b",
        is_param_search=False,
        sampling_config={
            "avg_k": 1,
            "pass_ks": [1],
            "prompt_profile": "normal",
            "sampling_config": {
                "answer": {
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "max_new_tokens": 128,
                    "stop_tokens": [0],
                },
            },
        },
        allow_resume=False,
    )
    await service.insert_completion_payloads_batch(
        task_id=task_id,
        payloads=[
            {
                "sample_index": 0,
                "repeat_index": 0,
                "pass_index": 0,
                "prompt1": "What is 1+1?",
                "completion1": "2",
                "stop_reason1": "stop",
                "sampling_config": {"answer": {"temperature": 0.2, "stop_tokens": [0]}},
            },
            {
                "sample_index": 1,
                "repeat_index": 0,
                "pass_index": 0,
                "prompt1": "What is 1+2?",
                "completion1": "4",
                "stop_reason1": "stop",
                "sampling_config": {"answer": {"temperature": 0.2, "stop_tokens": [0]}},
            },
        ],
    )
    await service.ingest_eval_payloads(
        task_id=task_id,
        payloads=[
            {
                "sample_index": 0,
                "repeat_index": 0,
                "pass_index": 0,
                "answer": "2",
                "ref_answer": "2",
                "is_passed": True,
                "fail_reason": "",
            },
            {
                "sample_index": 1,
                "repeat_index": 0,
                "pass_index": 0,
                "answer": "4",
                "ref_answer": "3",
                "is_passed": False,
                "fail_reason": "wrong arithmetic",
            },
        ],
    )
    await service.record_score_payload(
        task_id=task_id,
        payload={"cot_mode": "NoCoT", "metrics": {"avg@1": 0.5}},
    )
    coding_task_id = await service.get_or_create_task(
        job_name="code_generation",
        job_id="smoke-code",
        dataset="humaneval_test",
        model="rwkv7-g1g-1.5b",
        is_param_search=False,
        sampling_config={"pass_ks": [1]},
        allow_resume=False,
    )
    await service.record_score_payload(
        task_id=coding_task_id,
        payload={"cot_mode": "NoCoT", "metrics": {"pass@1": 0.25}},
    )
    print(f"seeded task_id={task_id} coding_task_id={coding_task_id}")
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
