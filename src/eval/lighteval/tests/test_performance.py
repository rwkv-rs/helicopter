from lighteval_runner.results.performance import summarize_run_performance


def test_per_request_usage_is_attributable_but_shared_server_metrics_are_not() -> None:
    evidence = summarize_run_performance(
        [
            {
                "generation": {
                    "request_id": "request-1",
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                }
            }
        ]
    )
    assert evidence.token_usage_attribution == "per_request_usage"
    assert evidence.server_metrics_attribution == "not_attributable"
    assert evidence.total_tokens == 5


def test_missing_request_usage_is_not_attributable() -> None:
    evidence = summarize_run_performance([{"generation": {"usage": None}}])
    assert evidence.token_usage_attribution == "not_attributable"
    assert evidence.total_tokens is None
