import inspect


def test_pinned_lighteval_public_api_is_available() -> None:
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.pipeline import Pipeline, PipelineParameters
    from lighteval.tasks.prompt_manager import PromptManager
    from lighteval.tasks.registry import Registry

    assert "tasks" in inspect.signature(Pipeline).parameters
    assert "max_samples" in inspect.signature(PipelineParameters).parameters
    assert "output_dir" in inspect.signature(EvaluationTracker).parameters
    assert hasattr(Registry, "load_tasks")
    assert hasattr(PromptManager, "prepare_prompt_api")


def test_pinned_litellm_backend_does_not_preserve_terminal_evidence() -> None:
    from lighteval.models.endpoints.litellm_model import LiteLLMClient

    source = inspect.getsource(LiteLLMClient.greedy_until)
    assert "finish_reason" not in source
    assert "prompt_token_ids" not in source
    assert "output_token_ids" not in source
