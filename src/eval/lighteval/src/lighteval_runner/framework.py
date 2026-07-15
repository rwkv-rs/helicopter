from __future__ import annotations

from importlib import import_module
import json
from typing import Any, Mapping

from lighteval.models.model_output import ModelResponse

from .context import from_lighteval_doc
from .task_runtime import PreparedSample


class LightEvalTaskRuntime:
    """Pinned boundary around LightEval's task-native prompt and scorer."""

    def __init__(
        self,
        *,
        module_name: str,
        task_name: str,
        dataset_repository: str,
        generation_limit: int,
        primary_metric: str,
    ) -> None:
        module = import_module(module_name)
        matches = [
            config
            for config in getattr(module, "TASKS_TABLE", ())
            if config.name == task_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"pinned LightEval task is missing or ambiguous: {task_name}"
            )
        config = matches[0]
        if config.hf_repo != dataset_repository:
            raise ValueError("pinned LightEval dataset repository drifted")
        if config.generation_size != generation_limit:
            raise ValueError("pinned LightEval generation limit drifted")
        metric_names = {metric.metric_name for metric in config.metrics}
        if primary_metric not in metric_names:
            raise ValueError("pinned LightEval primary metric drifted")
        self._config = config
        self._task_name = task_name

    def prepare(self, row: Mapping[str, Any]) -> PreparedSample:
        doc = self._config.prompt_function(dict(row), task_name=self._task_name)
        golds = doc.get_golds()
        reference = (
            str(golds[0])
            if len(golds) == 1
            else json.dumps(golds, sort_keys=True, ensure_ascii=False, default=str)
        )
        return PreparedSample(
            context=from_lighteval_doc(doc),
            scoring_state=doc,
            reference_answer=reference,
        )

    def score(
        self,
        sample: PreparedSample,
        *,
        prompt: str,
        completion: str,
        output_token_ids: tuple[int, ...],
    ) -> Mapping[str, float]:
        return self._config.metrics[0].compute_sample(
            doc=sample.scoring_state,
            model_response=ModelResponse(
                input=prompt,
                text=[completion],
                output_tokens=[list(output_token_ids)],
            ),
        )
