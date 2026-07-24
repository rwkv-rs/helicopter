from types import SimpleNamespace

from helicopter_lighteval.datasets.knowledge.gpqa.diamond import GpqaDiamond
from helicopter_lighteval.datasets.knowledge.gpqa.main import GpqaMain
from helicopter_lighteval.datasets.maths.gsm8k import Gsm8k


def test_benchmark_prepares_documents_from_the_original_dataset_row() -> None:
    document = SimpleNamespace(
        id="0",
        query="Question: How many?\nAnswer:",
        original_query=None,
        instruction="Answer the question.",
        fewshot_samples=[],
    )
    task = SimpleNamespace(
        dataset={"test": [{"question": "How many?"}]},
        config=SimpleNamespace(evaluation_splits=("test",)),
    )

    benchmark = Gsm8k()
    benchmark.prepare_documents(task=task, documents=[document])

    assert document.query == "How many?"
    assert document.original_query == "How many?"
    assert document.instruction is None


def test_query_revision_is_a_stable_sha256_digest() -> None:
    benchmark = Gsm8k()

    revision = benchmark.query_revision()

    assert len(revision) == 64
    assert revision == benchmark.query_revision()


def test_gpqa_query_revision_uses_the_inherited_query_implementation() -> None:
    assert GpqaDiamond().query_revision() == GpqaMain().query_revision()
