from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from any2rwkv.external_scores import normalize_lm_eval, normalize_ruler2


class ExternalScoreTests(unittest.TestCase):
    def test_lm_eval_normalizer_requires_exact_frozen_metric_and_sample_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results.json"
            results.write_text(json.dumps({"results": {"mmlu": {"acc,none": 0.5}}}), encoding="utf-8")
            samples = root / "samples"
            samples.mkdir()
            (samples / "samples_mmlu_2026.jsonl").write_text(
                json.dumps(
                    {
                        "doc_hash": "a",
                        "prompt_hash": "b",
                        "target_hash": "c",
                        "acc": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = normalize_lm_eval(
                results_json=results,
                samples_dir=samples,
                task_metrics=(("mmlu", "acc"),),
            )
            self.assertEqual(rows, [{"sample_id": "mmlu:a:b:c", "group": "mmlu", "score": 1.0}])
            (samples / "samples_mmlu_duplicate.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one sample log"):
                normalize_lm_eval(
                    results_json=results,
                    samples_dir=samples,
                    task_metrics=(("mmlu", "acc"),),
                )

    def test_ruler_normalizer_requires_complete_bucket_matrix_and_exact_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"id": index, "generation": "x", "is_correct": index / 2}) + "\n"
                    for index in range(3)
                ),
                encoding="utf-8",
            )
            rows = normalize_ruler2(
                inputs=((4096, "qa_basic", path),),
                expected_tasks=("qa_basic",),
                expected_lengths=(4096,),
                samples_per_bucket=3,
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["group"], "qa_basic@4096")
            with self.assertRaisesRegex(ValueError, "bucket mismatch"):
                normalize_ruler2(
                    inputs=(),
                    expected_tasks=("qa_basic",),
                    expected_lengths=(4096,),
                    samples_per_bucket=3,
                )


if __name__ == "__main__":
    unittest.main()
