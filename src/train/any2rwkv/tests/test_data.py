from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from any2rwkv.data import (
    DataPreparationConfig,
    DataPreparationError,
    DuplicateSampleError,
    SPLIT_NAMES,
    prepare_jsonl_dataset,
    prepare_rows,
    stable_split,
)


class TinyTokenizer:
    eos_token_id = 0
    chat_template = "{{ messages }}"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [1 + ord(character) % 31 for character in text]

    def apply_chat_template(self, conversation, *, tokenize: bool, add_generation_prompt: bool):
        self.assert_template_arguments = (tokenize, add_generation_prompt)
        text = " ".join(f"{message['role']}:{message['content']}" for message in conversation)
        return self.encode(text, add_special_tokens=False)


class DataPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = TinyTokenizer()
        self.config = DataPreparationConfig(
            burn_in_tokens=3,
            supervised_tokens=5,
            seed=71,
            exact_duplicate_policy="drop",
            near_duplicate_threshold=0.7,
        )

    def test_output_is_deterministic_and_independent_of_input_order(self) -> None:
        rows = [
            {"sample_id": f"sample-{index:04d}", "text": f"document number {index} has stable content"}
            for index in range(300)
        ]
        forward = prepare_rows(rows, tokenizer=self.tokenizer, config=self.config)
        reverse = prepare_rows(reversed(rows), tokenizer=self.tokenizer, config=self.config)
        self.assertEqual(forward, reverse)
        self.assertEqual(
            stable_split("fixed-id", seed=71),
            stable_split("fixed-id", seed=71),
        )

    def test_split_source_ids_are_mutually_exclusive_and_calibration_is_not_a_gate(self) -> None:
        rows = [
            {"sample_id": f"sample-{index:05d}", "text": "x" * 16 + str(index)}
            for index in range(2000)
        ]
        _, report = prepare_rows(rows, tokenizer=self.tokenizer, config=self.config)
        observed: set[str] = set()
        for split in SPLIT_NAMES:
            ids = set(report["split_sample_ids"][split])
            self.assertFalse(observed & ids)
            observed |= ids
            self.assertTrue(ids, f"expected deterministic fixture coverage for {split}")
        calibration = set(report["split_sample_ids"]["nvfp4_calibration"])
        quality = set().union(
            *(set(report["split_sample_ids"][split]) for split in ("validation", "ruler", "downstream", "smoke"))
        )
        self.assertFalse(calibration & quality)
        self.assertEqual(report["invariants"]["calibration_quality_gate_overlap"], [])

    def test_packing_has_fixed_burn_in_and_supervised_boundaries(self) -> None:
        ratios = {name: "0.0001" for name in SPLIT_NAMES}
        ratios["distill_train"] = "0.9995"
        config = DataPreparationConfig(
            burn_in_tokens=3,
            supervised_tokens=5,
            seed=1,
            split_ratios=ratios,
        )
        rows = [{"sample_id": f"doc-{index}", "text": "abcdefghijk"} for index in range(8)]
        packed, report = prepare_rows(rows, tokenizer=self.tokenizer, config=config)
        self.assertTrue(packed["distill_train"])
        for row in packed["distill_train"]:
            self.assertEqual(len(row["input_ids"]), 8)
            self.assertEqual(row["burn_in_tokens"], 3)
            self.assertEqual(row["supervised_tokens"], 5)
            self.assertEqual(row["source_token_spans"][0]["start"], 0)
            self.assertEqual(row["source_token_spans"][-1]["end"], 8)
        packing = report["packing"]["distill_train"]
        self.assertEqual(packing["packed_token_count"], len(packed["distill_train"]) * 8)
        self.assertLess(packing["dropped_tail_tokens"], 8)

    def test_exact_duplicates_can_be_rejected_or_dropped_with_audit_pair(self) -> None:
        rows = [
            {"sample_id": "canonical", "text": "same   content"},
            {"sample_id": "duplicate", "text": "same content"},
        ]
        strict = DataPreparationConfig(
            burn_in_tokens=1,
            supervised_tokens=2,
            exact_duplicate_policy="reject",
        )
        with self.assertRaises(DuplicateSampleError) as raised:
            prepare_rows(rows, tokenizer=self.tokenizer, config=strict)
        self.assertEqual(raised.exception.report["exact_duplicate_pair_count"], 1)

        packed, report = prepare_rows(rows, tokenizer=self.tokenizer, config=self.config)
        reversed_packed, reversed_report = prepare_rows(
            reversed(rows), tokenizer=self.tokenizer, config=self.config
        )
        self.assertEqual((packed, report), (reversed_packed, reversed_report))
        self.assertEqual(report["accepted_sample_count"], 1)
        self.assertEqual(
            report["exact_duplicates"]["pairs"][0]["duplicate_sample_id"],
            "duplicate",
        )

    def test_near_duplicates_are_reported_with_exact_score_and_split_provenance(self) -> None:
        common = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        rows = [
            {"sample_id": "near-a", "text": common},
            {"sample_id": "near-b", "text": common.replace("lambda", "mu")},
            {"sample_id": "other", "text": "completely unrelated words live in this row"},
        ]
        _, report = prepare_rows(rows, tokenizer=self.tokenizer, config=self.config)
        pairs = report["near_duplicates"]["pairs"]
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0]["left_sample_id"], pairs[0]["right_sample_id"]}, {"near-a", "near-b"})
        self.assertGreaterEqual(pairs[0]["jaccard"], 0.7)
        self.assertIn("left_split", pairs[0])
        self.assertTrue(report["near_duplicates"]["candidate_search_complete"])

    def test_jsonl_output_records_source_tokenizer_template_and_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_path = root / "tokenizer"
            tokenizer_path.mkdir()
            (tokenizer_path / "tokenizer.json").write_text("{}", encoding="utf-8")
            source = root / "rows.jsonl"
            source.write_text(
                "".join(
                    json.dumps({"sample_id": f"id-{index}", "text": "abcdefghijk"}) + "\n"
                    for index in range(20)
                ),
                encoding="utf-8",
            )
            output = root / "prepared"
            manifest = prepare_jsonl_dataset(
                [source],
                output_dir=output,
                tokenizer=self.tokenizer,
                tokenizer_path=tokenizer_path,
                dataset_repository="local/test-data",
                dataset_revision="revision-123",
                tokenizer_repository="Qwen/test",
                tokenizer_revision="tokenizer-revision",
                config=self.config,
            )
            self.assertEqual(manifest["dataset"]["revision"], "revision-123")
            self.assertEqual(len(manifest["dataset"]["local_input_files"][0]["sha256"]), 64)
            self.assertEqual(len(manifest["tokenizer"]["local_tree_sha256"]), 64)
            self.assertEqual(len(manifest["tokenizer"]["chat_template_sha256"]), 64)
            self.assertTrue((output / "deduplication-report.json").is_file())
            self.assertTrue((output / "data-splits.json").is_file())
            self.assertTrue(all((output / f"{split}.jsonl").is_file() for split in SPLIT_NAMES))

    def test_repeated_sample_id_is_rejected_even_when_content_differs(self) -> None:
        rows = [
            {"sample_id": "collision", "text": "first"},
            {"sample_id": "collision", "text": "second"},
        ]
        with self.assertRaisesRegex(DataPreparationError, "globally unique"):
            prepare_rows(rows, tokenizer=self.tokenizer, config=self.config)


if __name__ == "__main__":
    unittest.main()
