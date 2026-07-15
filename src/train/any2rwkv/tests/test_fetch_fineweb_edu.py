from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "fetch_fineweb_edu.py"
SPEC = importlib.util.spec_from_file_location("fetch_fineweb_edu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool):
        del add_special_tokens
        return list(range(len(text)))


class FetchFineWebEduTests(unittest.TestCase):
    def test_direct_parquet_mode_uses_pinned_revision_without_repo_listing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            output = root / "rows.jsonl"
            with (
                mock.patch.object(MODULE.AutoTokenizer, "from_pretrained", return_value=_Tokenizer()),
                mock.patch.object(
                    MODULE,
                    "load_dataset",
                    return_value=iter(({"id": "row-1", "text": "enough text"},)),
                ) as load_dataset,
                mock.patch.dict(os.environ, {"HF_ENDPOINT": "https://mirror.example"}),
            ):
                status = MODULE.main(
                    [
                        "--repository", "owner/data",
                        "--revision", "abc123",
                        "--subset", "sample-10BT",
                        "--data-file", "sample/10BT/000.parquet",
                        "--tokenizer-path", str(tokenizer),
                        "--target-tokens", "4",
                        "--output", str(output),
                    ]
                )

            self.assertEqual(status, 0)
            load_dataset.assert_called_once_with(
                "parquet",
                data_files={
                    "train": (
                        "https://mirror.example/datasets/owner/data/resolve/abc123/"
                        "sample/10BT/000.parquet"
                    )
                },
                split="train",
                streaming=True,
            )
            manifest = json.loads(
                output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["data_file"], "sample/10BT/000.parquet")
            self.assertEqual(manifest["revision"], "abc123")

            with (
                mock.patch.object(MODULE.AutoTokenizer, "from_pretrained") as tokenizer_load,
                mock.patch.object(MODULE, "load_dataset") as load_dataset_again,
            ):
                repeated = MODULE.main(
                    [
                        "--repository", "owner/data",
                        "--revision", "abc123",
                        "--subset", "sample-10BT",
                        "--data-file", "sample/10BT/000.parquet",
                        "--tokenizer-path", str(tokenizer),
                        "--target-tokens", "4",
                        "--output", str(output),
                    ]
                )
            self.assertEqual(repeated, 0)
            tokenizer_load.assert_not_called()
            load_dataset_again.assert_not_called()


if __name__ == "__main__":
    unittest.main()
