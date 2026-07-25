from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from datasets import Dataset, load_dataset


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_gsm8k", ROOT / "scripts/prepare_gsm8k.py")
assert SPEC is not None and SPEC.loader is not None
prepare_gsm8k = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_gsm8k)


def test_convert_split_and_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw.parquet"
    Dataset.from_dict(
        {
            "question": ["How many?"],
            "answer": ["Reasoning.\n#### 1,234"],
        }
    ).to_parquet(str(raw))

    train = tmp_path / "gsm8k/train.parquet"
    test = tmp_path / "gsm8k/test.parquet"
    prepare_gsm8k.convert_split(raw, train, "train")
    prepare_gsm8k.convert_split(raw, test, "test")
    manifest = tmp_path / "gsm8k/gsm8k.manifest.json"
    prepare_gsm8k.write_manifest(manifest, [train, test], "a" * 40)

    row = load_dataset("parquet", data_files=str(train), split="train")[0]
    assert row["data_source"] == "openai/gsm8k"
    assert row["prompt"][0]["content"].endswith('final answer after "####".')
    assert row["reward_model"] == {"ground_truth": "1234", "style": "rule"}
    assert row["extra_info"]["split"] == "train"

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["source"] == {"repo": "openai/gsm8k", "revision": "a" * 40}
    assert [Path(item["path"]).name for item in payload["files"]] == ["train.parquet", "test.parquet"]
    assert all(len(item["sha256"]) == 64 for item in payload["files"])
