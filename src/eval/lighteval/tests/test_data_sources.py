from __future__ import annotations

import hashlib
import json

import pytest

from lighteval_runner.data_sources.acquisition import (
    DatasetSource,
    materialize_jsonl_snapshot,
    materialize_parquet_snapshot,
    select_snapshot_rows,
)


REVISION = "a" * 40


def write_snapshot(tmp_path, rows):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "data.jsonl"
    encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(encoded)
    source = DatasetSource(
        "owner/data", REVISION, "data.jsonl", hashlib.sha256(encoded).hexdigest()
    )
    return path, source


def test_mutable_revision_and_digest_mismatch_are_rejected(tmp_path):
    path, source = write_snapshot(tmp_path, [{"id": "a"}])
    with pytest.raises(ValueError, match="immutable"):
        materialize_jsonl_snapshot(
            path,
            source.__class__(
                source.repository, "main", source.source_file, source.sha256
            ),
            validate_row=lambda _: None,
        )
    with pytest.raises(ValueError, match="digest mismatch"):
        materialize_jsonl_snapshot(
            path,
            source.__class__(
                source.repository, source.revision, source.source_file, "0" * 64
            ),
            validate_row=lambda _: None,
        )


def test_row_identity_and_selection_are_stable_under_dataset_reorder(tmp_path):
    rows = [{"question": "b"}, {"question": "a"}, {"id": "explicit", "question": "c"}]
    first_path, first_source = write_snapshot(tmp_path / "first", rows)
    second_path, second_source = write_snapshot(
        tmp_path / "second", list(reversed(rows))
    )
    first = materialize_jsonl_snapshot(
        first_path, first_source, validate_row=lambda _: None
    )
    second = materialize_jsonl_snapshot(
        second_path, second_source, validate_row=lambda _: None
    )

    assert {row.row_id for row in first.accepted_rows} == {
        row.row_id for row in second.accepted_rows
    }
    assert [row.row_id for row in select_snapshot_rows(first, 2)] == [
        row.row_id for row in select_snapshot_rows(second, 2)
    ]


def test_duplicate_stable_row_identity_is_rejected(tmp_path):
    path, source = write_snapshot(
        tmp_path, [{"question": "same"}, {"question": "same"}]
    )
    with pytest.raises(ValueError, match="duplicate stable"):
        materialize_jsonl_snapshot(path, source, validate_row=lambda _: None)


def test_dataset_and_formatter_rejections_remain_separate(tmp_path):
    path, source = write_snapshot(tmp_path, [{"id": "ok"}, {"id": "bad", "skip": True}])
    snapshot = materialize_jsonl_snapshot(
        path, source, validate_row=lambda row: "schema" if row.get("skip") else None
    )

    assert snapshot.source_rows == 2
    assert [row.row_id for row in snapshot.accepted_rows] == ["ok"]
    assert [(row.row_id, row.reason) for row in snapshot.rejected_rows] == [
        ("bad", "schema")
    ]


def test_max_samples_selects_only_from_accepted_partition(tmp_path):
    path, source = write_snapshot(
        tmp_path,
        [{"id": "rejected", "skip": True}, {"id": "b"}, {"id": "a"}],
    )
    snapshot = materialize_jsonl_snapshot(
        path,
        source,
        validate_row=lambda row: "schema" if row.get("skip") else None,
    )
    selected = select_snapshot_rows(snapshot, 1)
    assert snapshot.source_rows == 3
    assert len(snapshot.rejected_rows) == 1
    assert [row.row_id for row in selected] == ["a"]


def test_max_samples_must_be_positive(tmp_path):
    path, source = write_snapshot(tmp_path, [{"id": "a"}])
    snapshot = materialize_jsonl_snapshot(path, source, validate_row=lambda _: None)
    with pytest.raises(ValueError):
        select_snapshot_rows(snapshot, 0)


def test_parquet_is_verified_before_parsing_and_preserves_stable_rows(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "data.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"id": "b", "value": 2}, {"id": "a", "value": 1}]), path
    )
    encoded = path.read_bytes()
    source = DatasetSource(
        "owner/data", REVISION, path.name, hashlib.sha256(encoded).hexdigest()
    )
    snapshot = materialize_parquet_snapshot(path, source, validate_row=lambda _: None)

    assert snapshot.source_rows == 2
    assert [row.row_id for row in select_snapshot_rows(snapshot, None)] == ["a", "b"]
    with pytest.raises(ValueError, match="digest mismatch"):
        materialize_parquet_snapshot(
            path,
            DatasetSource(
                source.repository, source.revision, source.source_file, "0" * 64
            ),
            validate_row=lambda _: None,
        )
