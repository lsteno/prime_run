from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import scripts.datasets.build_semantic_delegation_v5_splits as builder  # noqa: E402


SCHEMA = pa.schema(
    [
        ("id", pa.large_string()),
        ("dataset", pa.large_string()),
        ("task", pa.large_string()),
        ("prompt", pa.large_string()),
        ("context", pa.large_string()),
        ("answer", pa.large_string()),
        ("answer_type", pa.large_string()),
        ("metadata", pa.large_string()),
        ("context_token_count", pa.int32()),
    ]
)


def _write(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path, compression="zstd")


def _semantic_row() -> dict:
    record_labels = {"r00001": "positive", "r00002": "positive", "r00003": "negative"}
    record_chunks = {record_id: "chunk_001" for record_id in record_labels}
    context = "### chunk_001\n" + "\n".join(
        f"Record ID: {record_id} || Text: semantic text {index}"
        for index, record_id in enumerate(record_labels, start=1)
    )
    metadata = {
        "derived_dataset": builder.DERIVED_DATASET_V4,
        "semantic_delegation_version": "v4",
        "semantic_task_type": "chunk_map",
        "curriculum_bucket": "semantic_4",
        "label_space": ["negative", "positive"],
        "record_labels": record_labels,
        "record_chunks": record_chunks,
        "chunk_labels": {"chunk_001": "positive"},
    }
    return {
        "id": "oolong-semantic-delegation-v4-train-000001",
        "dataset": "oolong",
        "task": "TASK_TYPE.SEMANTIC_CHUNK_MAP",
        "prompt": "old verifier-facing prompt",
        "context": context,
        "answer": '[{"chunk_001":"positive"}]',
        "answer_type": "ANSWER_TYPE.JSON",
        "metadata": json.dumps(metadata),
        "context_token_count": 100,
    }


def test_v5_migration_hides_grading_protocol_and_preserves_base_rows(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    base_row = {
        "id": "frames-1",
        "dataset": "frames",
        "task": "frames",
        "prompt": "unchanged",
        "context": "base",
        "answer": '["ok"]',
        "answer_type": "text",
        "metadata": "{}",
        "context_token_count": 1,
    }
    for split in builder.SPLITS:
        _write(input_dir / f"{split}.parquet", [base_row, _semantic_row()])

    manifest = builder.build_dataset(input_dir=input_dir, output_dir=output_dir, batch_size=1)

    assert manifest["target_private_hf_repo"] == builder.PRIVATE_HF_REPO
    assert manifest["visible_record_ids"] is False
    for split in builder.SPLITS:
        rows = pq.read_table(output_dir / f"{split}.parquet").to_pylist()
        assert rows[0] == base_row
        semantic = rows[1]
        metadata = json.loads(semantic["metadata"])
        assert metadata["derived_dataset"] == builder.DERIVED_DATASET_V5
        assert metadata["semantic_child_credit"] == "natural_visible_majority"
        assert "record_labels" not in metadata
        assert "record_chunks" not in metadata
        assert len(metadata["record_labels_by_text_hash"]) == 3
        assert len(metadata["record_chunks_by_text_hash"]) == 3
        assert "Record ID:" not in semantic["context"]
        assert "|| Text:" not in semantic["context"]
        assert semantic["context"].count("\n- ") == 3
        assert "independent" in semantic["prompt"]
        assert "llm_query" in semantic["prompt"]
        assert "CHUNK_LABEL" not in semantic["prompt"]
        assert "RECORD_MAP" not in semantic["prompt"]


def test_v5_migration_rejects_conflicting_duplicate_visible_text(tmp_path: Path) -> None:
    row = _semantic_row()
    metadata = json.loads(row["metadata"])
    metadata["record_labels"]["r00002"] = "negative"
    row["metadata"] = json.dumps(metadata)
    row["context"] = row["context"].replace("semantic text 2", "semantic text 1")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for split in builder.SPLITS:
        _write(input_dir / f"{split}.parquet", [row])

    with pytest.raises(ValueError, match="conflicting labels"):
        builder.build_dataset(input_dir=input_dir, output_dir=tmp_path / "output")


def test_v5_migration_keeps_large_macos_guard(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for split in builder.SPLITS:
        _write(input_dir / f"{split}.parquet", [_semantic_row()])
    monkeypatch.setattr(builder.platform, "system", lambda: "Darwin")

    with pytest.raises(SystemExit, match="Refusing to migrate"):
        builder.build_dataset(
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            local_guard_bytes=1,
        )
