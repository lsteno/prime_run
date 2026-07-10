from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import scripts.datasets.build_semantic_delegation_v4_splits as builder  # noqa: E402


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
    record_labels = {f"r{index:05d}": "positive" for index in range(1, 9)}
    record_chunks = {record_id: "chunk_001" for record_id in record_labels}
    context = "### chunk_001\n" + "\n".join(
        f"Record ID: {record_id} || Text: semantic text {index}"
        for index, record_id in enumerate(record_labels, start=1)
    )
    metadata = {
        "derived_dataset": builder.DERIVED_DATASET_V3,
        "semantic_delegation_version": "v3",
        "semantic_task_type": "chunk_map",
        "curriculum_bucket": "semantic_4",
        "label_space": ["negative", "positive"],
        "record_labels": record_labels,
        "record_chunks": record_chunks,
        "chunk_labels": {"chunk_001": "positive"},
    }
    return {
        "id": "oolong-semantic-delegation-v3-train-000001",
        "dataset": "oolong",
        "task": "TASK_TYPE.SEMANTIC_CHUNK_MAP",
        "prompt": "old v3 prompt",
        "context": context,
        "answer": '[{"chunk_001":"positive"}]',
        "answer_type": "ANSWER_TYPE.JSON",
        "metadata": json.dumps(metadata),
        "context_token_count": 100,
    }


def test_v4_migration_changes_only_semantic_contract(tmp_path: Path) -> None:
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
    for split in builder.SPLITS:
        rows = pq.read_table(output_dir / f"{split}.parquet").to_pylist()
        assert rows[0] == base_row
        semantic = rows[1]
        metadata = json.loads(semantic["metadata"])
        assert metadata["derived_dataset"] == builder.DERIVED_DATASET_V4
        assert metadata["semantic_child_contract"] == "verified_chunk_or_record_map"
        assert metadata["record_labels"]
        assert "CHUNK_LABEL" in semantic["prompt"]
        assert "RECORD_MAP" in semantic["prompt"]
        assert "at least eight" in semantic["prompt"]
        assert "record_labels" not in semantic["prompt"]
        assert semantic["context"] == _semantic_row()["context"]


def test_v4_migration_keeps_large_macos_guard(tmp_path: Path, monkeypatch) -> None:
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
