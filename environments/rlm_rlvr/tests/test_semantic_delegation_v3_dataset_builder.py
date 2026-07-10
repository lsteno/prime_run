from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rlm_rlvr.dataset import build_datasets


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import scripts.datasets.build_semantic_delegation_v3_splits as builder  # noqa: E402


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


def _write_split(path: Path, rows: list[dict]) -> None:
    pq.write_table(
        pa.Table.from_pydict({name: [row[name] for row in rows] for name in SCHEMA.names}, schema=SCHEMA),
        path,
        compression="zstd",
    )


def _read_rows(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def _fixture_pool_loader(
    *, split_role: str, max_pool: int, seed: int, shuffle_buffer: int
) -> list[builder.SemanticPool]:
    del max_pool, seed, shuffle_buffer
    prefix = "train" if split_role == "train" else "eval"
    by_label = {}
    for label in ("negative", "positive"):
        by_label[label] = tuple(
            builder.LabeledExample(
                source_id=f"fixture:{prefix}:{label}:{index}",
                text=f"{prefix} semantic review {label} number {index} with enough words.",
                label=label,
            )
            for index in range(400)
        )
    return [
        builder.SemanticPool(
            source_name="fixture_reviews",
            source_split=prefix,
            description="Classify each review as negative or positive.",
            label_space=("negative", "positive"),
            examples_by_label=by_label,
        )
    ]


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(
        builder,
        "DEFAULT_RECORDS_PER_CHUNK",
        {"semantic_4": 20, "semantic_8": 20, "semantic_16": 20, "semantic_global": 20},
    )


def test_v3_builder_preserves_mix_and_constructs_solvable_hidden_semantics(
    tmp_path: Path, small_chunks
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    base_rows = [
        {
            "id": "frames-1",
            "dataset": "frames",
            "task": "frames",
            "prompt": "Keep me unchanged.",
            "context": "base context",
            "answer": '["ok"]',
            "answer_type": "text",
            "metadata": '{"base":true}',
            "context_token_count": 10,
        }
    ]
    oolong_rows = [
        {
            "id": f"old-oolong-{index}",
            "dataset": "oolong",
            "task": "old",
            "prompt": "old",
            "context": "Text: visible || Label: positive",
            "answer": '["positive"]',
            "answer_type": "label",
            "metadata": "{}",
            "context_token_count": 10,
        }
        for index in range(10)
    ]
    for split in ("train", "eval"):
        _write_split(input_dir / f"{split}.parquet", [*base_rows, *oolong_rows])

    manifest = builder.build_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=2,
        pool_loader=_fixture_pool_loader,
    )

    assert manifest["derived_dataset"] == builder.DERIVED_DATASET
    assert manifest["target_private_hf_repo"] == builder.PRIVATE_HF_REPO
    for split in ("train", "eval"):
        rows = _read_rows(output_dir / f"{split}.parquet")
        assert len(rows) == 11
        assert next(row for row in rows if row["dataset"] == "frames") == base_rows[0]
        semantic_rows = [row for row in rows if row["dataset"] == "oolong"]
        assert len(semantic_rows) == 10
        counts = {}
        for row in semantic_rows:
            metadata = json.loads(row["metadata"])
            bucket = metadata["curriculum_bucket"]
            counts[bucket] = counts.get(bucket, 0) + 1
            assert metadata["record_labels"]
            assert metadata["chunk_labels"]
            assert "record_labels" not in row["prompt"]
            assert "record_labels" not in row["context"]
            assert "|| Label:" not in row["context"]
            per_chunk = {}
            for record_id, label in metadata["record_labels"].items():
                chunk_id = metadata["record_chunks"][record_id]
                per_chunk.setdefault(chunk_id, []).append(label)
            for chunk_id, labels in per_chunk.items():
                gold = metadata["chunk_labels"][chunk_id]
                assert labels.count(gold) == 13
                assert len(labels) == 20
        assert counts == {"semantic_4": 3, "semantic_8": 3, "semantic_16": 3, "semantic_global": 1}


def test_v3_dataset_loads_through_rlm_adapter(tmp_path: Path, small_chunks) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    row = {
        "id": "old-oolong",
        "dataset": "oolong",
        "task": "old",
        "prompt": "old",
        "context": "old",
        "answer": '["old"]',
        "answer_type": "text",
        "metadata": "{}",
        "context_token_count": 1,
    }
    for split in ("train", "eval"):
        _write_split(input_dir / f"{split}.parquet", [row])
    builder.build_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=1,
        pool_loader=_fixture_pool_loader,
    )

    train_builder, eval_builder = build_datasets(
        data_paths=[str(output_dir / "train.parquet")],
        eval_data_paths=[str(output_dir / "eval.parquet")],
        dataset_id=None,
        dataset_train_split="train",
        dataset_eval_split="eval",
        dataset_config=None,
        dataset_revision=None,
        seed=42,
        max_examples=0,
        max_eval_examples=0,
    )
    train_row = train_builder()[0]
    eval_row = eval_builder()[0]
    for adapted in (train_row, eval_row):
        info = json.loads(adapted["info"])
        assert info["metadata"]["derived_dataset"] == builder.DERIVED_DATASET
        assert info["metadata"]["curriculum_bucket"] in builder.BUCKET_FRACTIONS
        assert info["metadata"]["record_labels"]
        assert "record_labels" not in info["context"]


def test_v3_source_configs_reject_oolong_benchmark_sources() -> None:
    with pytest.raises(ValueError, match="denied Oolong benchmark"):
        builder._validate_source_configs(
            [
                builder.SourceDatasetConfig(
                    hf_path="imdb",
                    text_column="text",
                    label_column="label",
                    label_names=("negative", "positive"),
                    description="bad",
                )
            ]
        )


def test_v3_split_deduplication_removes_eval_source_overlap() -> None:
    train = _fixture_pool_loader(split_role="train", max_pool=1, seed=1, shuffle_buffer=1)
    eval_pool = train[0]
    with pytest.raises(RuntimeError, match="No evaluation semantic pools"):
        builder._drop_cross_split_duplicates(train, [eval_pool])


def test_v3_large_macos_guard_is_retained(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "large.parquet"
    path.write_bytes(b"0123456789")
    monkeypatch.setattr(builder.platform, "system", lambda: "Darwin")
    with pytest.raises(SystemExit, match="training VM"):
        builder._guard_large_local_inputs([path], allow_large_local=False, local_guard_bytes=5)
