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

from scripts.datasets.build_semantic_aggregation_oolong_splits import (  # noqa: E402
    DENIED_OOLONG_BENCHMARK_SOURCES,
    LabeledExample,
    SemanticPool,
    SourceDatasetConfig,
    _validate_source_configs,
    build_dataset,
)


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
    columns = {name: [row[name] for row in rows] for name in SCHEMA.names}
    pq.write_table(pa.Table.from_pydict(columns, schema=SCHEMA), path, compression="zstd")


def _rows(path: Path) -> list[dict]:
    data = pq.read_table(path).to_pydict()
    return [dict(zip(data.keys(), values, strict=True)) for values in zip(*data.values(), strict=True)]


def _fixture_pool_loader(*, split_role: str, max_pool: int, seed: int) -> list[SemanticPool]:
    del split_role, max_pool, seed
    examples = []
    for index in range(80):
        if index % 2 == 0:
            examples.append(LabeledExample(text=f"I liked the service and the food was pleasant {index}.", label="positive"))
        else:
            examples.append(LabeledExample(text=f"The visit was frustrating and the meal was disappointing {index}.", label="negative"))
    return [
        SemanticPool(
            source_name="fixture_reviews",
            description="Each record is a short review. Infer whether sentiment is negative or positive.",
            label_space=("negative", "positive"),
            examples=tuple(examples),
        )
    ]


def test_semantic_aggregation_builder_replaces_only_oolong_and_hides_labels(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    rows = [
        {
            "id": "oolong-1",
            "dataset": "oolong",
            "task": "TASK_TYPE.MOST_FREQ",
            "prompt": "Old code-solvable Oolong row.",
            "context": "Date: Jan 01, 2024 || User: 1 || Text: good || Label: positive",
            "answer": json.dumps(["positive"]),
            "answer_type": "ANSWER_TYPE.LABEL",
            "metadata": json.dumps({"context_window_text_with_labels": "leaky"}),
            "context_token_count": 100,
        },
        {
            "id": "frames-1",
            "dataset": "frames",
            "task": "frames",
            "prompt": "Keep me.",
            "context": "Frame context stays unchanged || Label: allowed for non-oolong",
            "answer": json.dumps(["ok"]),
            "answer_type": "text",
            "metadata": json.dumps({"source": "fixture"}),
            "context_token_count": 10,
        },
    ]
    _write_split(input_dir / "train.parquet", rows)
    _write_split(input_dir / "eval.parquet", rows)

    manifest = build_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=1,
        context_lens=[1600],
        pool_loader=_fixture_pool_loader,
    )

    assert manifest["derived_dataset"] == "oolong_semantic_agg_v2"
    output_rows = {row["id"]: row for row in _rows(output_dir / "train.parquet")}
    assert output_rows["frames-1"]["context"] == rows[1]["context"]

    semantic_rows = [row for row in output_rows.values() if row["dataset"] == "oolong"]
    assert len(semantic_rows) == 1
    semantic = semantic_rows[0]
    assert semantic["id"].startswith("oolong-semantic-v2-train-")
    assert semantic["task"].startswith("TASK_TYPE.SEMANTIC_")
    assert "|| Label:" not in semantic["context"]
    assert "Label:" not in semantic["context"]
    assert "Allowed labels: negative, positive." in semantic["context"]
    assert "### Chunk 1" in semantic["context"]
    metadata = json.loads(semantic["metadata"])
    assert metadata["derived_dataset"] == "oolong_semantic_agg_v2"
    assert metadata["source_dataset"] == "fixture_reviews"
    assert "context_window_text_with_labels" not in metadata
    assert isinstance(json.loads(semantic["answer"]), list)


def test_semantic_aggregation_dataset_loads_through_rlm_adapter(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    rows = [
        {
            "id": "oolong-1",
            "dataset": "oolong",
            "task": "TASK_TYPE.MOST_FREQ",
            "prompt": "Old row.",
            "context": "Text: good || Label: positive",
            "answer": json.dumps(["positive"]),
            "answer_type": "ANSWER_TYPE.LABEL",
            "metadata": "{}",
            "context_token_count": 10,
        }
    ]
    _write_split(input_dir / "train.parquet", rows)
    _write_split(input_dir / "eval.parquet", rows)
    build_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=1,
        context_lens=[1600],
        pool_loader=_fixture_pool_loader,
    )

    build_train, build_eval = build_datasets(
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

    train_dataset = build_train()
    eval_dataset = build_eval()
    assert len(train_dataset) == 1
    assert len(eval_dataset) == 1
    info = json.loads(train_dataset[0]["info"])
    assert info["dataset_name"] == "oolong"
    assert info["source_task"].startswith("TASK_TYPE.SEMANTIC_")
    assert info["metadata"]["derived_dataset"] == "oolong_semantic_agg_v2"
    assert "|| Label:" not in info["context"]


def test_semantic_aggregation_source_configs_reject_oolong_benchmark_sources() -> None:
    denied = next(iter(DENIED_OOLONG_BENCHMARK_SOURCES))
    with pytest.raises(ValueError, match="denied Oolong benchmark"):
        _validate_source_configs(
            [
                SourceDatasetConfig(
                    hf_path=denied,
                    text_column="text",
                    label_column="label",
                    label_names=("negative", "positive"),
                    description="bad source",
                )
            ]
        )
