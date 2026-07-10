from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rlm_rlvr.dataset import build_datasets

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.datasets.build_label_hidden_oolong_splits import build_dataset


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


def test_label_hidden_builder_preserves_full_mix_and_hides_only_oolong(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    oolong_context = (
        "Oolong reviews. The possible labels are listed at the end of each line.\n"
        "Date: 2024-01-01 || User: 1 || Text: loved it || Label: 5_star\n"
        "Date: 2024-01-02 || User: 2 || Text: fine || Label: 3_star"
    )
    frame_context = "Frame row should remain untouched || Label: keep_me"
    rows = [
        {
            "id": "oolong-1",
            "dataset": "oolong",
            "task": "TASK_TYPE.CROSS_COUNT_LABEL_USER",
            "prompt": "How many examples are labeled '5_star'?",
            "context": oolong_context,
            "answer": json.dumps(["1"]),
            "answer_type": "count",
            "metadata": json.dumps(
                {
                    "context_window_text_with_labels": oolong_context,
                    "source": "fixture",
                }
            ),
            "context_token_count": 123,
        },
        {
            "id": "frames-1",
            "dataset": "frames",
            "task": "frames",
            "prompt": "Return the answer.",
            "context": frame_context,
            "answer": json.dumps(["ok"]),
            "answer_type": "text",
            "metadata": json.dumps({"context_window_text_with_labels": frame_context}),
            "context_token_count": 12,
        },
        {
            "id": "longcodeu-1",
            "dataset": "longcodeu",
            "task": "longcodeu",
            "prompt": "Return 7.",
            "context": "Code context",
            "answer": json.dumps(["7"]),
            "answer_type": "number",
            "metadata": json.dumps({"source": "fixture"}),
            "context_token_count": 10,
        },
    ]
    _write_split(input_dir / "train.parquet", rows)
    _write_split(input_dir / "eval.parquet", rows[:2])

    manifest = build_dataset(input_dir=input_dir, output_dir=output_dir, batch_size=2)

    assert manifest["splits"]["train"]["input_rows"] == 3
    assert manifest["splits"]["train"]["output_rows"] == 3
    assert manifest["splits"]["train"]["family/oolong"] == 1
    assert manifest["splits"]["train"]["family/frames"] == 1
    assert manifest["splits"]["train"]["family/longcodeu"] == 1
    assert manifest["splits"]["train"]["oolong_label_suffixes_removed"] == 2

    output_rows = {row["id"]: row for row in _rows(output_dir / "train.parquet")}
    hidden_oolong = output_rows["oolong-1"]
    assert "|| Label:" not in hidden_oolong["context"]
    assert "labels are hidden and must be inferred from the text" in hidden_oolong["context"]
    assert hidden_oolong["answer"] == json.dumps(["1"])
    metadata = json.loads(hidden_oolong["metadata"])
    assert metadata["label_hidden"] is True
    assert metadata["label_hidden_version"] == "v1"
    assert "context_window_text_with_labels" not in metadata
    assert metadata["label_hidden_removed_metadata_keys"] == ["context_window_text_with_labels"]

    assert output_rows["frames-1"]["context"] == frame_context
    assert json.loads(output_rows["frames-1"]["metadata"]) == {
        "context_window_text_with_labels": frame_context
    }


def test_label_hidden_dataset_loads_through_rlm_dataset_adapter(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    rows = [
        {
            "id": "oolong-1",
            "dataset": "oolong",
            "task": "oolong",
            "prompt": "How many examples are labeled '5_star'?",
            "context": "The possible labels are listed at the end of each line.\nText: great || Label: 5_star",
            "answer": json.dumps(["1"]),
            "answer_type": "count",
            "metadata": json.dumps({"context_window_text_with_labels": "Text: great || Label: 5_star"}),
            "context_token_count": 12,
        }
    ]
    _write_split(input_dir / "train.parquet", rows)
    _write_split(input_dir / "eval.parquet", rows)
    build_dataset(input_dir=input_dir, output_dir=output_dir, batch_size=1)

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
    assert "|| Label:" not in info["context"]
    assert info["acceptable_answers"] == ["1"]
