from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import scripts.datasets.build_semantic_delegation_v6_splits as builder  # noqa: E402


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


def _base_row(index: int, dataset: str) -> dict:
    return {
        "id": f"{dataset}-{index}",
        "dataset": dataset,
        "task": dataset,
        "prompt": "unchanged",
        "context": "base",
        "answer": '["ok"]',
        "answer_type": "text",
        "metadata": "{}",
        "context_token_count": 1,
    }


def _pool(
    source_name: str, labels: tuple[str, str], split_role: str
) -> builder.SemanticPool:
    return builder.SemanticPool(
        source_name=source_name,
        source_split=split_role,
        description=f"Classify {source_name} text.",
        label_space=labels,
        examples_by_label={
            label: tuple(
                builder.LabeledExample(
                    source_id=f"{split_role}:{source_name}:{label}:{index}",
                    text=f"{split_role} {source_name} {label} semantic example {index}",
                    label=label,
                )
                for index in range(140)
            )
            for label in labels
        },
    )


def _pool_loader(*, split_role: str, **_: object) -> list[builder.SemanticPool]:
    return [
        _pool("sst2", ("negative", "positive"), split_role),
        _pool("rotten_tomatoes", ("negative", "positive"), split_role),
        _pool("tweet_eval_irony", ("non-irony", "irony"), split_role),
        _pool("glue_qnli", ("entailment", "not_entailment"), split_role),
    ]


def test_v6_builds_balanced_packet_tasks_and_preserves_base_rows(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    rows = [
        _base_row(0, "frames"),
        *(_base_row(index, "oolong") for index in range(1, 5)),
    ]
    for split in builder.SPLITS:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=SCHEMA), input_dir / f"{split}.parquet"
        )
    monkeypatch.setattr(builder, "SEMANTIC_EVAL_ROWS", 4)

    manifest = builder.build_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=2,
        pool_loader=_pool_loader,
    )

    assert manifest["target_private_hf_repo"] == builder.PRIVATE_HF_REPO
    assert "split: semantic_eval" in (output_dir / "README.md").read_text()
    assert pq.ParquetFile(output_dir / "semantic_eval.parquet").metadata.num_rows == 4
    for split in builder.SPLITS:
        output_rows = pq.read_table(output_dir / f"{split}.parquet").to_pylist()
        assert sum(row["dataset"] == "frames" for row in output_rows) == 1
        semantic_rows = [row for row in output_rows if row["dataset"] == "oolong"]
        assert len(semantic_rows) == 4
        assert {
            json.loads(row["metadata"])["curriculum_bucket"] for row in semantic_rows
        } == set(builder.BUCKET_FRACTIONS)
        for row in semantic_rows:
            metadata = json.loads(row["metadata"])
            assert metadata["derived_dataset"] == builder.DERIVED_DATASET
            assert "Record ID:" not in row["context"]
            assert "record_labels" not in row["prompt"]
            assert Counter(metadata["chunk_labels"].values()) == Counter(
                {metadata["label_space"][0]: 2, metadata["label_space"][1]: 2}
            )
            per_packet: dict[str, Counter[str]] = defaultdict(Counter)
            for text_hash, label in metadata["record_labels_by_text_hash"].items():
                per_packet[metadata["record_packets_by_text_hash"][text_hash]][
                    label
                ] += 1
            assert all(
                sum(counts.values()) == 12 and max(counts.values()) == 9
                for counts in per_packet.values()
            )


def test_v6_large_macos_guard_remains_active(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "large.parquet"
    path.write_bytes(b"12")
    monkeypatch.setattr(builder.platform, "system", lambda: "Darwin")

    with pytest.raises(SystemExit, match="Refusing to process large local parquet"):
        builder._guard_large_local_inputs(
            [path], allow_large_local=False, local_guard_bytes=1
        )


def test_v6_cross_split_deduplication_ignores_source_name() -> None:
    train = _pool("sst2", ("negative", "positive"), "train")
    eval_pool = _pool("rotten_tomatoes", ("negative", "positive"), "eval")
    duplicate = train.examples_by_label["positive"][0]
    eval_examples = dict(eval_pool.examples_by_label)
    eval_examples["positive"] = (
        builder.LabeledExample("different-source-id", duplicate.text, "positive"),
        *eval_examples["positive"],
    )
    eval_pool = builder.SemanticPool(
        source_name=eval_pool.source_name,
        source_split=eval_pool.source_split,
        description=eval_pool.description,
        label_space=eval_pool.label_space,
        examples_by_label=eval_examples,
    )

    cleaned, removed = builder._drop_cross_split_duplicates([train], [eval_pool])

    assert removed == 1
    assert all(
        example.text != duplicate.text
        for example in cleaned[0].examples_by_label["positive"]
    )
