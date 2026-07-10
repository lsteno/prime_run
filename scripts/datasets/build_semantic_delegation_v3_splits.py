#!/usr/bin/env python3
"""Build the private BEEG semantic-delegation v3 training dataset.

The builder preserves every non-Oolong row and replaces exactly the Oolong
rows with long-context semantic tasks. Source datasets and parquet inputs are
streamed, and large local macOS runs are rejected unless explicitly allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


DEFAULT_INPUT_DIR = Path("data/beeg_agents_balanced_35_40_25_frames40_v1")
DEFAULT_OUTPUT_DIR = Path("data/beeg_agents_semantic_delegation_v3")
DEFAULT_LOCAL_GUARD_BYTES = 1_000_000_000
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_POOL = 30_000
DEFAULT_STREAM_SHUFFLE_BUFFER = 10_000
DEFAULT_RECORDS_PER_CHUNK = {
    "semantic_4": 120,
    "semantic_8": 80,
    "semantic_16": 60,
    "semantic_global": 60,
}
BUCKET_FRACTIONS = {
    "semantic_4": 0.30,
    "semantic_8": 0.30,
    "semantic_16": 0.30,
    "semantic_global": 0.10,
}
BUCKET_CHUNKS = {
    "semantic_4": 4,
    "semantic_8": 8,
    "semantic_16": 16,
    "semantic_global": 16,
}
SPLITS = ("train", "eval")
DERIVED_DATASET = "oolong_semantic_delegation_v3"
PRIVATE_HF_REPO = "lsteno/BEEG-agents-semantic-delegation-v3"
CHARS_PER_TOKEN = 4.0
MAJORITY_NUMERATOR = 13
MAJORITY_DENOMINATOR = 20
logger = logging.getLogger(__name__)


DENIED_OOLONG_BENCHMARK_SOURCES = {
    "ag_news",
    "app_reviews",
    "dd_encounter",
    "hitz/negation_dataset",
    "imdb",
    "metaphor",
    "metaphors",
    "multi_nli",
    "multinli",
    "oolong-real",
    "oolong-synth",
    "pavlick/formality_scores",
    "sms_spam",
    "spam",
    "trec",
    "trec-qc",
    "yahoo_answers_topics",
}


@dataclass(frozen=True)
class SourceDatasetConfig:
    hf_path: str
    text_column: str
    label_column: str
    label_names: tuple[str, str]
    description: str
    hf_name: str | None = None
    train_split: str = "train"
    eval_split: str = "validation"
    text_column_b: str | None = None
    separator: str = " | "

    @property
    def short_name(self) -> str:
        suffix = f"_{self.hf_name}" if self.hf_name else ""
        return f"{self.hf_path.split('/')[-1]}{suffix}"


@dataclass(frozen=True)
class LabeledExample:
    source_id: str
    text: str
    label: str


@dataclass(frozen=True)
class SemanticPool:
    source_name: str
    source_split: str
    description: str
    label_space: tuple[str, str]
    examples_by_label: dict[str, tuple[LabeledExample, ...]]

    @property
    def source_hashes(self) -> set[str]:
        return {
            example.source_id
            for examples in self.examples_by_label.values()
            for example in examples
        }


SOURCE_DATASETS: tuple[SourceDatasetConfig, ...] = (
    SourceDatasetConfig(
        hf_path="stanfordnlp/sst2",
        text_column="sentence",
        label_column="label",
        label_names=("negative", "positive"),
        description="Each record is a movie-review sentence. Classify its sentiment as negative or positive.",
    ),
    SourceDatasetConfig(
        hf_path="rotten_tomatoes",
        text_column="text",
        label_column="label",
        label_names=("negative", "positive"),
        description="Each record is a movie-review excerpt. Classify its sentiment as negative or positive.",
    ),
    SourceDatasetConfig(
        hf_path="tweet_eval",
        hf_name="hate",
        text_column="text",
        label_column="label",
        label_names=("non-hate", "hate"),
        description="Each record is a tweet. Classify it as non-hate or hate.",
    ),
    SourceDatasetConfig(
        hf_path="tweet_eval",
        hf_name="offensive",
        text_column="text",
        label_column="label",
        label_names=("non-offensive", "offensive"),
        description="Each record is a tweet. Classify it as non-offensive or offensive.",
    ),
    SourceDatasetConfig(
        hf_path="tweet_eval",
        hf_name="irony",
        text_column="text",
        label_column="label",
        label_names=("non-irony", "irony"),
        description="Each record is a tweet. Classify it as non-irony or irony.",
    ),
    SourceDatasetConfig(
        hf_path="glue",
        hf_name="qnli",
        text_column="question",
        text_column_b="sentence",
        label_column="label",
        label_names=("entailment", "not_entailment"),
        description=(
            "Each record contains a question and a sentence separated by ' | '. "
            "Classify it as entailment or not_entailment."
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pool", type=int, default=DEFAULT_MAX_POOL)
    parser.add_argument("--stream-shuffle-buffer", type=int, default=DEFAULT_STREAM_SHUFFLE_BUFFER)
    parser.add_argument("--allow-large-local", action="store_true")
    parser.add_argument("--local-guard-bytes", type=int, default=DEFAULT_LOCAL_GUARD_BYTES)
    return parser.parse_args()


def _resolve_dir(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _guard_large_local_inputs(
    input_paths: list[Path], *, allow_large_local: bool, local_guard_bytes: int
) -> None:
    if allow_large_local or platform.system() != "Darwin":
        return
    total_bytes = sum(path.stat().st_size for path in input_paths)
    if total_bytes > local_guard_bytes:
        raise SystemExit(
            "Refusing to process large local parquet inputs on Darwin without "
            f"--allow-large-local: {total_bytes:,} bytes > {local_guard_bytes:,}. "
            "Run full generation on the training VM."
        )


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _source_id(source_name: str, text: str) -> str:
    digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
    return f"{source_name}:{digest}"


def _estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def _validate_source_configs(configs: Iterable[SourceDatasetConfig]) -> None:
    for cfg in configs:
        identifiers = {cfg.hf_path.casefold(), cfg.short_name.casefold()}
        if cfg.hf_name:
            identifiers.update({cfg.hf_name.casefold(), f"{cfg.hf_path}/{cfg.hf_name}".casefold()})
        overlap = identifiers & DENIED_OOLONG_BENCHMARK_SOURCES
        if overlap:
            raise ValueError(f"Source config overlaps denied Oolong benchmark source(s): {sorted(overlap)}")
        if len(cfg.label_names) != 2 or any(re.fullmatch(r"\d+", label) for label in cfg.label_names):
            raise ValueError(f"Source {cfg.short_name} must expose two named semantic labels.")


def _stream_source_rows(
    cfg: SourceDatasetConfig, *, split_role: str, seed: int, shuffle_buffer: int
) -> Iterator[dict[str, Any]]:
    split = cfg.train_split if split_role == "train" else cfg.eval_split
    kwargs: dict[str, Any] = {"path": cfg.hf_path, "split": split, "streaming": True}
    if cfg.hf_name:
        kwargs["name"] = cfg.hf_name
    dataset = load_dataset(**kwargs)
    if shuffle_buffer > 1:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    yield from dataset


def _row_text(cfg: SourceDatasetConfig, row: dict[str, Any]) -> str | None:
    first = _clean_text(row.get(cfg.text_column))
    if not first:
        return None
    if cfg.text_column_b:
        second = _clean_text(row.get(cfg.text_column_b))
        if not second:
            return None
        return f"{first}{cfg.separator}{second}"
    return first


def _row_label(cfg: SourceDatasetConfig, row: dict[str, Any]) -> str | None:
    try:
        index = int(row.get(cfg.label_column))
    except (TypeError, ValueError):
        return None
    return cfg.label_names[index] if 0 <= index < len(cfg.label_names) else None


def _load_pool_for_config(
    cfg: SourceDatasetConfig,
    *,
    split_role: str,
    max_pool: int,
    seed: int,
    shuffle_buffer: int,
) -> SemanticPool | None:
    by_label: dict[str, list[LabeledExample]] = {label: [] for label in cfg.label_names}
    seen: set[str] = set()
    try:
        rows = _stream_source_rows(cfg, split_role=split_role, seed=seed, shuffle_buffer=shuffle_buffer)
        for row in rows:
            text = _row_text(cfg, row)
            label = _row_label(cfg, row)
            if not text or label is None or len(text) < 10 or len(text) > 2_000:
                continue
            source_id = _source_id(cfg.short_name, text)
            if source_id in seen:
                continue
            seen.add(source_id)
            by_label[label].append(LabeledExample(source_id=source_id, text=text, label=label))
            if sum(len(items) for items in by_label.values()) >= max_pool:
                break
    except Exception:
        logger.warning("Could not stream %s for split %s.", cfg.short_name, split_role, exc_info=True)
        return None

    if min(len(items) for items in by_label.values()) < 20:
        logger.warning("Skipping %s/%s because a label has fewer than 20 records.", cfg.short_name, split_role)
        return None
    return SemanticPool(
        source_name=cfg.short_name,
        source_split=cfg.train_split if split_role == "train" else cfg.eval_split,
        description=cfg.description,
        label_space=cfg.label_names,
        examples_by_label={label: tuple(items) for label, items in by_label.items()},
    )


def load_semantic_pools(
    *, split_role: str, max_pool: int, seed: int, shuffle_buffer: int = DEFAULT_STREAM_SHUFFLE_BUFFER
) -> list[SemanticPool]:
    _validate_source_configs(SOURCE_DATASETS)
    pools = [
        pool
        for index, cfg in enumerate(SOURCE_DATASETS)
        if (
            pool := _load_pool_for_config(
                cfg,
                split_role=split_role,
                max_pool=max_pool,
                seed=seed + index * 9973,
                shuffle_buffer=shuffle_buffer,
            )
        )
        is not None
    ]
    if not pools:
        raise RuntimeError(f"No semantic source pools loaded for {split_role!r}.")
    return pools


def _drop_cross_split_duplicates(
    train_pools: list[SemanticPool], eval_pools: list[SemanticPool]
) -> tuple[list[SemanticPool], int]:
    train_hashes = set().union(*(pool.source_hashes for pool in train_pools))
    cleaned: list[SemanticPool] = []
    removed = 0
    for pool in eval_pools:
        by_label: dict[str, tuple[LabeledExample, ...]] = {}
        for label, examples in pool.examples_by_label.items():
            kept = tuple(example for example in examples if example.source_id not in train_hashes)
            removed += len(examples) - len(kept)
            by_label[label] = kept
        if min(len(items) for items in by_label.values()) >= 20:
            cleaned.append(
                SemanticPool(
                    source_name=pool.source_name,
                    source_split=pool.source_split,
                    description=pool.description,
                    label_space=pool.label_space,
                    examples_by_label=by_label,
                )
            )
    if not cleaned:
        raise RuntimeError("No evaluation semantic pools remain after split de-duplication.")
    return cleaned, removed


def _largest_remainder_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {key: total * value for key, value in fractions.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(fractions, key=lambda key: (raw[key] - counts[key], key), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def _majority_labels_for_row(
    *, chunk_count: int, labels: tuple[str, str], rng: random.Random, global_outcome: str | None
) -> list[str]:
    left, right = labels
    if global_outcome == left:
        left_count = chunk_count // 2 + 2
    elif global_outcome == right:
        left_count = chunk_count // 2 - 2
    else:
        left_count = chunk_count // 2
    values = [left] * left_count + [right] * (chunk_count - left_count)
    rng.shuffle(values)
    return values


def _required_per_label(
    *, records_per_chunk: int, majority_labels: list[str], labels: tuple[str, str]
) -> dict[str, int]:
    majority_count = records_per_chunk * MAJORITY_NUMERATOR // MAJORITY_DENOMINATOR
    minority_count = records_per_chunk - majority_count
    required = {label: 0 for label in labels}
    for majority_label in majority_labels:
        minority_label = labels[1] if majority_label == labels[0] else labels[0]
        required[majority_label] += majority_count
        required[minority_label] += minority_count
    return required


def _eligible_pools(
    pools: list[SemanticPool], *, bucket: str, global_outcome_token: str | None
) -> list[SemanticPool]:
    records_per_chunk = DEFAULT_RECORDS_PER_CHUNK[bucket]
    chunk_count = BUCKET_CHUNKS[bucket]
    eligible = []
    for pool in pools:
        if global_outcome_token == "left":
            global_outcome = pool.label_space[0]
        elif global_outcome_token == "right":
            global_outcome = pool.label_space[1]
        elif global_outcome_token == "same":
            global_outcome = "same"
        else:
            global_outcome = None
        majority_labels = _majority_labels_for_row(
            chunk_count=chunk_count,
            labels=pool.label_space,
            rng=random.Random(0),
            global_outcome=global_outcome,
        )
        required = _required_per_label(
            records_per_chunk=records_per_chunk,
            majority_labels=majority_labels,
            labels=pool.label_space,
        )
        if all(len(pool.examples_by_label[label]) >= count for label, count in required.items()):
            eligible.append(pool)
    return eligible


def _sample_row_records(
    *,
    pool: SemanticPool,
    bucket: str,
    global_outcome: str | None,
    rng: random.Random,
) -> tuple[list[tuple[str, list[LabeledExample]]], dict[str, str]]:
    chunk_count = BUCKET_CHUNKS[bucket]
    records_per_chunk = DEFAULT_RECORDS_PER_CHUNK[bucket]
    majority_count = records_per_chunk * MAJORITY_NUMERATOR // MAJORITY_DENOMINATOR
    majority_labels = _majority_labels_for_row(
        chunk_count=chunk_count,
        labels=pool.label_space,
        rng=rng,
        global_outcome=global_outcome,
    )
    required = _required_per_label(
        records_per_chunk=records_per_chunk,
        majority_labels=majority_labels,
        labels=pool.label_space,
    )
    selected = {
        label: iter(rng.sample(list(pool.examples_by_label[label]), count))
        for label, count in required.items()
    }

    chunks: list[tuple[str, list[LabeledExample]]] = []
    chunk_labels: dict[str, str] = {}
    for chunk_index, majority_label in enumerate(majority_labels, start=1):
        chunk_id = f"chunk_{chunk_index:03d}"
        minority_label = pool.label_space[1] if majority_label == pool.label_space[0] else pool.label_space[0]
        records = [next(selected[majority_label]) for _ in range(majority_count)]
        records.extend(next(selected[minority_label]) for _ in range(records_per_chunk - majority_count))
        rng.shuffle(records)
        chunks.append((chunk_id, records))
        chunk_labels[chunk_id] = majority_label
    return chunks, chunk_labels


def _format_context(
    *, pool: SemanticPool, chunks: list[tuple[str, list[LabeledExample]]]
) -> tuple[str, dict[str, str], dict[str, str]]:
    lines = [
        f"The following context contains {len(chunks)} independent named chunks of unlabeled records.",
        pool.description,
        f"Allowed labels: {pool.label_space[0]}, {pool.label_space[1]}.",
        "A chunk's answer is the label held by strictly more records in that chunk.",
    ]
    record_labels: dict[str, str] = {}
    record_chunks: dict[str, str] = {}
    record_number = 0
    for chunk_id, records in chunks:
        lines.append(f"### {chunk_id}")
        for record in records:
            record_number += 1
            record_id = f"r{record_number:05d}"
            lines.append(f"Record ID: {record_id} || Text: {record.text}")
            record_labels[record_id] = record.label
            record_chunks[record_id] = chunk_id
    context = "\n".join(lines)
    if re.search(r"\|\|\s*Label\s*:", context, flags=re.IGNORECASE):
        raise RuntimeError("Generated visible semantic context leaked per-record labels.")
    return context, record_labels, record_chunks


def _chunk_map_prompt(chunk_ids: list[str], labels: tuple[str, str]) -> str:
    schema_example = json.dumps({chunk_ids[0]: labels[0], chunk_ids[1]: labels[1]}, separators=(",", ":"))
    return (
        "Determine the semantic majority label for every named chunk in the context. "
        f"Return one compact JSON object whose keys are exactly {json.dumps(chunk_ids)} and whose values are "
        f"either '{labels[0]}' or '{labels[1]}'.\n"
        "Use semantic subcalls to classify records rather than guessing from source priors. A useful decomposition is: "
        "send one chunk excerpt to llm_query (or several excerpts to llm_query_batched), ask each child to return a "
        "compact JSON record_id -> label map for every supplied record, then count those labels in the REPL. "
        f"For example, after aggregating two chunks the required final shape is {schema_example}."
    )


def _global_prompt(labels: tuple[str, str]) -> str:
    return (
        "First determine the semantic majority label of each of the 16 named chunks. Then compare how many chunks "
        f"have majority '{labels[0]}' versus '{labels[1]}'. Return exactly one of: {labels[0]}, {labels[1]}, same.\n"
        "Use semantic subcalls to classify records. A useful decomposition is to send each chunk excerpt to llm_query "
        "and request a compact JSON record_id -> label map, then aggregate all chunk majorities in the REPL."
    )


def _semantic_rows_from_pools(
    *, split: str, target_rows: int, seed: int, pools: list[SemanticPool]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    bucket_counts = _largest_remainder_counts(target_rows, BUCKET_FRACTIONS)
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter(target_rows=target_rows)
    global_pools = [
        pool
        for pool in pools
        if all(
            pool in _eligible_pools(pools, bucket="semantic_global", global_outcome_token=outcome)
            for outcome in ("left", "right", "same")
        )
    ]

    for bucket, bucket_count in bucket_counts.items():
        for bucket_index in range(bucket_count):
            if bucket == "semantic_global":
                if not global_pools:
                    raise RuntimeError("No source pool can support all semantic_global outcomes without reuse.")
                source_index = bucket_index % len(global_pools)
                source_round = bucket_index // len(global_pools)
                outcome_token = ("left", "right", "same")[(source_round + source_index) % 3]
                candidate_pools = global_pools
            else:
                outcome_token = None
                candidate_pools = _eligible_pools(pools, bucket=bucket, global_outcome_token=None)
            if not candidate_pools:
                raise RuntimeError(f"No source pool can support {bucket} without within-context record reuse.")
            pool = candidate_pools[bucket_index % len(candidate_pools)]
            if outcome_token == "left":
                global_outcome = pool.label_space[0]
            elif outcome_token == "right":
                global_outcome = pool.label_space[1]
            elif outcome_token == "same":
                global_outcome = "same"
            else:
                global_outcome = None

            chunks, chunk_labels = _sample_row_records(
                pool=pool,
                bucket=bucket,
                global_outcome=global_outcome,
                rng=rng,
            )
            context, record_labels, record_chunks = _format_context(pool=pool, chunks=chunks)
            chunk_ids = [chunk_id for chunk_id, _ in chunks]
            if bucket == "semantic_global":
                prompt = _global_prompt(pool.label_space)
                answer_value: str | dict[str, str] = str(global_outcome)
                task = "TASK_TYPE.SEMANTIC_GLOBAL_CHUNK_COMPARISON"
                answer_type = "ANSWER_TYPE.COMPARISON"
                semantic_task_type = "global_comparison"
            else:
                prompt = _chunk_map_prompt(chunk_ids, pool.label_space)
                answer_value = chunk_labels
                task = "TASK_TYPE.SEMANTIC_CHUNK_MAP"
                answer_type = "ANSWER_TYPE.JSON"
                semantic_task_type = "chunk_map"

            row_index = len(rows)
            metadata = {
                "derived_dataset": DERIVED_DATASET,
                "semantic_delegation_version": "v3",
                "semantic_task_type": semantic_task_type,
                "curriculum_bucket": bucket,
                "source_dataset": pool.source_name,
                "source_split": pool.source_split,
                "split_role": split,
                "label_space": list(pool.label_space),
                "num_chunks": len(chunks),
                "num_records": len(record_labels),
                "records_per_chunk": DEFAULT_RECORDS_PER_CHUNK[bucket],
                "majority_ratio": 0.65,
                "record_labels": record_labels,
                "record_chunks": record_chunks,
                "chunk_labels": chunk_labels,
                "global_outcome": global_outcome,
            }
            rows.append(
                {
                    "id": f"oolong-semantic-delegation-v3-{split}-{row_index:06d}",
                    "dataset": "oolong",
                    "task": task,
                    "prompt": prompt,
                    "context": context,
                    "answer": json.dumps([answer_value], ensure_ascii=False, separators=(",", ":")),
                    "answer_type": answer_type,
                    "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    "context_token_count": _estimate_tokens(context),
                }
            )
            counters[f"bucket/{bucket}"] += 1
            counters[f"source/{pool.source_name}"] += 1
            if global_outcome is not None:
                counters[f"global_outcome/{global_outcome}"] += 1
            counters["records_total"] += len(record_labels)

    rng.shuffle(rows)
    counters["output_rows"] = len(rows)
    return rows, {**dict(counters), "bucket_counts": bucket_counts}


def _table_from_rows(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pydict({name: [row[name] for row in rows] for name in schema.names}, schema=schema)


def _write_table(
    writer: pq.ParquetWriter | None, table: pa.Table, output_path: Path
) -> pq.ParquetWriter:
    if writer is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _count_oolong_and_copy_base(
    *, input_path: Path, output_path: Path, batch_size: int
) -> tuple[int, Counter[str], pq.ParquetWriter | None, pa.Schema]:
    parquet = pq.ParquetFile(input_path)
    if output_path.exists():
        output_path.unlink()
    counters: Counter[str] = Counter(input_rows=parquet.metadata.num_rows)
    writer: pq.ParquetWriter | None = None
    target = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        data = batch.to_pydict()
        keep_indices: list[int] = []
        for index, family in enumerate(data["dataset"]):
            counters[f"input_family/{family}"] += 1
            if family == "oolong":
                target += 1
            else:
                keep_indices.append(index)
                counters[f"output_family/{family}"] += 1
        if keep_indices:
            kept = {name: [data[name][index] for index in keep_indices] for name in batch.schema.names}
            writer = _write_table(writer, pa.Table.from_pydict(kept, schema=batch.schema), output_path)
    return target, counters, writer, parquet.schema_arrow


def transform_split(
    *,
    input_path: Path,
    output_path: Path,
    split: str,
    batch_size: int,
    seed: int,
    pools: list[SemanticPool],
) -> dict[str, Any]:
    target, counters, writer, schema = _count_oolong_and_copy_base(
        input_path=input_path, output_path=output_path, batch_size=batch_size
    )
    try:
        semantic_rows, generation = _semantic_rows_from_pools(
            split=split, target_rows=target, seed=seed, pools=pools
        )
        for start in range(0, len(semantic_rows), batch_size):
            table = _table_from_rows(semantic_rows[start : start + batch_size], schema)
            writer = _write_table(writer, table, output_path)
    finally:
        if writer is not None:
            writer.close()
    output = pq.ParquetFile(output_path)
    counters["output_family/oolong"] = target
    counters["output_rows"] = output.metadata.num_rows
    if counters["input_rows"] != counters["output_rows"]:
        raise RuntimeError(f"{split} row count changed during replacement.")
    return {**dict(counters), "semantic_generation": generation}


def _validate_output_split(path: Path) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    source_ids: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=16,
        columns=["id", "dataset", "task", "prompt", "context", "answer", "metadata"],
    ):
        for row in pa.Table.from_batches([batch]).to_pylist():
            counters["rows"] += 1
            counters[f"family/{row['dataset']}"] += 1
            if not str(row["task"]).startswith("TASK_TYPE.SEMANTIC_"):
                continue
            counters["semantic_rows"] += 1
            metadata = json.loads(row["metadata"])
            bucket = metadata.get("curriculum_bucket")
            counters[f"bucket/{bucket}"] += 1
            if re.search(r"\|\|\s*Label\s*:", row["context"], flags=re.IGNORECASE):
                counters["visible_label_leaks"] += 1
            if "record_labels" in row["prompt"] or "record_labels" in row["context"]:
                counters["visible_gold_map_leaks"] += 1
            record_labels = metadata.get("record_labels") or {}
            record_chunks = metadata.get("record_chunks") or {}
            chunk_labels = metadata.get("chunk_labels") or {}
            if set(record_labels) != set(record_chunks):
                counters["bad_record_maps"] += 1
            per_chunk: dict[str, Counter[str]] = defaultdict(Counter)
            for record_id, label in record_labels.items():
                per_chunk[record_chunks[record_id]][label] += 1
            for chunk_id, counts in per_chunk.items():
                total = sum(counts.values())
                majority_label = max(counts, key=counts.get)
                if counts[majority_label] * 100 != total * 65:
                    counters["bad_majority_ratio"] += 1
                if chunk_labels.get(chunk_id) != majority_label:
                    counters["bad_chunk_gold"] += 1
            expected = json.loads(row["answer"])[0]
            if metadata.get("semantic_task_type") == "chunk_map" and expected != chunk_labels:
                counters["unsolvable_gold"] += 1
            if metadata.get("semantic_task_type") == "global_comparison":
                chunk_counts = Counter(chunk_labels.values())
                labels = metadata["label_space"]
                derived = labels[0] if chunk_counts[labels[0]] > chunk_counts[labels[1]] else labels[1]
                if chunk_counts[labels[0]] == chunk_counts[labels[1]]:
                    derived = "same"
                if expected != derived:
                    counters["unsolvable_gold"] += 1
            source_ids.add(str(row["id"]))
    return {**dict(counters), "semantic_ids": sorted(source_ids)}


def build_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 42,
    max_pool: int = DEFAULT_MAX_POOL,
    stream_shuffle_buffer: int = DEFAULT_STREAM_SHUFFLE_BUFFER,
    allow_large_local: bool = False,
    local_guard_bytes: int = DEFAULT_LOCAL_GUARD_BYTES,
    pool_loader: Callable[..., list[SemanticPool]] = load_semantic_pools,
) -> dict[str, Any]:
    input_dir = _resolve_dir(input_dir)
    output_dir = _resolve_dir(output_dir)
    input_paths = [input_dir / f"{split}.parquet" for split in SPLITS]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input parquet split(s): {missing}")
    if batch_size < 1 or max_pool < 1 or stream_shuffle_buffer < 1:
        raise ValueError("batch_size, max_pool, and stream_shuffle_buffer must be positive.")
    _guard_large_local_inputs(
        input_paths, allow_large_local=allow_large_local, local_guard_bytes=local_guard_bytes
    )

    train_pools = pool_loader(
        split_role="train", max_pool=max_pool, seed=seed, shuffle_buffer=stream_shuffle_buffer
    )
    eval_pools = pool_loader(
        split_role="eval", max_pool=max_pool, seed=seed + 10_000, shuffle_buffer=stream_shuffle_buffer
    )
    eval_pools, duplicate_count = _drop_cross_split_duplicates(train_pools, eval_pools)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_pools = {"train": train_pools, "eval": eval_pools}
    split_summaries = {
        split: transform_split(
            input_path=input_dir / f"{split}.parquet",
            output_path=output_dir / f"{split}.parquet",
            split=split,
            batch_size=batch_size,
            seed=seed + split_index * 10_000,
            pools=split_pools[split],
        )
        for split_index, split in enumerate(SPLITS)
    }
    validation = {split: _validate_output_split(output_dir / f"{split}.parquet") for split in SPLITS}
    failure_keys = (
        "visible_label_leaks",
        "visible_gold_map_leaks",
        "bad_record_maps",
        "bad_majority_ratio",
        "bad_chunk_gold",
        "unsolvable_gold",
    )
    for split, summary in validation.items():
        for key in failure_keys:
            if summary.get(key, 0):
                raise RuntimeError(f"{split} validation failed: {key}={summary[key]}")

    public_validation = {
        split: {key: value for key, value in summary.items() if key != "semantic_ids"}
        for split, summary in validation.items()
    }
    manifest = {
        "source_dataset": "lsteno/BEEG-agents",
        "target_private_hf_repo": PRIVATE_HF_REPO,
        "derived_dataset": DERIVED_DATASET,
        "description": (
            "Full 35/40/25 BEEG mixture with Oolong replaced by chunked semantic delegation tasks."
        ),
        "source_policy": {
            "denied_oolong_benchmark_sources": sorted(DENIED_OOLONG_BENCHMARK_SOURCES),
            "configured_sources": [cfg.short_name for cfg in SOURCE_DATASETS],
            "streaming": True,
            "cross_split_duplicates_removed": duplicate_count,
        },
        "generation": {
            "seed": seed,
            "bucket_fractions": BUCKET_FRACTIONS,
            "records_per_chunk": DEFAULT_RECORDS_PER_CHUNK,
            "majority_ratio": 0.65,
            "max_pool": max_pool,
            "stream_shuffle_buffer": stream_shuffle_buffer,
        },
        "splits": split_summaries,
        "validation": public_validation,
        "parquet_paths": {split: str(output_dir / f"{split}.parquet") for split in SPLITS},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "validation_summary.json").write_text(json.dumps(public_validation, indent=2) + "\n")
    return manifest


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    manifest = build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        max_pool=args.max_pool,
        stream_shuffle_buffer=args.stream_shuffle_buffer,
        allow_large_local=args.allow_large_local,
        local_guard_bytes=args.local_guard_bytes,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "validation": manifest["validation"]}, indent=2))


if __name__ == "__main__":
    main()
