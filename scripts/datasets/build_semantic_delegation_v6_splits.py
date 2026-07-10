#!/usr/bin/env python3
"""Build private BEEG semantic-delegation v6 packet-majority data.

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
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


DEFAULT_INPUT_DIR = Path("data/beeg_agents_balanced_35_40_25_frames40_v1")
DEFAULT_OUTPUT_DIR = Path("data/beeg_agents_semantic_delegation_v6")
DEFAULT_LOCAL_GUARD_BYTES = 1_000_000_000
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_POOL = 30_000
DEFAULT_STREAM_SHUFFLE_BUFFER = 10_000
RECORDS_PER_PACKET = 12
PACKET_MAJORITY_COUNT = 9
SECTION_COUNT = 4
PACKETS_PER_SECTION = {"small": 3, "medium": 5}
BUCKET_FRACTIONS = {
    "sentiment_small": 0.35,
    "sentiment_medium": 0.25,
    "advanced_small": 0.20,
    "advanced_medium": 0.20,
}
SPLITS = ("train", "eval")
DERIVED_DATASET = "oolong_semantic_delegation_v6"
PRIVATE_HF_REPO = "lsteno/BEEG-agents-semantic-delegation-v6"
CHARS_PER_TOKEN = 4.0
MAX_CONTEXT_CHARS = 150_000
SEMANTIC_EVAL_ROWS = 128
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
    parser.add_argument(
        "--stream-shuffle-buffer", type=int, default=DEFAULT_STREAM_SHUFFLE_BUFFER
    )
    parser.add_argument("--allow-large-local", action="store_true")
    parser.add_argument(
        "--local-guard-bytes", type=int, default=DEFAULT_LOCAL_GUARD_BYTES
    )
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
            identifiers.update(
                {cfg.hf_name.casefold(), f"{cfg.hf_path}/{cfg.hf_name}".casefold()}
            )
        overlap = identifiers & DENIED_OOLONG_BENCHMARK_SOURCES
        if overlap:
            raise ValueError(
                f"Source config overlaps denied Oolong benchmark source(s): {sorted(overlap)}"
            )
        if len(cfg.label_names) != 2 or any(
            re.fullmatch(r"\d+", label) for label in cfg.label_names
        ):
            raise ValueError(
                f"Source {cfg.short_name} must expose two named semantic labels."
            )


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
        rows = _stream_source_rows(
            cfg, split_role=split_role, seed=seed, shuffle_buffer=shuffle_buffer
        )
        for row in rows:
            text = _row_text(cfg, row)
            label = _row_label(cfg, row)
            if not text or label is None or len(text) < 10 or len(text) > 2_000:
                continue
            source_id = _source_id(cfg.short_name, text)
            if source_id in seen:
                continue
            seen.add(source_id)
            by_label[label].append(
                LabeledExample(source_id=source_id, text=text, label=label)
            )
            if sum(len(items) for items in by_label.values()) >= max_pool:
                break
    except Exception:
        logger.warning(
            "Could not stream %s for split %s.",
            cfg.short_name,
            split_role,
            exc_info=True,
        )
        return None

    if min(len(items) for items in by_label.values()) < 20:
        logger.warning(
            "Skipping %s/%s because a label has fewer than 20 records.",
            cfg.short_name,
            split_role,
        )
        return None
    return SemanticPool(
        source_name=cfg.short_name,
        source_split=cfg.train_split if split_role == "train" else cfg.eval_split,
        description=cfg.description,
        label_space=cfg.label_names,
        examples_by_label={label: tuple(items) for label, items in by_label.items()},
    )


def load_semantic_pools(
    *,
    split_role: str,
    max_pool: int,
    seed: int,
    shuffle_buffer: int = DEFAULT_STREAM_SHUFFLE_BUFFER,
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
            kept = tuple(
                example for example in examples if example.source_id not in train_hashes
            )
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
        raise RuntimeError(
            "No evaluation semantic pools remain after split de-duplication."
        )
    return cleaned, removed


def _largest_remainder_counts(
    total: int, fractions: dict[str, float]
) -> dict[str, int]:
    raw = {key: total * value for key, value in fractions.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        fractions, key=lambda key: (raw[key] - counts[key], key), reverse=True
    )
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def _canonical_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _text_hash(value: str) -> str:
    return hashlib.sha256(_canonical_text(value).encode("utf-8")).hexdigest()


def _bucket_stage(bucket: str) -> str:
    return bucket.split("_", 1)[0]


def _bucket_difficulty(bucket: str) -> str:
    return bucket.rsplit("_", 1)[-1]


def _pool_stage(pool: SemanticPool) -> str:
    return (
        "sentiment" if pool.source_name in {"sst2", "rotten_tomatoes"} else "advanced"
    )


def _packet_majorities(
    section_label: str,
    labels: tuple[str, str],
    packets_per_section: int,
    rng: random.Random,
) -> list[str]:
    other = labels[1] if section_label == labels[0] else labels[0]
    majority_packets = packets_per_section // 2 + 1
    values = [section_label] * majority_packets + [other] * (
        packets_per_section - majority_packets
    )
    rng.shuffle(values)
    return values


def _sample_sections(
    *, pool: SemanticPool, difficulty: str, rng: random.Random
) -> tuple[
    list[tuple[str, list[tuple[str, list[LabeledExample]]]]],
    dict[str, str],
    dict[str, str],
]:
    packets_per_section = PACKETS_PER_SECTION[difficulty]
    section_labels = [pool.label_space[0]] * 2 + [pool.label_space[1]] * 2
    rng.shuffle(section_labels)
    packet_specs: list[tuple[str, str, str]] = []
    chunk_labels: dict[str, str] = {}
    packet_labels: dict[str, str] = {}
    for section_index, section_label in enumerate(section_labels, start=1):
        section_id = f"section_{section_index:03d}"
        chunk_labels[section_id] = section_label
        for packet_index, packet_label in enumerate(
            _packet_majorities(
                section_label, pool.label_space, packets_per_section, rng
            ),
            start=1,
        ):
            packet_id = f"{section_id}_packet_{packet_index:02d}"
            packet_specs.append((section_id, packet_id, packet_label))
            packet_labels[packet_id] = packet_label

    required = {label: 0 for label in pool.label_space}
    for _, _, packet_label in packet_specs:
        other = (
            pool.label_space[1]
            if packet_label == pool.label_space[0]
            else pool.label_space[0]
        )
        required[packet_label] += PACKET_MAJORITY_COUNT
        required[other] += RECORDS_PER_PACKET - PACKET_MAJORITY_COUNT
    if any(
        len(pool.examples_by_label[label]) < count for label, count in required.items()
    ):
        raise RuntimeError(
            f"Source {pool.source_name} cannot support one {difficulty} row without record reuse."
        )

    selected = {
        label: iter(rng.sample(list(pool.examples_by_label[label]), count))
        for label, count in required.items()
    }
    by_section: dict[str, list[tuple[str, list[LabeledExample]]]] = defaultdict(list)
    for section_id, packet_id, packet_label in packet_specs:
        other = (
            pool.label_space[1]
            if packet_label == pool.label_space[0]
            else pool.label_space[0]
        )
        records = [next(selected[packet_label]) for _ in range(PACKET_MAJORITY_COUNT)]
        records.extend(
            next(selected[other])
            for _ in range(RECORDS_PER_PACKET - PACKET_MAJORITY_COUNT)
        )
        rng.shuffle(records)
        by_section[section_id].append((packet_id, records))
    sections = [(section_id, by_section[section_id]) for section_id in chunk_labels]
    return sections, chunk_labels, packet_labels


def _format_context(
    *,
    pool: SemanticPool,
    sections: list[tuple[str, list[tuple[str, list[LabeledExample]]]]],
) -> tuple[str, dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    lines = [
        f"The context contains {len(sections)} independent sections split into equal-sized packets.",
        pool.description,
        f"Allowed labels: {pool.label_space[0]}, {pool.label_space[1]}.",
        "A section's answer is the label held by strictly more records across all of its packets.",
    ]
    labels_by_hash: dict[str, str] = {}
    packets_by_hash: dict[str, str] = {}
    sections_by_hash: dict[str, str] = {}
    packet_sections: dict[str, str] = {}
    for section_id, packets in sections:
        lines.append(f"### {section_id}")
        for packet_id, records in packets:
            lines.append(f"#### {packet_id}")
            packet_sections[packet_id] = section_id
            for record in records:
                text_hash = _text_hash(record.text)
                if text_hash in labels_by_hash:
                    raise RuntimeError(
                        "A visible record was reused within one semantic example."
                    )
                lines.append(f"- {_canonical_text(record.text)}")
                labels_by_hash[text_hash] = record.label
                packets_by_hash[text_hash] = packet_id
                sections_by_hash[text_hash] = section_id
    context = "\n".join(lines)
    if len(context) > MAX_CONTEXT_CHARS:
        raise ValueError(
            f"Generated semantic context exceeds {MAX_CONTEXT_CHARS:,} characters."
        )
    return context, labels_by_hash, packets_by_hash, sections_by_hash, packet_sections


def _task_prompt(section_ids: list[str], labels: tuple[str, str]) -> str:
    return (
        "Determine the semantic majority label across all records in every named section. "
        f"Return one compact JSON object whose keys are exactly {json.dumps(section_ids)} and whose values are "
        f"either '{labels[0]}' or '{labels[1]}'.\n\n"
        "Packets are equal-sized natural work units. You may analyze packets independently with llm_query or "
        "llm_query_batched and then combine their conclusions for each section."
    )


def _semantic_rows_from_pools(
    *,
    split: str,
    target_rows: int,
    seed: int,
    pools: list[SemanticPool],
    bucket_counts: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    bucket_counts = bucket_counts or _largest_remainder_counts(
        target_rows, BUCKET_FRACTIONS
    )
    if sum(bucket_counts.values()) != target_rows:
        raise ValueError("Semantic bucket counts must sum to target_rows.")
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter(target_rows=target_rows)

    for bucket, bucket_count in bucket_counts.items():
        stage = _bucket_stage(bucket)
        difficulty = _bucket_difficulty(bucket)
        candidate_pools = [pool for pool in pools if _pool_stage(pool) == stage]
        if not candidate_pools:
            raise RuntimeError(f"No {stage} semantic source pools are available.")
        for bucket_index in range(bucket_count):
            pool = candidate_pools[bucket_index % len(candidate_pools)]
            for attempt in range(20):
                try:
                    sections, chunk_labels, packet_labels = _sample_sections(
                        pool=pool, difficulty=difficulty, rng=rng
                    )
                    (
                        context,
                        labels_by_hash,
                        packets_by_hash,
                        sections_by_hash,
                        packet_sections,
                    ) = _format_context(pool=pool, sections=sections)
                    break
                except ValueError:
                    if attempt == 19:
                        raise
            section_ids = list(chunk_labels)
            row_index = len(rows)
            metadata = {
                "derived_dataset": DERIVED_DATASET,
                "semantic_delegation_version": "v6",
                "semantic_task_type": "chunk_map",
                "semantic_child_credit": "complete_natural_packet_majority",
                "curriculum_bucket": bucket,
                "semantic_source_stage": stage,
                "semantic_difficulty": difficulty,
                "source_dataset": pool.source_name,
                "source_split": pool.source_split,
                "split_role": split,
                "label_space": list(pool.label_space),
                "num_chunks": SECTION_COUNT,
                "num_packets": len(packet_labels),
                "num_records": len(labels_by_hash),
                "records_per_packet": RECORDS_PER_PACKET,
                "packet_majority_count": PACKET_MAJORITY_COUNT,
                "record_labels_by_text_hash": labels_by_hash,
                "record_packets_by_text_hash": packets_by_hash,
                "record_chunks_by_text_hash": sections_by_hash,
                "packet_sections": packet_sections,
                "packet_labels": packet_labels,
                "chunk_labels": chunk_labels,
            }
            rows.append(
                {
                    "id": f"oolong-semantic-delegation-v6-{split}-{row_index:06d}",
                    "dataset": "oolong",
                    "task": "TASK_TYPE.SEMANTIC_CHUNK_MAP",
                    "prompt": _task_prompt(section_ids, pool.label_space),
                    "context": context,
                    "answer": json.dumps(
                        [chunk_labels], ensure_ascii=False, separators=(",", ":")
                    ),
                    "answer_type": "ANSWER_TYPE.JSON",
                    "metadata": json.dumps(
                        metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "context_token_count": _estimate_tokens(context),
                }
            )
            counters[f"bucket/{bucket}"] += 1
            counters[f"source/{pool.source_name}"] += 1
            counters["records_total"] += len(labels_by_hash)

    rng.shuffle(rows)
    counters["output_rows"] = len(rows)
    return rows, {**dict(counters), "bucket_counts": bucket_counts}


def _table_from_rows(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pydict(
        {name: [row[name] for row in rows] for name in schema.names}, schema=schema
    )


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
            kept = {
                name: [data[name][index] for index in keep_indices]
                for name in batch.schema.names
            }
            writer = _write_table(
                writer, pa.Table.from_pydict(kept, schema=batch.schema), output_path
            )
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


def write_semantic_eval_split(
    *, output_path: Path, schema: pa.Schema, seed: int, pools: list[SemanticPool]
) -> dict[str, Any]:
    per_bucket = SEMANTIC_EVAL_ROWS // len(BUCKET_FRACTIONS)
    bucket_counts = {bucket: per_bucket for bucket in BUCKET_FRACTIONS}
    rows, generation = _semantic_rows_from_pools(
        split="semantic_eval",
        target_rows=SEMANTIC_EVAL_ROWS,
        seed=seed,
        pools=pools,
        bucket_counts=bucket_counts,
    )
    if output_path.exists():
        output_path.unlink()
    writer: pq.ParquetWriter | None = None
    try:
        for start in range(0, len(rows), DEFAULT_BATCH_SIZE):
            writer = _write_table(
                writer,
                _table_from_rows(rows[start : start + DEFAULT_BATCH_SIZE], schema),
                output_path,
            )
    finally:
        if writer is not None:
            writer.close()
    return generation


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
            if "Record ID:" in row["context"]:
                counters["visible_record_ids"] += 1
            if "record_labels" in row["prompt"] or "record_labels" in row["context"]:
                counters["visible_gold_map_leaks"] += 1
            record_labels = metadata.get("record_labels_by_text_hash") or {}
            record_packets = metadata.get("record_packets_by_text_hash") or {}
            record_chunks = metadata.get("record_chunks_by_text_hash") or {}
            packet_sections = metadata.get("packet_sections") or {}
            packet_labels = metadata.get("packet_labels") or {}
            chunk_labels = metadata.get("chunk_labels") or {}
            if set(record_labels) != set(record_packets) or set(record_labels) != set(
                record_chunks
            ):
                counters["bad_record_maps"] += 1
            per_packet: dict[str, Counter[str]] = defaultdict(Counter)
            per_chunk: dict[str, Counter[str]] = defaultdict(Counter)
            for text_hash, label in record_labels.items():
                per_packet[record_packets[text_hash]][label] += 1
                per_chunk[record_chunks[text_hash]][label] += 1
            for packet_id, counts in per_packet.items():
                total = sum(counts.values())
                majority_label = max(counts, key=counts.get)
                if (
                    total != RECORDS_PER_PACKET
                    or counts[majority_label] != PACKET_MAJORITY_COUNT
                ):
                    counters["bad_packet_composition"] += 1
                if packet_labels.get(packet_id) != majority_label:
                    counters["bad_packet_gold"] += 1
                if packet_sections.get(packet_id) not in chunk_labels:
                    counters["bad_packet_section"] += 1
            for chunk_id, counts in per_chunk.items():
                majority_label = max(counts, key=counts.get)
                if chunk_labels.get(chunk_id) != majority_label:
                    counters["bad_chunk_gold"] += 1
            expected = json.loads(row["answer"])[0]
            if expected != chunk_labels:
                counters["unsolvable_gold"] += 1
            if Counter(chunk_labels.values()) != Counter(
                {metadata["label_space"][0]: 2, metadata["label_space"][1]: 2}
            ):
                counters["unbalanced_section_targets"] += 1
            if len(row["context"]) > MAX_CONTEXT_CHARS:
                counters["context_char_overflow"] += 1
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
        raise ValueError(
            "batch_size, max_pool, and stream_shuffle_buffer must be positive."
        )
    _guard_large_local_inputs(
        input_paths,
        allow_large_local=allow_large_local,
        local_guard_bytes=local_guard_bytes,
    )

    train_pools = pool_loader(
        split_role="train",
        max_pool=max_pool,
        seed=seed,
        shuffle_buffer=stream_shuffle_buffer,
    )
    eval_pools = pool_loader(
        split_role="eval",
        max_pool=max_pool,
        seed=seed + 10_000,
        shuffle_buffer=stream_shuffle_buffer,
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
    semantic_eval_generation = write_semantic_eval_split(
        output_path=output_dir / "semantic_eval.parquet",
        schema=pq.ParquetFile(input_dir / "eval.parquet").schema_arrow,
        seed=seed + 30_000,
        pools=eval_pools,
    )
    validation_splits = (*SPLITS, "semantic_eval")
    validation = {
        split: _validate_output_split(output_dir / f"{split}.parquet")
        for split in validation_splits
    }
    failure_keys = (
        "visible_label_leaks",
        "visible_record_ids",
        "visible_gold_map_leaks",
        "bad_record_maps",
        "bad_packet_composition",
        "bad_packet_gold",
        "bad_packet_section",
        "bad_chunk_gold",
        "unbalanced_section_targets",
        "context_char_overflow",
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
            "Full 35/40/25 BEEG mixture with Oolong replaced by packetized semantic delegation tasks."
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
            "section_count": SECTION_COUNT,
            "packets_per_section": PACKETS_PER_SECTION,
            "records_per_packet": RECORDS_PER_PACKET,
            "packet_majority_count": PACKET_MAJORITY_COUNT,
            "max_context_chars": MAX_CONTEXT_CHARS,
            "max_pool": max_pool,
            "stream_shuffle_buffer": stream_shuffle_buffer,
        },
        "splits": split_summaries,
        "semantic_eval_generation": semantic_eval_generation,
        "validation": public_validation,
        "parquet_paths": {
            split: str(output_dir / f"{split}.parquet") for split in validation_splits
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "validation_summary.json").write_text(
        json.dumps(public_validation, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
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
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "validation": manifest["validation"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
