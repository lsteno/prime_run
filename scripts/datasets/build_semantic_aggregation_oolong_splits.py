#!/usr/bin/env python3
"""Build BEEG splits with a semantic-aggregation Oolong replacement.

This builder preserves the balanced BEEG mix while replacing rows where
``dataset == "oolong"`` with new tasks that require semantic classification of
unlabeled text records before aggregation. It refuses large local macOS inputs
by default and streams parquet batches so full generation can run on the
training VM without materializing the source parquets in memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


DEFAULT_INPUT_DIR = Path("data/beeg_agents_balanced_35_40_25_frames40_v1")
DEFAULT_OUTPUT_DIR = Path("data/beeg_agents_balanced_35_40_25_frames40_oolong_semantic_agg_v2")
DEFAULT_LOCAL_GUARD_BYTES = 1_000_000_000
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_POOL = 30_000
DEFAULT_CONTEXT_LENS = (18_000, 30_000, 42_000)
DEFAULT_CHUNK_SIZE = 50
DEFAULT_MIN_WINDOW_FILL_RATIO = 0.88
SPLITS = ("train", "eval")
DERIVED_DATASET = "oolong_semantic_agg_v2"
CHARS_PER_TOKEN = 4.0
DATE_RANGE_START = date(2020, 1, 1)
DATE_RANGE_DAYS = (date(2025, 12, 31) - DATE_RANGE_START).days
USER_ID_MIN = 10_000
USER_ID_MAX = 99_999
USERS_PER_WINDOW_MIN = 5
USERS_PER_WINDOW_MAX = 20
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
    label_names: tuple[str, ...]
    description: str
    hf_name: str | None = None
    train_split: str = "train"
    eval_split: str | None = None
    text_column_b: str | None = None
    separator: str = " | "
    labels_are_strings: bool = False
    label_map: dict[str, str | None] | None = None
    max_pool: int = DEFAULT_MAX_POOL

    @property
    def short_name(self) -> str:
        suffix = f"_{self.hf_name}" if self.hf_name else ""
        return f"{self.hf_path.split('/')[-1]}{suffix}"


@dataclass(frozen=True)
class LabeledExample:
    text: str
    label: str


@dataclass(frozen=True)
class SemanticPool:
    source_name: str
    description: str
    label_space: tuple[str, ...]
    examples: tuple[LabeledExample, ...]


@dataclass(frozen=True)
class AnnotatedExample:
    record_id: str
    text: str
    label: str
    date_str: str
    month_year: str
    user_id: int


@dataclass(frozen=True)
class GeneratedTask:
    question: str
    answer: list[str]
    task_group: str
    task: str
    answer_type: str


SOURCE_DATASETS: tuple[SourceDatasetConfig, ...] = (
    SourceDatasetConfig(
        hf_path="stanfordnlp/sst2",
        train_split="train",
        eval_split="validation",
        text_column="sentence",
        label_column="label",
        label_names=("negative", "positive"),
        description="Each record is a movie review sentence. Infer whether its sentiment is negative or positive.",
    ),
    SourceDatasetConfig(
        hf_path="rotten_tomatoes",
        train_split="train",
        eval_split="validation",
        text_column="text",
        label_column="label",
        label_names=("negative", "positive"),
        description="Each record is a movie review excerpt. Infer whether its sentiment is negative or positive.",
    ),
    SourceDatasetConfig(
        hf_path="tweet_eval",
        hf_name="hate",
        train_split="train",
        eval_split="validation",
        text_column="text",
        label_column="label",
        label_names=("non-hate", "hate"),
        description="Each record is a tweet. Infer whether it contains hate speech.",
    ),
    SourceDatasetConfig(
        hf_path="tweet_eval",
        hf_name="offensive",
        train_split="train",
        eval_split="validation",
        text_column="text",
        label_column="label",
        label_names=("non-offensive", "offensive"),
        description="Each record is a tweet. Infer whether it is offensive.",
    ),
    SourceDatasetConfig(
        hf_path="tweet_eval",
        hf_name="irony",
        train_split="train",
        eval_split="validation",
        text_column="text",
        label_column="label",
        label_names=("non-irony", "irony"),
        description="Each record is a tweet. Infer whether it uses irony.",
    ),
    SourceDatasetConfig(
        hf_path="glue",
        hf_name="qnli",
        train_split="train",
        eval_split="validation",
        text_column="question",
        text_column_b="sentence",
        label_column="label",
        label_names=("entailment", "not_entailment"),
        description="Each record contains a question and sentence separated by ' | '. Infer whether the sentence entails the question.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic Oolong aggregation BEEG splits.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pool", type=int, default=DEFAULT_MAX_POOL)
    parser.add_argument("--context-lens", type=int, nargs="+", default=list(DEFAULT_CONTEXT_LENS))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--allow-large-local", action="store_true")
    parser.add_argument("--local-guard-bytes", type=int, default=DEFAULT_LOCAL_GUARD_BYTES)
    return parser.parse_args()


def _resolve_dir(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _guard_large_local_inputs(input_paths: list[Path], *, allow_large_local: bool, local_guard_bytes: int) -> None:
    if allow_large_local or platform.system() != "Darwin":
        return
    total_bytes = sum(path.stat().st_size for path in input_paths)
    if total_bytes > local_guard_bytes:
        raise SystemExit(
            "Refusing to process large local parquet inputs on Darwin without "
            f"--allow-large-local: {total_bytes:,} bytes > {local_guard_bytes:,}. "
            "Run this on the training VM or pass the flag deliberately."
        )


def _estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _validate_source_configs(configs: Iterable[SourceDatasetConfig]) -> None:
    for cfg in configs:
        identifiers = {cfg.hf_path.casefold(), cfg.short_name.casefold()}
        if cfg.hf_name:
            identifiers.add(f"{cfg.hf_path}/{cfg.hf_name}".casefold())
            identifiers.add(cfg.hf_name.casefold())
        overlap = identifiers & DENIED_OOLONG_BENCHMARK_SOURCES
        if overlap:
            raise ValueError(f"Source config overlaps denied Oolong benchmark source(s): {sorted(overlap)}")


def _label_from_row(cfg: SourceDatasetConfig, raw_label: Any) -> str | None:
    if raw_label is None:
        return None
    if cfg.labels_are_strings:
        source_label = _clean_text(str(raw_label))
    else:
        try:
            label_index = int(raw_label)
        except (TypeError, ValueError):
            return None
        if label_index < 0 or label_index >= len(cfg.label_names):
            return None
        source_label = cfg.label_names[label_index]
    if not source_label:
        return None
    if cfg.label_map is not None:
        return cfg.label_map.get(source_label)
    return source_label


def _text_from_row(cfg: SourceDatasetConfig, row: dict[str, Any]) -> str | None:
    first = _clean_text(str(row.get(cfg.text_column, "") or ""))
    if not first:
        return None
    if cfg.text_column_b:
        second = _clean_text(str(row.get(cfg.text_column_b, "") or ""))
        if not second:
            return None
        return f"{first}{cfg.separator}{second}"
    return first


def _load_pool_for_config(cfg: SourceDatasetConfig, *, split_role: str, max_pool: int, seed: int) -> SemanticPool | None:
    split = cfg.train_split if split_role == "train" else (cfg.eval_split or cfg.train_split)
    kwargs: dict[str, Any] = {"path": cfg.hf_path, "split": split}
    if cfg.hf_name:
        kwargs["name"] = cfg.hf_name
    try:
        dataset = load_dataset(**kwargs)
    except Exception:
        if split == cfg.train_split:
            logger.warning("Could not load source %s split %s.", cfg.short_name, split, exc_info=True)
            return None
        kwargs["split"] = cfg.train_split
        try:
            dataset = load_dataset(**kwargs)
        except Exception:
            logger.warning("Could not load source %s fallback split %s.", cfg.short_name, cfg.train_split, exc_info=True)
            return None

    try:
        dataset = dataset.shuffle(seed=seed)
    except Exception:
        logger.warning("Could not shuffle source %s; using source order.", cfg.short_name, exc_info=True)

    examples: list[LabeledExample] = []
    label_space: set[str] = set()
    limit = min(max_pool, cfg.max_pool)
    for row in dataset:
        text = _text_from_row(cfg, row)
        if not text or len(text) < 10 or len(text) > 2_000:
            continue
        label = _label_from_row(cfg, row.get(cfg.label_column))
        if label is None:
            continue
        label = _clean_text(label)
        if not label or re.fullmatch(r"\d+", label):
            continue
        examples.append(LabeledExample(text=text, label=label))
        label_space.add(label)
        if len(examples) >= limit:
            break

    if len(label_space) < 2 or len(examples) < 50:
        logger.warning(
            "Skipping source %s for %s: %d examples across labels %s.",
            cfg.short_name,
            split_role,
            len(examples),
            sorted(label_space),
        )
        return None
    rng = random.Random(seed)
    rng.shuffle(examples)
    logger.info("Loaded source %s for %s: %d examples, labels=%s.", cfg.short_name, split_role, len(examples), sorted(label_space))
    return SemanticPool(
        source_name=cfg.short_name,
        description=cfg.description,
        label_space=tuple(sorted(label_space)),
        examples=tuple(examples),
    )


def load_semantic_pools(*, split_role: str, max_pool: int, seed: int) -> list[SemanticPool]:
    _validate_source_configs(SOURCE_DATASETS)
    pools = [
        pool
        for index, cfg in enumerate(SOURCE_DATASETS)
        if (pool := _load_pool_for_config(cfg, split_role=split_role, max_pool=max_pool, seed=seed + index * 9973))
        is not None
    ]
    if not pools:
        raise RuntimeError(f"No semantic source pools loaded for split role {split_role!r}.")
    return pools


def _random_date(rng: random.Random) -> date:
    return DATE_RANGE_START + timedelta(days=rng.randint(0, DATE_RANGE_DAYS))


def _annotate_window(examples: list[LabeledExample], rng: random.Random) -> list[AnnotatedExample]:
    user_pool = [rng.randint(USER_ID_MIN, USER_ID_MAX) for _ in range(rng.randint(USERS_PER_WINDOW_MIN, USERS_PER_WINDOW_MAX))]
    annotated: list[AnnotatedExample] = []
    for index, example in enumerate(examples, start=1):
        sampled_date = _random_date(rng)
        annotated.append(
            AnnotatedExample(
                record_id=f"r{index:05d}",
                text=example.text,
                label=example.label,
                date_str=sampled_date.strftime("%b %d, %Y"),
                month_year=sampled_date.strftime("%b %Y"),
                user_id=rng.choice(user_pool),
            )
        )
    return annotated


def _sample_window(pool: SemanticPool, *, target_tokens: int, rng: random.Random) -> list[LabeledExample] | None:
    budget = max(1, target_tokens - 160)
    examples_with_costs = [
        (
            example,
            _estimate_tokens(f"Record ID: r99999 || Date: Oct 06, 2022 || User: 81824 || Text: {example.text}"),
        )
        for example in pool.examples
    ]
    min_cost = min(cost for _, cost in examples_with_costs)
    if min_cost > budget:
        return None

    selected: list[LabeledExample] = []
    used = 0
    misses = 0
    while used + min_cost <= budget and misses < 2000:
        example, cost = rng.choice(examples_with_costs)
        if used + cost <= budget:
            selected.append(example)
            used += cost
            misses = 0
        else:
            misses += 1

    if len(selected) < 40 or used < int(budget * DEFAULT_MIN_WINDOW_FILL_RATIO):
        return None
    if len({example.label for example in selected}) < 2:
        return None
    return selected


def _format_context(pool: SemanticPool, annotated: list[AnnotatedExample], *, chunk_size: int) -> str:
    label_list = ", ".join(pool.label_space)
    lines = [
        f"The following unlabeled records contain {len(annotated)} examples, one per line.",
        pool.description,
        f"Allowed labels: {label_list}.",
        (
            "To answer the question, infer the semantic label of each relevant text record, "
            "then aggregate exactly. Labels are not provided in the records."
        ),
    ]
    for index, example in enumerate(annotated):
        if index % chunk_size == 0:
            chunk_index = index // chunk_size + 1
            lines.append(f"### Chunk {chunk_index}")
        lines.append(
            f"Record ID: {example.record_id} || Date: {example.date_str} || "
            f"User: {example.user_id} || Text: {example.text}"
        )
    return "\n".join(lines)


def _ties(counter: Counter[str]) -> list[str]:
    if not counter:
        return []
    top = max(counter.values())
    return sorted(key for key, value in counter.items() if value == top)


def _generate_tasks(annotated: list[AnnotatedExample], pool: SemanticPool, rng: random.Random) -> list[GeneratedTask]:
    label_counter = Counter(example.label for example in annotated)
    present_labels = sorted(label_counter)
    if len(present_labels) < 2:
        return []
    label_list = ", ".join(present_labels)
    tasks = [
        GeneratedTask(
            question=(
                f"In the records above, how many records should be classified as '{rng.choice(present_labels)}'? "
                "Give your final answer in the form 'Count: N'."
            ),
            answer=[],
            task_group="semantic_counting",
            task="TASK_TYPE.SEMANTIC_COUNT_LABEL",
            answer_type="ANSWER_TYPE.NUMERIC",
        )
    ]
    chosen_label = re.search(r"'([^']+)'", tasks[0].question).group(1)  # type: ignore[union-attr]
    tasks[0] = GeneratedTask(
        question=tasks[0].question,
        answer=[str(label_counter[chosen_label])],
        task_group=tasks[0].task_group,
        task=tasks[0].task,
        answer_type=tasks[0].answer_type,
    )

    histogram = {label: label_counter[label] for label in present_labels}
    tasks.append(
        GeneratedTask(
            question=(
                "In the records above, report the exact count for each semantic label. "
                f"Use only these labels as keys: {label_list}. "
                "Give your final answer as a compact JSON object with string keys and integer values."
            ),
            answer=[json.dumps(histogram, sort_keys=True, separators=(",", ":"))],
            task_group="semantic_counting",
            task="TASK_TYPE.SEMANTIC_LABEL_HISTOGRAM",
            answer_type="ANSWER_TYPE.JSON",
        )
    )

    left, right = rng.sample(present_labels, 2)
    if label_counter[left] > label_counter[right]:
        comparison_answer = left
    elif label_counter[right] > label_counter[left]:
        comparison_answer = right
    else:
        comparison_answer = "same"
    tasks.append(
        GeneratedTask(
            question=(
                f"In the records above, which is more common: '{left}', '{right}', or are they the same? "
                f"Give your final answer in the form 'Label: answer' where answer is one of: {left}, {right}, same."
            ),
            answer=[comparison_answer],
            task_group="semantic_counting",
            task="TASK_TYPE.SEMANTIC_COMPARE_LABELS",
            answer_type="ANSWER_TYPE.COMPARISON",
        )
    )

    user_label: dict[int, Counter[str]] = {}
    for example in annotated:
        user_label.setdefault(example.user_id, Counter())[example.label] += 1
    eligible_users = [user for user, counts in user_label.items() if len(counts) >= 2 and sum(counts.values()) >= 4]
    if eligible_users:
        user = rng.choice(eligible_users)
        counts = user_label[user]
        user_labels = sorted(counts)
        user_label_choice = rng.choice(user_labels)
        tasks.append(
            GeneratedTask(
                question=(
                    f"Among records from User {user}, how many should be classified as '{user_label_choice}'? "
                    "Give your final answer in the form 'Count: N'."
                ),
                answer=[str(counts[user_label_choice])],
                task_group="semantic_cross",
                task="TASK_TYPE.SEMANTIC_USER_LABEL_COUNT",
                answer_type="ANSWER_TYPE.NUMERIC",
            )
        )
        top_labels = _ties(counts)
        tasks.append(
            GeneratedTask(
                question=(
                    f"Among records from User {user}, which semantic label is most common? "
                    f"If there is a tie, give any one of the tied labels. "
                    f"Give your final answer in the form 'Label: answer' where answer is one of: {', '.join(user_labels)}."
                ),
                answer=top_labels,
                task_group="semantic_cross",
                task="TASK_TYPE.SEMANTIC_USER_TOP_LABEL",
                answer_type="ANSWER_TYPE.LABEL",
            )
        )

    return tasks


def _semantic_rows(
    *,
    split: str,
    target_rows: int,
    seed: int,
    max_pool: int,
    context_lens: list[int],
    chunk_size: int,
    pool_loader: Callable[..., list[SemanticPool]] = load_semantic_pools,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_rows <= 0:
        return [], {"target_rows": target_rows, "output_rows": 0}
    if not context_lens or any(length < 1000 for length in context_lens):
        raise ValueError("context_lens must contain token targets >= 1000.")
    rng = random.Random(seed)
    pools = pool_loader(split_role=split, max_pool=max_pool, seed=seed)
    if not pools:
        raise RuntimeError(f"No semantic pools available for {split}.")

    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter(target_rows=target_rows, source_pools=len(pools))
    window_id = 0
    attempts = 0
    max_attempts = max(200, target_rows * 20)
    while len(rows) < target_rows and attempts < max_attempts:
        pool = pools[attempts % len(pools)]
        target_tokens = context_lens[(attempts // len(pools)) % len(context_lens)]
        attempts += 1
        sampled = _sample_window(pool, target_tokens=target_tokens, rng=rng)
        if sampled is None:
            counters["skipped_window_sampling"] += 1
            continue
        annotated = _annotate_window(sampled, rng)
        tasks = _generate_tasks(annotated, pool, rng)
        if not tasks:
            counters["skipped_no_tasks"] += 1
            continue
        context = _format_context(pool, annotated, chunk_size=chunk_size)
        if "|| Label:" in context or "Label:" in context:
            raise RuntimeError("Generated visible semantic context leaked labels.")
        context_tokens = _estimate_tokens(context)
        window_id += 1
        counters[f"source/{pool.source_name}"] += 1
        counters[f"context_tokens/{target_tokens}"] += 1
        counters["records_total"] += len(annotated)
        for task in tasks:
            if len(rows) >= target_rows:
                break
            row_index = len(rows)
            metadata = {
                "derived_dataset": DERIVED_DATASET,
                "semantic_aggregation_version": "v2",
                "source_dataset": pool.source_name,
                "task_group": task.task_group,
                "context_window_id": f"{split}-{window_id:05d}",
                "target_context_len": int(target_tokens),
                "context_len": int(context_tokens),
                "num_records": len(annotated),
                "num_labels": len(pool.label_space),
                "label_space": list(pool.label_space),
                "chunk_size": chunk_size,
                "split_role": split,
            }
            rows.append(
                {
                    "id": f"oolong-semantic-v2-{split}-{row_index:06d}",
                    "dataset": "oolong",
                    "task": task.task,
                    "prompt": task.question,
                    "context": context,
                    "answer": json.dumps(task.answer, ensure_ascii=False),
                    "answer_type": task.answer_type,
                    "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    "context_token_count": int(context_tokens),
                }
            )
            counters[f"task/{task.task}"] += 1

    if len(rows) < target_rows:
        raise RuntimeError(f"Generated only {len(rows)} semantic rows for {split}; target was {target_rows}.")
    counters["output_rows"] = len(rows)
    counters["windows"] = window_id
    return rows, dict(counters)


def _table_from_rows(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pydict({name: [row[name] for row in rows] for name in schema.names}, schema=schema)


def _write_table(writer: pq.ParquetWriter | None, table: pa.Table, output_path: Path) -> pq.ParquetWriter:
    if writer is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return pq.ParquetWriter(output_path, table.schema, compression="zstd")
    return writer


def _write_rows(writer: pq.ParquetWriter | None, rows: list[dict[str, Any]], *, schema: pa.Schema, output_path: Path) -> pq.ParquetWriter | None:
    if not rows:
        return writer
    table = _table_from_rows(rows, schema)
    writer = _write_table(writer, table, output_path)
    writer.write_table(table)
    return writer


def transform_split(
    *,
    input_path: Path,
    output_path: Path,
    split: str,
    batch_size: int,
    seed: int,
    max_pool: int,
    context_lens: list[int],
    chunk_size: int,
    pool_loader: Callable[..., list[SemanticPool]] = load_semantic_pools,
) -> dict[str, Any]:
    parquet = pq.ParquetFile(input_path)
    if output_path.exists():
        output_path.unlink()

    counters: Counter[str] = Counter(input_rows=parquet.metadata.num_rows, input_row_groups=parquet.num_row_groups)
    writer: pq.ParquetWriter | None = None
    target_oolong_rows = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            data = batch.to_pydict()
            families = data.get("dataset", [])
            keep_indices = []
            for index, family in enumerate(families):
                counters[f"input_family/{family}"] += 1
                if family == "oolong":
                    target_oolong_rows += 1
                else:
                    keep_indices.append(index)
                    counters[f"output_family/{family}"] += 1
            if not keep_indices:
                continue
            kept = {name: [data[name][index] for index in keep_indices] for name in batch.schema.names}
            table = pa.Table.from_pydict(kept, schema=batch.schema)
            writer = _write_table(writer, table, output_path)
            writer.write_table(table)

        semantic_rows, semantic_summary = _semantic_rows(
            split=split,
            target_rows=target_oolong_rows,
            seed=seed,
            max_pool=max_pool,
            context_lens=context_lens,
            chunk_size=chunk_size,
            pool_loader=pool_loader,
        )
        counters["output_family/oolong"] = len(semantic_rows)
        writer = _write_rows(writer, semantic_rows, schema=parquet.schema_arrow, output_path=output_path)
    finally:
        if writer is not None:
            writer.close()

    output_parquet = pq.ParquetFile(output_path)
    counters["output_rows"] = output_parquet.metadata.num_rows
    counters["output_row_groups"] = output_parquet.num_row_groups
    counters["semantic_target_rows"] = target_oolong_rows
    counters["semantic_output_rows"] = semantic_summary["output_rows"]
    return {**dict(counters), "semantic_generation": semantic_summary}


def _validate_output_split(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    counters: Counter[str] = Counter(rows=parquet.metadata.num_rows)
    for batch in parquet.iter_batches(batch_size=32, columns=["dataset", "task", "context", "metadata", "answer"]):
        data = batch.to_pydict()
        for dataset, task, context, metadata_raw, answer in zip(
            data["dataset"], data["task"], data["context"], data["metadata"], data["answer"], strict=True
        ):
            counters[f"family/{dataset}"] += 1
            if str(task).startswith("TASK_TYPE.SEMANTIC_"):
                counters["semantic_rows"] += 1
                if "|| Label:" in context or "Label:" in context:
                    counters["semantic_context_label_leaks"] += 1
                metadata = json.loads(metadata_raw or "{}")
                if metadata.get("derived_dataset") != DERIVED_DATASET:
                    counters["semantic_missing_metadata_flag"] += 1
                if "context_window_text_with_labels" in metadata or "labels" in metadata and isinstance(metadata.get("labels"), dict):
                    counters["semantic_metadata_label_leaks"] += 1
                parsed_answer = json.loads(answer)
                if not isinstance(parsed_answer, list) or not parsed_answer:
                    counters["semantic_bad_answer"] += 1
    return dict(counters)


def build_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 42,
    max_pool: int = DEFAULT_MAX_POOL,
    context_lens: list[int] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
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
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    resolved_context_lens = list(context_lens or DEFAULT_CONTEXT_LENS)
    _guard_large_local_inputs(input_paths, allow_large_local=allow_large_local, local_guard_bytes=local_guard_bytes)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_summaries = {
        split: transform_split(
            input_path=input_dir / f"{split}.parquet",
            output_path=output_dir / f"{split}.parquet",
            split=split,
            batch_size=batch_size,
            seed=seed + split_index * 10_000,
            max_pool=max_pool,
            context_lens=resolved_context_lens,
            chunk_size=chunk_size,
            pool_loader=pool_loader,
        )
        for split_index, split in enumerate(SPLITS)
    }
    validation = {split: _validate_output_split(output_dir / f"{split}.parquet") for split in SPLITS}
    for split, summary in validation.items():
        for leak_key in ("semantic_context_label_leaks", "semantic_metadata_label_leaks", "semantic_bad_answer"):
            if summary.get(leak_key, 0):
                raise RuntimeError(f"{split} validation failed: {leak_key}={summary[leak_key]}")

    manifest = {
        "source_dataset": "lsteno/BEEG-agents",
        "source_local_dir": str(input_dir),
        "output_dir": str(output_dir),
        "derived_dataset": DERIVED_DATASET,
        "description": (
            "Full balanced BEEG splits with Oolong rows replaced by unlabeled semantic "
            "classification-plus-aggregation tasks."
        ),
        "source_policy": {
            "denied_oolong_benchmark_sources": sorted(DENIED_OOLONG_BENCHMARK_SOURCES),
            "configured_sources": [cfg.short_name for cfg in SOURCE_DATASETS],
            "basic_semantic_labels_only": True,
        },
        "generation": {
            "seed": seed,
            "context_lens": resolved_context_lens,
            "chunk_size": chunk_size,
            "max_pool": max_pool,
        },
        "splits": split_summaries,
        "validation": validation,
        "parquet_paths": {split: str(output_dir / f"{split}.parquet") for split in SPLITS},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "validation_summary.json").write_text(json.dumps(validation, indent=2) + "\n")

    with (output_dir / "split_counts.csv").open("w") as handle:
        handle.write("split,family,count\n")
        for split, summary in validation.items():
            for key, value in sorted(summary.items()):
                if key.startswith("family/"):
                    handle.write(f"{split},{key.removeprefix('family/')},{value}\n")
    return manifest


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    manifest = build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        max_pool=args.max_pool,
        context_lens=list(args.context_lens),
        chunk_size=args.chunk_size,
        allow_large_local=args.allow_large_local,
        local_guard_bytes=args.local_guard_bytes,
    )
    print(json.dumps({"output_dir": manifest["output_dir"], "splits": manifest["validation"]}, indent=2))


if __name__ == "__main__":
    main()
