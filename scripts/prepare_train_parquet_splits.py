from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class SplitSummary:
    input_path: str
    output_dir: str
    tokenizer_name: str
    seed: int
    total_rows: int
    sft_rows: int
    eval_rows: int
    train_rows: int
    sft_fraction: float
    eval_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add context token counts to train parquet, shuffle deterministically, and split into SFT/Eval/Train."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("/home/coder/rl_training/data/train.parquet"),
        help="Path to the source train.parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/train_with_context_tokens_split_20260313"),
        help="Directory where processed parquet files will be written.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Hugging Face tokenizer used to estimate context token counts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic shuffling.",
    )
    parser.add_argument(
        "--sft-fraction",
        type=float,
        default=0.10,
        help="Fraction of rows allocated to SFT traces split.",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.15,
        help="Fraction of rows allocated to eval split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Parquet read batch size.",
    )
    parser.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=32,
        help="Tokenizer batch size.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=128,
        help="Parquet row group size used for writing output files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.input_path}")
    if args.sft_fraction < 0 or args.eval_fraction < 0:
        raise ValueError("Split fractions must be non-negative.")
    if args.sft_fraction + args.eval_fraction >= 1.0:
        raise ValueError("sft_fraction + eval_fraction must be less than 1.0")
    if args.batch_size <= 0 or args.tokenizer_batch_size <= 0:
        raise ValueError("batch sizes must be positive integers")
    if args.row_group_size <= 0:
        raise ValueError("row-group-size must be a positive integer")


def _token_lengths_for_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    tokenizer_batch_size: int,
) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), tokenizer_batch_size):
        batch = texts[start : start + tokenizer_batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths.extend(len(token_ids) for token_ids in encoded["input_ids"])
    return lengths


def load_with_context_token_counts(
    input_path: Path,
    tokenizer_name: str,
    *,
    batch_size: int,
    tokenizer_batch_size: int,
) -> pa.Table:
    parquet_file = pq.ParquetFile(input_path)
    if "context" not in parquet_file.schema.names:
        raise ValueError("Input parquet must include a 'context' column.")
    total_rows = parquet_file.metadata.num_rows

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    processed_batches: list[pa.RecordBatch] = []
    token_progress = tqdm(
        total=total_rows,
        desc="Tokenizing context",
        unit="rows",
    )

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table_batch = pa.Table.from_batches([batch])
            context_values = table_batch.column("context").to_pylist()
            context_texts = ["" if value is None else str(value) for value in context_values]
            token_lengths = _token_lengths_for_texts(
                context_texts,
                tokenizer,
                tokenizer_batch_size,
            )
            token_lengths_array = pa.array(token_lengths, type=pa.int32())
            batch_with_counts = table_batch.append_column("context_token_count", token_lengths_array)
            processed_batches.extend(batch_with_counts.to_batches())
            token_progress.update(table_batch.num_rows)
    finally:
        token_progress.close()

    return pa.Table.from_batches(processed_batches)


def build_splits(
    table: pa.Table,
    *,
    seed: int,
    sft_fraction: float,
    eval_fraction: float,
) -> tuple[pa.Table, pa.Table, pa.Table, pa.Table]:
    total_rows = table.num_rows
    if total_rows == 0:
        raise ValueError("Input parquet has 0 rows; cannot split.")

    rng = np.random.default_rng(seed)
    permutation = pa.array(rng.permutation(total_rows), type=pa.int64())
    shuffled = table.take(permutation)

    sft_rows = int(round(total_rows * sft_fraction))
    eval_rows = int(round(total_rows * eval_fraction))

    if sft_rows + eval_rows > total_rows:
        overflow = sft_rows + eval_rows - total_rows
        eval_rows = max(0, eval_rows - overflow)

    sft_end = sft_rows
    eval_end = sft_rows + eval_rows

    sft_table = shuffled.slice(0, sft_rows)
    eval_table = shuffled.slice(sft_end, eval_rows)
    train_table = shuffled.slice(eval_end, total_rows - eval_end)
    return shuffled, sft_table, eval_table, train_table


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_table = load_with_context_token_counts(
        args.input_path,
        args.tokenizer_name,
        batch_size=args.batch_size,
        tokenizer_batch_size=args.tokenizer_batch_size,
    )

    shuffled_table, sft_table, eval_table, train_table = build_splits(
        source_table,
        seed=args.seed,
        sft_fraction=args.sft_fraction,
        eval_fraction=args.eval_fraction,
    )

    write_targets = [
        ("train_shuffled_with_context_tokens.parquet", shuffled_table),
        ("sft_traces.parquet", sft_table),
        ("eval.parquet", eval_table),
        ("train.parquet", train_table),
    ]
    for filename, split_table in tqdm(write_targets, desc="Writing parquet files", unit="file"):
        pq.write_table(
            split_table,
            output_dir / filename,
            row_group_size=args.row_group_size,
        )

    summary = SplitSummary(
        input_path=str(args.input_path),
        output_dir=str(output_dir),
        tokenizer_name=args.tokenizer_name,
        seed=args.seed,
        total_rows=shuffled_table.num_rows,
        sft_rows=sft_table.num_rows,
        eval_rows=eval_table.num_rows,
        train_rows=train_table.num_rows,
        sft_fraction=args.sft_fraction,
        eval_fraction=args.eval_fraction,
    )
    (output_dir / "split_summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
