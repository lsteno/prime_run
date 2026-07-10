#!/usr/bin/env python3
"""Build BEEG splits with Oolong per-example labels hidden from context.

The source balanced parquets contain all BEEG families. This script preserves
that mixture and rewrites only rows where dataset == "oolong". It streams
parquet batches to avoid materializing multi-GB contexts in memory.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT_DIR = Path("data/beeg_agents_balanced_35_40_25_frames40_v1")
DEFAULT_OUTPUT_DIR = Path("data/beeg_agents_balanced_35_40_25_frames40_oolong_label_hidden_v1")
DEFAULT_LOCAL_GUARD_BYTES = 1_000_000_000
DEFAULT_BATCH_SIZE = 16
SPLITS = ("train", "eval")

LABEL_SUFFIX_RE = re.compile(r"\s+\|\|\s*Label:\s*[^\r\n]*")
HEADER_PATTERNS = (
    re.compile(r"possible labels are listed at the end of each line", flags=re.IGNORECASE),
    re.compile(r"labels are listed at the end of each line", flags=re.IGNORECASE),
)
HIDDEN_LABEL_HEADER = "labels are hidden and must be inferred from the text when needed"
LEAKY_METADATA_KEYS = {
    "context_window_text_with_labels",
    "context_text_with_labels",
    "context_with_labels",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream balanced BEEG parquets and hide Oolong visible labels."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--allow-large-local",
        action="store_true",
        help="Allow processing large local files on Darwin/macOS. Intended for explicit operator use only.",
    )
    parser.add_argument(
        "--local-guard-bytes",
        type=int,
        default=DEFAULT_LOCAL_GUARD_BYTES,
        help="Total input bytes above which local Darwin processing is refused without --allow-large-local.",
    )
    return parser.parse_args()


def _resolve_dir(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _guard_large_local_inputs(
    input_paths: list[Path],
    *,
    allow_large_local: bool,
    local_guard_bytes: int,
) -> None:
    if allow_large_local or platform.system() != "Darwin":
        return
    total_bytes = sum(path.stat().st_size for path in input_paths)
    if total_bytes > local_guard_bytes:
        raise SystemExit(
            "Refusing to process large local parquet inputs on Darwin without "
            f"--allow-large-local: {total_bytes:,} bytes > {local_guard_bytes:,}. "
            "Run this on the training VM or pass the flag deliberately."
        )


def _hide_context_labels(context: str) -> tuple[str, int]:
    label_suffixes = LABEL_SUFFIX_RE.findall(context)
    rewritten = LABEL_SUFFIX_RE.sub("", context)
    for pattern in HEADER_PATTERNS:
        rewritten, count = pattern.subn(HIDDEN_LABEL_HEADER, rewritten, count=1)
        if count:
            break
    else:
        if HIDDEN_LABEL_HEADER not in rewritten:
            rewritten = f"Note: {HIDDEN_LABEL_HEADER}.\n{rewritten}"
    return rewritten, len(label_suffixes)


def _parse_metadata(raw_metadata: Any) -> dict[str, Any]:
    if isinstance(raw_metadata, dict):
        return dict(raw_metadata)
    if isinstance(raw_metadata, str) and raw_metadata.strip():
        try:
            parsed = json.loads(raw_metadata)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _redact_oolong_metadata(raw_metadata: Any) -> tuple[str, list[str]]:
    metadata = _parse_metadata(raw_metadata)
    removed_keys: list[str] = []
    for key in list(metadata):
        key_lower = key.lower()
        value_text = str(metadata[key])
        if (
            key in LEAKY_METADATA_KEYS
            or "with_labels" in key_lower
            or "|| Label:" in value_text
            or ("context" in key_lower and "label:" in value_text.lower())
        ):
            removed_keys.append(key)
            metadata.pop(key, None)

    metadata["label_hidden"] = True
    metadata["label_hidden_version"] = "v1"
    if removed_keys:
        metadata["label_hidden_removed_metadata_keys"] = sorted(removed_keys)
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")), removed_keys


def _transform_batch(batch: pa.RecordBatch) -> tuple[pa.Table, dict[str, int]]:
    data = batch.to_pydict()
    counters: Counter[str] = Counter(rows=len(next(iter(data.values()), [])))
    families = data.get("dataset", [])
    contexts = data.get("context", [])
    metadata_values = data.get("metadata", [])

    for index, family in enumerate(families):
        counters[f"family/{family}"] += 1
        if family != "oolong":
            continue

        counters["oolong_rows"] += 1
        context = contexts[index] or ""
        hidden_context, removed_labels = _hide_context_labels(context)
        contexts[index] = hidden_context
        counters["oolong_label_suffixes_removed"] += removed_labels

        redacted_metadata, removed_keys = _redact_oolong_metadata(metadata_values[index])
        metadata_values[index] = redacted_metadata
        counters["oolong_metadata_keys_removed"] += len(removed_keys)

    return pa.Table.from_pydict(data, schema=batch.schema), dict(counters)


def transform_split(
    *,
    input_path: Path,
    output_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    parquet = pq.ParquetFile(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    counters: Counter[str] = Counter(
        input_rows=parquet.metadata.num_rows,
        input_row_groups=parquet.num_row_groups,
    )
    writer: pq.ParquetWriter | None = None
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            table, batch_counts = _transform_batch(batch)
            counters.update(batch_counts)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    output_parquet = pq.ParquetFile(output_path)
    counters["output_rows"] = output_parquet.metadata.num_rows
    counters["output_row_groups"] = output_parquet.num_row_groups
    return dict(counters)


def build_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    allow_large_local: bool = False,
    local_guard_bytes: int = DEFAULT_LOCAL_GUARD_BYTES,
) -> dict[str, Any]:
    input_dir = _resolve_dir(input_dir)
    output_dir = _resolve_dir(output_dir)
    input_paths = [input_dir / f"{split}.parquet" for split in SPLITS]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input parquet split(s): {missing}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    _guard_large_local_inputs(
        input_paths,
        allow_large_local=allow_large_local,
        local_guard_bytes=local_guard_bytes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_summaries = {
        split: transform_split(
            input_path=input_dir / f"{split}.parquet",
            output_path=output_dir / f"{split}.parquet",
            batch_size=batch_size,
        )
        for split in SPLITS
    }

    manifest = {
        "source_dataset": "lsteno/BEEG-agents",
        "source_local_dir": str(input_dir),
        "output_dir": str(output_dir),
        "label_hidden_version": "v1",
        "description": (
            "Full balanced BEEG train/eval splits with only Oolong visible per-example "
            "label suffixes removed from context. Gold answers are preserved."
        ),
        "transform": {
            "families_preserved": True,
            "transformed_family": "oolong",
            "removed_context_regex": LABEL_SUFFIX_RE.pattern,
            "metadata_redaction": sorted(LEAKY_METADATA_KEYS),
        },
        "splits": split_summaries,
        "parquet_paths": {split: str(output_dir / f"{split}.parquet") for split in SPLITS},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    with (output_dir / "split_counts.csv").open("w") as handle:
        handle.write("split,family,count\n")
        for split, summary in split_summaries.items():
            family_counts = {
                key.removeprefix("family/"): value
                for key, value in summary.items()
                if key.startswith("family/")
            }
            for family, count in sorted(family_counts.items()):
                handle.write(f"{split},{family},{count}\n")

    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        allow_large_local=args.allow_large_local,
        local_guard_bytes=args.local_guard_bytes,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
