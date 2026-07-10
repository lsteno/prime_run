#!/usr/bin/env python3
"""Migrate semantic delegation v3 parquet splits to the strict v4 child contract."""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT_DIR = Path("data/beeg_agents_semantic_delegation_v3")
DEFAULT_OUTPUT_DIR = Path("data/beeg_agents_semantic_delegation_v4")
DEFAULT_BATCH_SIZE = 16
DEFAULT_LOCAL_GUARD_BYTES = 1_000_000_000
DERIVED_DATASET_V3 = "oolong_semantic_delegation_v3"
DERIVED_DATASET_V4 = "oolong_semantic_delegation_v4"
PRIVATE_HF_REPO = "lsteno/BEEG-agents-semantic-delegation-v4"
SPLITS = ("train", "eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--allow-large-local", action="store_true")
    parser.add_argument("--local-guard-bytes", type=int, default=DEFAULT_LOCAL_GUARD_BYTES)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _guard_inputs(paths: list[Path], *, allow_large_local: bool, limit: int) -> None:
    if allow_large_local or platform.system() != "Darwin":
        return
    total = sum(path.stat().st_size for path in paths)
    if total > limit:
        raise SystemExit(
            "Refusing to migrate large semantic parquet inputs on Darwin without "
            f"--allow-large-local: {total:,} bytes > {limit:,}. Run this on the training VM."
        )


def _dual_contract_instructions(metadata: dict[str, Any]) -> str:
    chunk_ids = [str(chunk_id) for chunk_id in (metadata.get("chunk_labels") or {})]
    labels = [str(label) for label in metadata.get("label_space") or []]
    if len(labels) != 2 or not chunk_ids:
        raise ValueError("Semantic v4 rows require two labels and at least one chunk.")
    task_type = str(metadata.get("semantic_task_type") or "")
    if task_type == "global_comparison":
        final_schema = (
            f"After classifying all chunks, return exactly one of: {labels[0]}, {labels[1]}, same."
        )
    else:
        final_schema = (
            "Return one compact JSON object whose keys are exactly "
            f"{json.dumps(chunk_ids)} and whose values are either '{labels[0]}' or '{labels[1]}'."
        )

    example_chunk = chunk_ids[0]
    return (
        "Determine the semantic majority label for every named chunk in the context. "
        f"{final_schema}\n\n"
        "Delegate semantic classification with either of these equally valid child contracts:\n"
        "1. CHUNK_LABEL: send exactly one complete chunk, including its heading and every Record ID || Text line. "
        f"Ask for only the majority label, or a one-key map such as {{\"{example_chunk}\":\"{labels[0]}\"}}.\n"
        "2. RECORD_MAP: send at least eight complete Record ID || Text lines from exactly one chunk. "
        "Ask for a JSON or Python map containing every supplied record ID and its label. Missing records count as wrong.\n\n"
        "Do not send IDs without their text, alter record text, mix chunks in one child request, or request a chunk "
        "label from a partial chunk. Preserve complete record lines when slicing context. Use llm_query_batched for "
        "independent chunks or record batches, then aggregate verified child answers in the REPL."
    )


def _transform_row(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    metadata_value = row.get("metadata")
    metadata = json.loads(metadata_value) if isinstance(metadata_value, str) else dict(metadata_value or {})
    if metadata.get("derived_dataset") != DERIVED_DATASET_V3:
        return row, False

    metadata["derived_dataset"] = DERIVED_DATASET_V4
    metadata["semantic_delegation_version"] = "v4"
    metadata["semantic_child_contract"] = "verified_chunk_or_record_map"
    metadata["semantic_record_map_min_records"] = 8
    transformed = dict(row)
    transformed["id"] = str(row["id"]).replace("semantic-delegation-v3", "semantic-delegation-v4")
    transformed["prompt"] = _dual_contract_instructions(metadata)
    transformed["metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return transformed, True


def transform_split(input_path: Path, output_path: Path, *, batch_size: int) -> dict[str, int]:
    parquet = pq.ParquetFile(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    writer: pq.ParquetWriter | None = None
    counters: Counter[str] = Counter()
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            rows = []
            for row in pa.Table.from_batches([batch]).to_pylist():
                transformed, changed = _transform_row(row)
                rows.append(transformed)
                counters["rows"] += 1
                counters["semantic_v4_rows"] += int(changed)
                counters[f"family/{row.get('dataset')}"] += 1
            table = pa.Table.from_pylist(rows, schema=parquet.schema_arrow)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if counters["rows"] != parquet.metadata.num_rows:
        raise RuntimeError("Row count changed during v4 migration.")
    return dict(counters)


def build_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    allow_large_local: bool = False,
    local_guard_bytes: int = DEFAULT_LOCAL_GUARD_BYTES,
) -> dict[str, Any]:
    input_dir = _resolve(input_dir)
    output_dir = _resolve(output_dir)
    input_paths = [input_dir / f"{split}.parquet" for split in SPLITS]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing v3 parquet split(s): {missing}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    _guard_inputs(input_paths, allow_large_local=allow_large_local, limit=local_guard_bytes)

    split_stats = {
        split: transform_split(input_dir / f"{split}.parquet", output_dir / f"{split}.parquet", batch_size=batch_size)
        for split in SPLITS
    }
    manifest = {
        "derived_dataset": DERIVED_DATASET_V4,
        "source_dataset": DERIVED_DATASET_V3,
        "target_private_hf_repo": PRIVATE_HF_REPO,
        "semantic_child_contract": "verified_chunk_or_record_map",
        "semantic_record_map_min_records": 8,
        "splits": split_stats,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
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
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
