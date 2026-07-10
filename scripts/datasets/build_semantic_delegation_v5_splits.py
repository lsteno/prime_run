#!/usr/bin/env python3
"""Migrate semantic delegation v4 into natural, verifier-invisible child credit data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT_DIR = Path("data/beeg_agents_semantic_delegation_v4")
DEFAULT_OUTPUT_DIR = Path("data/beeg_agents_semantic_delegation_v5")
DEFAULT_BATCH_SIZE = 16
DEFAULT_LOCAL_GUARD_BYTES = 1_000_000_000
DERIVED_DATASET_V4 = "oolong_semantic_delegation_v4"
DERIVED_DATASET_V5 = "oolong_semantic_delegation_v5"
PRIVATE_HF_REPO = "lsteno/BEEG-agents-semantic-delegation-v5"
SPLITS = ("train", "eval")
_RECORD_LINE_RE = re.compile(
    r"^\s*Record ID:\s*(r\d{5})\s*\|\|\s*Text:\s*(.*?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


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


def _canonical_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _text_hash(value: str) -> str:
    return hashlib.sha256(_canonical_text(value).encode("utf-8")).hexdigest()


def _natural_task_prompt(metadata: dict[str, Any]) -> str:
    section_ids = [str(section_id) for section_id in (metadata.get("chunk_labels") or {})]
    labels = [str(label) for label in metadata.get("label_space") or []]
    if len(labels) != 2 or not section_ids:
        raise ValueError("Semantic v5 rows require two labels and at least one named section.")
    if str(metadata.get("semantic_task_type") or "") == "global_comparison":
        final_schema = f"Return exactly one of: {labels[0]}, {labels[1]}, same."
    else:
        final_schema = (
            "Return one compact JSON object whose keys are exactly "
            f"{json.dumps(section_ids)} and whose values are either '{labels[0]}' or '{labels[1]}'."
        )
    return (
        "Determine the semantic majority label for every named section in the context. "
        f"{final_schema}\n\n"
        "The sections are independent and can be analyzed in parallel with llm_query or llm_query_batched. "
        "Delegate semantic reading when useful, then synthesize the section-level results in the REPL."
    )


def _naturalize_context(
    context: str, metadata: dict[str, Any]
) -> tuple[str, dict[str, str], dict[str, str], int]:
    labels_by_id = {str(record_id).casefold(): str(label) for record_id, label in metadata["record_labels"].items()}
    chunks_by_id = {
        str(record_id).casefold(): str(chunk_id) for record_id, chunk_id in metadata["record_chunks"].items()
    }
    labels_by_hash: dict[str, str] = {}
    chunks_by_hash: dict[str, str] = {}
    record_count = 0

    def replace_record(match: re.Match[str]) -> str:
        nonlocal record_count
        record_id = match.group(1).casefold()
        text = _canonical_text(match.group(2))
        if record_id not in labels_by_id or record_id not in chunks_by_id:
            raise ValueError(f"Missing hidden gold metadata for {record_id}.")
        text_hash = _text_hash(text)
        label = labels_by_id[record_id]
        chunk = chunks_by_id[record_id]
        if text_hash in labels_by_hash and labels_by_hash[text_hash] != label:
            raise ValueError("Identical visible text has conflicting labels within one example.")
        if text_hash in chunks_by_hash and chunks_by_hash[text_hash] != chunk:
            raise ValueError("Identical visible text appears in multiple sections within one example.")
        labels_by_hash[text_hash] = label
        chunks_by_hash[text_hash] = chunk
        record_count += 1
        return f"- {text}"

    natural_context = _RECORD_LINE_RE.sub(replace_record, context)
    if record_count != len(labels_by_id):
        raise ValueError(f"Visible/gold record mismatch: context={record_count}, gold={len(labels_by_id)}")
    if "Record ID:" in natural_context or "|| Text:" in natural_context:
        raise ValueError("Visible record identifiers leaked into semantic v5 context.")
    return natural_context, labels_by_hash, chunks_by_hash, record_count


def _transform_row(row: dict[str, Any]) -> tuple[dict[str, Any], bool, int]:
    metadata_value = row.get("metadata")
    metadata = json.loads(metadata_value) if isinstance(metadata_value, str) else dict(metadata_value or {})
    if metadata.get("derived_dataset") != DERIVED_DATASET_V4:
        return row, False, 0

    context, labels_by_hash, chunks_by_hash, record_count = _naturalize_context(str(row["context"]), metadata)
    metadata.pop("record_labels", None)
    metadata.pop("record_chunks", None)
    metadata.pop("semantic_record_map_min_records", None)
    metadata["derived_dataset"] = DERIVED_DATASET_V5
    metadata["semantic_delegation_version"] = "v5"
    metadata["semantic_child_credit"] = "natural_visible_majority"
    metadata["record_labels_by_text_hash"] = labels_by_hash
    metadata["record_chunks_by_text_hash"] = chunks_by_hash

    transformed = dict(row)
    transformed["id"] = str(row["id"]).replace("semantic-delegation-v4", "semantic-delegation-v5")
    transformed["prompt"] = _natural_task_prompt(metadata)
    transformed["context"] = context
    transformed["metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return transformed, True, record_count


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
                transformed, changed, record_count = _transform_row(row)
                rows.append(transformed)
                counters["rows"] += 1
                counters["semantic_v5_rows"] += int(changed)
                counters["semantic_visible_records"] += record_count
                counters[f"family/{row.get('dataset')}"] += 1
            table = pa.Table.from_pylist(rows, schema=parquet.schema_arrow)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if counters["rows"] != parquet.metadata.num_rows:
        raise RuntimeError("Row count changed during v5 migration.")
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
        raise FileNotFoundError(f"Missing v4 parquet split(s): {missing}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    _guard_inputs(input_paths, allow_large_local=allow_large_local, limit=local_guard_bytes)

    split_stats = {
        split: transform_split(input_dir / f"{split}.parquet", output_dir / f"{split}.parquet", batch_size=batch_size)
        for split in SPLITS
    }
    manifest = {
        "derived_dataset": DERIVED_DATASET_V5,
        "source_dataset": DERIVED_DATASET_V4,
        "target_private_hf_repo": PRIVATE_HF_REPO,
        "semantic_child_credit": "natural_visible_majority",
        "visible_record_ids": False,
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
