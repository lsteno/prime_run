from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from datasets import load_dataset


DEFAULT_QUOTAS = {
    "frames_rag": 24,
    "CU_P": 12,
    "CU_DFA": 9,
    "DRA_T1": 6,
    "TASK_TYPE.MOST_FREQ": 8,
    "TASK_TYPE.SECOND_MOST_FREQ": 8,
    "TASK_TYPE.RELATIVE_FREQ": 7,
    "TASK_TYPE.NUMERIC_ONE_CLASS": 6,
    "TASK_TYPE.LEAST_FREQ": 6,
    "TASK_TYPE.REPRESENTED_N_TIMES": 5,
    "TASK_TYPE.CROSS_MOST_LABEL_BY_USER": 3,
    "TASK_TYPE.CROSS_COUNT_LABEL_USER": 3,
    "TASK_TYPE.LABEL_HISTOGRAM": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the fixed 100-sample stratified RLM prompt-eval manifest."
    )
    parser.add_argument("--dataset-id", default="lsteno/BEEG-agents")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/rlm_rlvr/prompt_eval_100_seed42.parquet"))
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-revision", default=None)
    return parser.parse_args()


def parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def stable_key(seed: int, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in (seed, *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_reasoning_types(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, list):
        parts = [str(item).strip() for item in value]
    else:
        parts = [part.strip() for part in str(value).split("|")]
    parts = sorted(part for part in parts if part)
    return " | ".join(parts) if parts else "unknown"


def secondary_key(row: dict[str, Any]) -> str:
    metadata = parse_metadata(row.get("metadata"))
    dataset = str(row.get("dataset") or "")
    if dataset == "oolong":
        source_dataset = str(metadata.get("source_dataset") or "unknown_source")
        task_group = str(metadata.get("task_group") or "unknown_group")
        return f"{source_dataset}::{task_group}"
    if dataset == "frames":
        return normalize_reasoning_types(metadata.get("reasoning_types"))
    if dataset == "longcodeu":
        return str(metadata.get("repo") or "unknown_repo")
    return "unknown"


def balanced_select(
    rows: list[dict[str, Any]],
    *,
    quota: int,
    seed: int,
    task: str,
    key_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    if len(rows) < quota:
        raise ValueError(f"Task {task!r} has only {len(rows)} rows; quota is {quota}.")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)

    for key, group_rows in groups.items():
        group_rows.sort(
            key=lambda row: stable_key(
                seed,
                task,
                key,
                row.get("id", ""),
                row.get("_source_row_index", ""),
            )
        )

    group_keys = sorted(groups, key=lambda key: stable_key(seed, task, key))
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < quota:
        key = group_keys[cursor % len(group_keys)]
        if groups[key]:
            selected.append(groups[key].pop(0))
        cursor += 1
        if cursor > len(group_keys) * (quota + len(group_keys) + 1):
            remaining = [row for group_rows in groups.values() for row in group_rows]
            remaining.sort(
                key=lambda row: stable_key(
                    seed,
                    task,
                    row.get("id", ""),
                    row.get("_source_row_index", ""),
                )
            )
            selected.extend(remaining[: quota - len(selected)])
            break

    selected.sort(
        key=lambda row: stable_key(
            seed,
            "final_order",
            task,
            row.get("id", ""),
            row.get("_source_row_index", ""),
        )
    )
    return selected[:quota]


def row_to_record(row: dict[str, Any], manifest_index: int) -> dict[str, Any]:
    metadata = parse_metadata(row.get("metadata"))
    return {
        "manifest_index": manifest_index,
        "id": str(row.get("id") or f"row-{row.get('_source_row_index')}"),
        "dataset": str(row.get("dataset") or ""),
        "task": str(row.get("task") or ""),
        "prompt": str(row.get("prompt") or ""),
        "context": str(row.get("context") or ""),
        "answer": str(row.get("answer") or ""),
        "answer_type": str(row.get("answer_type") or ""),
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "context_token_count": int(row.get("context_token_count") or 0),
        "source_dataset": str(metadata.get("source_dataset") or ""),
        "task_group": str(metadata.get("task_group") or ""),
        "reasoning_types": normalize_reasoning_types(metadata.get("reasoning_types")),
        "repo": str(metadata.get("repo") or ""),
        "source_row_index": int(row.get("_source_row_index") or 0),
    }


def main() -> None:
    args = parse_args()
    load_kwargs: dict[str, Any] = {"split": args.split}
    if args.dataset_revision:
        load_kwargs["revision"] = args.dataset_revision

    if args.dataset_config:
        dataset = load_dataset(args.dataset_id, args.dataset_config, **load_kwargs)
    else:
        dataset = load_dataset(args.dataset_id, **load_kwargs)

    rows = [dict(row, _source_row_index=index) for index, row in enumerate(dataset)]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task") or "")].append(row)

    selected: list[dict[str, Any]] = []
    for task, quota in DEFAULT_QUOTAS.items():
        selected.extend(
            balanced_select(
                by_task.get(task, []),
                quota=quota,
                seed=args.seed,
                task=task,
                key_fn=secondary_key,
            )
        )

    selected.sort(
        key=lambda row: stable_key(
            args.seed,
            "manifest",
            row.get("dataset", ""),
            row.get("task", ""),
            row.get("id", ""),
            row.get("_source_row_index", ""),
        )
    )
    records = [row_to_record(row, index) for index, row in enumerate(selected)]
    frame = pd.DataFrame.from_records(records)

    if len(frame) != sum(DEFAULT_QUOTAS.values()):
        raise AssertionError(f"Expected {sum(DEFAULT_QUOTAS.values())} rows, got {len(frame)}.")
    if frame["id"].duplicated().any():
        duplicated = frame.loc[frame["id"].duplicated(), "id"].tolist()
        raise AssertionError(f"Duplicate ids in manifest: {duplicated[:5]}")
    if (frame[["prompt", "context", "answer"]] == "").any().any():
        raise AssertionError("Manifest contains empty prompt, context, or answer values.")

    actual_quotas = Counter(frame["task"])
    if dict(actual_quotas) != DEFAULT_QUOTAS:
        raise AssertionError(f"Quota mismatch: expected {DEFAULT_QUOTAS}, got {dict(actual_quotas)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    print(f"Wrote {len(frame)} rows to {args.out}")
    print("Dataset counts:")
    for name, count in Counter(frame["dataset"]).most_common():
        print(f"  {name}: {count}")
    print("Task counts:")
    for task, count in actual_quotas.most_common():
        print(f"  {task}: {count}")


if __name__ == "__main__":
    main()
