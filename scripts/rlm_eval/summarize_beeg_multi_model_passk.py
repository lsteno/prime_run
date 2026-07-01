#!/usr/bin/env python3
"""Summarize multi-model BEEG pass@k eval results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


def _json_loads(value: Any) -> Any:
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _number(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _metric(record: dict[str, Any], key: str) -> float:
    metrics = _json_loads(record.get("metrics"))
    if isinstance(metrics, dict):
        for candidate in (key, f"rlm_rlvr/{key}", f"eval/{key}", f"{key}_metric"):
            if candidate in metrics:
                return _number(metrics[candidate])
    return _number(record.get(key))


def _result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    candidates = sorted(path.rglob("results.jsonl"), key=lambda item: item.as_posix())
    if not candidates:
        raise FileNotFoundError(f"No results.jsonl found under {path}")
    return candidates


def _records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in _result_files(path):
        with result_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    row["_result_path"] = str(result_path)
                    rows.append(row)
    return rows


def _record_info(record: dict[str, Any]) -> dict[str, Any]:
    info = _json_loads(record.get("info"))
    return info if isinstance(info, dict) else {}


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    info = _record_info(record)
    metadata = _json_loads(info.get("metadata"))
    return metadata if isinstance(metadata, dict) else {}


def _source_id(record: dict[str, Any]) -> str:
    metadata = _record_metadata(record)
    for key in ("original_source_id", "source_id"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    info = _record_info(record)
    value = info.get("source_id") or record.get("source_id") or record.get("id") or record.get("example_id")
    if value not in (None, ""):
        text = str(value)
        return text.split("__rollout_", 1)[0]
    return ""


def _rollout_index(record: dict[str, Any], row_order: int, rollouts_per_example: int) -> int:
    metadata = _record_metadata(record)
    for key in ("original_rollout_index", "rollout_index"):
        value = metadata.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return row_order % rollouts_per_example


def _segments(record: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("rlm_segments", "segments"):
        value = _json_loads(record.get(key))
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _segment_count(segment: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = segment.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _segment_stats(record: dict[str, Any]) -> dict[str, float]:
    prompt = 0
    completion = 0
    trainable = 0
    plain = 0
    empty_plain = 0
    error_like_plain = 0
    for segment in _segments(record):
        segment_prompt = _segment_count(segment, ("prompt_token_count", "prompt_tokens"))
        segment_completion = _segment_count(segment, ("completion_token_count", "completion_tokens"))
        prompt += segment_prompt
        completion += segment_completion
        if segment.get("is_trainable_rlm_turn"):
            trainable += segment_prompt + segment_completion
        if segment.get("kind") == "plain_query":
            plain += segment_prompt + segment_completion
            text = str(segment.get("response_text") or "").strip()
            if not text:
                empty_plain += 1
            if not text or any(marker in text for marker in ("NOT_FOUND", "RESOURCE_EXHAUSTED", "429", "Error:")):
                error_like_plain += 1
    return {
        "segment_total_tokens": float(prompt + completion),
        "segment_trainable_tokens": float(trainable),
        "segment_plain_subcall_tokens": float(plain),
        "empty_plain_subcalls": float(empty_plain),
        "error_like_plain_subcalls": float(error_like_plain),
    }


def _manifest(dataset_id: str, split: str, seed: int) -> pd.DataFrame:
    dataset = load_dataset(dataset_id, split=split)
    if seed >= 0:
        dataset = dataset.shuffle(seed=seed)
    rows = []
    for idx, row in enumerate(dataset):
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            metadata = _json_loads(metadata)
        if not isinstance(metadata, dict):
            metadata = {}
        rows.append(
            {
                "source_id": str(row.get("id") or row.get("example_id") or idx),
                "dataset": row.get("dataset"),
                "task": row.get("task"),
                "answer_type": row.get("answer_type"),
                "context_token_count": row.get("context_token_count"),
                "source_dataset": metadata.get("source_dataset"),
                "task_group": metadata.get("task_group"),
            }
        )
    return pd.DataFrame(rows)


def _frame(label: str, path: Path, manifest: pd.DataFrame, rollouts_per_example: int) -> pd.DataFrame:
    rows = []
    for idx, record in enumerate(_records(path)):
        stats = _segment_stats(record)
        reward_correctness = _metric(record, "reward_correctness")
        reward = _number(record.get("reward"))
        correct = reward_correctness if not math.isnan(reward_correctness) else (1.0 if reward > 0.0 else 0.0)
        rows.append(
            {
                "model": label,
                "source_id": _source_id(record),
                "rollout_index": _rollout_index(record, idx, rollouts_per_example),
                "row_order": idx,
                "reward": reward,
                "correct": correct,
                "reward_correctness": reward_correctness,
                "has_error": bool(record.get("has_error") or record.get("error")),
                "is_truncated": bool(record.get("is_truncated")),
                "completion_len": _number(record.get("completion_len")),
                "used_repl": _metric(record, "used_repl"),
                "used_llm_subcalls": _metric(record, "used_llm_subcalls"),
                "used_rlm_subcalls": _metric(record, "used_rlm_subcalls"),
                "num_llm_subcalls": _metric(record, "num_llm_subcalls"),
                "num_rlm_subcalls": _metric(record, "num_rlm_subcalls"),
                "max_depth_reached": _metric(record, "max_depth_reached"),
                "cost_total_tokens": _metric(record, "cost_total_tokens"),
                "cost_trainable_tokens": _metric(record, "cost_trainable_tokens"),
                "cost_plain_subcall_tokens": _metric(record, "cost_plain_subcall_tokens"),
                **stats,
                "result_path": record.get("_result_path"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No result rows for {label}: {path}")
    missing_source = frame["source_id"].isna() | (frame["source_id"] == "")
    if missing_source.any():
        frame.loc[missing_source, "source_id"] = frame.loc[missing_source, "row_order"].map(
            lambda value: str(int(value) // rollouts_per_example)
        )
    return frame.merge(manifest, on="source_id", how="left", validate="many_to_one")


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else math.nan


def _pass_at_k(values: list[float], k: int) -> float:
    n = len(values)
    if n <= 0:
        return math.nan
    correct = sum(1 for value in values if value > 0)
    if k <= 1:
        return correct / n
    if correct <= 0:
        return 0.0
    incorrect = n - correct
    if incorrect < k:
        return 1.0
    return 1.0 - math.comb(incorrect, k) / math.comb(n, k)


def _summaries(frame: pd.DataFrame, group_cols: list[str], rollouts_per_example: int) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        ordered = group.sort_values(["source_id", "rollout_index", "row_order"])
        per_example = ordered.groupby("source_id", dropna=False)["correct"].agg(list)
        row = dict(zip(group_cols, key, strict=False))
        pass_values = {}
        for k in (1, 5, rollouts_per_example):
            pass_values[f"pass@{k}"] = float(per_example.map(lambda values, kk=k: _pass_at_k(values, kk)).mean())
        complete = per_example.map(lambda values: len(values) >= rollouts_per_example)
        row.update(
            {
                "examples": int(per_example.shape[0]),
                "complete_examples": int(complete.sum()),
                "rollouts": int(group.shape[0]),
                **pass_values,
                "avg_correct": _mean(group["correct"]),
                "avg_reward": _mean(group["reward"]),
                "mean_num_llm_subcalls": _mean(group["num_llm_subcalls"]),
                "llm_subcall_usage_rate": _mean(group["used_llm_subcalls"]),
                "empty_plain_subcall_rate": float(group["empty_plain_subcalls"].sum() / max(group["num_llm_subcalls"].sum(), 1.0)),
                "error_like_plain_subcall_rate": float(
                    group["error_like_plain_subcalls"].sum() / max(group["num_llm_subcalls"].sum(), 1.0)
                ),
                "mean_total_tokens": _mean(group["cost_total_tokens"].fillna(group["segment_total_tokens"])),
                "mean_plain_subcall_tokens": _mean(
                    group["cost_plain_subcall_tokens"].fillna(group["segment_plain_subcall_tokens"])
                ),
                "mean_trainable_tokens": _mean(group["cost_trainable_tokens"].fillna(group["segment_trainable_tokens"])),
                "used_repl_rate": _mean(group["used_repl"]),
                "used_rlm_subcalls_rate": _mean(group["used_rlm_subcalls"]),
                "truncated_rate": _mean(group["is_truncated"].astype(float)),
                "error_rate": _mean(group["has_error"].astype(float)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(value))
    lines = ["| " + " | ".join(display.columns) + " |", "| " + " | ".join("---" for _ in display.columns) + " |"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _validate_rows(frame: pd.DataFrame, expected_examples: int, rollouts_per_example: int) -> None:
    for model, group in frame.groupby("model"):
        expected_rows = expected_examples * rollouts_per_example
        if len(group) != expected_rows:
            raise RuntimeError(f"{model}: expected {expected_rows} rows, got {len(group)}")
        counts = group.groupby("source_id").size()
        bad = counts[counts != rollouts_per_example]
        if len(counts) != expected_examples or not bad.empty:
            preview = bad.head(10).to_dict()
            raise RuntimeError(
                f"{model}: expected {expected_examples} examples with {rollouts_per_example} rollouts each; bad={preview}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-result", action="append", required=True, help="LABEL=PATH. Repeat once per model.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default="lsteno/BEEG-agents")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-examples", type=int, default=452)
    parser.add_argument("--rollouts-per-example", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(args.dataset_id, args.split, args.seed)
    frames = []
    model_paths: dict[str, str] = {}
    for item in args.model_result:
        if "=" not in item:
            raise ValueError(f"--model-result must be LABEL=PATH, got {item}")
        label, path = item.split("=", 1)
        model_paths[label] = path
        frames.append(_frame(label, Path(path), manifest, args.rollouts_per_example))
    samples = pd.concat(frames, ignore_index=True)
    _validate_rows(samples, args.expected_examples, args.rollouts_per_example)

    aggregate = _summaries(samples, ["model"], args.rollouts_per_example)
    by_dataset = _summaries(samples, ["model", "dataset"], args.rollouts_per_example)
    by_task = _summaries(samples, ["model", "dataset", "task"], args.rollouts_per_example)

    samples.to_csv(args.out_dir / "per_rollout.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate.csv", index=False)
    by_dataset.to_csv(args.out_dir / "by_dataset.csv", index=False)
    by_task.to_csv(args.out_dir / "by_task.csv", index=False)
    payload = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "seed": args.seed,
        "expected_examples": args.expected_examples,
        "rollouts_per_example": args.rollouts_per_example,
        "model_paths": model_paths,
        "aggregate": aggregate.to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# BEEG Final Model Pass@10 Comparison",
        "",
        "## Aggregate",
        "",
        _markdown_table(aggregate),
        "",
        "## By Dataset",
        "",
        _markdown_table(by_dataset),
        "",
        "## By Task",
        "",
        _markdown_table(by_task),
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
