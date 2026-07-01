#!/usr/bin/env python3
"""Summarize RLM-Evals multi-model pass@1 results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


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


def _records(path: Path) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.rglob("results.jsonl"))
    if not files:
        raise FileNotFoundError(f"No results.jsonl files found under {path}")
    rows: list[dict[str, Any]] = []
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    row["_result_path"] = str(file)
                    rows.append(row)
    return rows


def _record_info(record: dict[str, Any]) -> dict[str, Any]:
    value = _json_loads(record.get("info"))
    return value if isinstance(value, dict) else {}


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    info = _record_info(record)
    value = _json_loads(info.get("metadata"))
    return value if isinstance(value, dict) else {}


def _source_id(record: dict[str, Any]) -> str:
    metadata = _record_metadata(record)
    for key in ("original_source_id", "source_id"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    info = _record_info(record)
    value = info.get("source_id") or record.get("source_id") or record.get("id") or record.get("example_id")
    return str(value).split("__rollout_", 1)[0] if value not in (None, "") else ""


def _segments(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = _json_loads(record.get("rlm_segments"))
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _segment_stats(record: dict[str, Any]) -> dict[str, float]:
    if record.get("rlm_segments") in (None, ""):
        return {
            "segment_total_tokens": _number(record.get("segment_total_tokens")),
            "segment_trainable_tokens": _number(record.get("segment_trainable_tokens")),
            "segment_plain_subcall_tokens": _number(record.get("segment_plain_subcall_tokens")),
            "empty_plain_subcalls": _number(record.get("empty_plain_subcalls")),
            "error_like_plain_subcalls": _number(record.get("error_like_plain_subcalls")),
        }
    prompt = completion = trainable = plain_tokens = 0
    empty_plain = error_like_plain = 0
    for segment in _segments(record):
        segment_prompt = int(segment.get("prompt_token_count") or segment.get("prompt_tokens") or 0)
        segment_completion = int(segment.get("completion_token_count") or segment.get("completion_tokens") or 0)
        prompt += segment_prompt
        completion += segment_completion
        if segment.get("is_trainable_rlm_turn"):
            trainable += segment_prompt + segment_completion
        if segment.get("kind") == "plain_query":
            plain_tokens += segment_prompt + segment_completion
            text = str(segment.get("response_text") or "").strip()
            if not text:
                empty_plain += 1
            if not text or any(marker in text for marker in ("NOT_FOUND", "RESOURCE_EXHAUSTED", "429", "Error:")):
                error_like_plain += 1
    return {
        "segment_total_tokens": float(prompt + completion),
        "segment_trainable_tokens": float(trainable),
        "segment_plain_subcall_tokens": float(plain_tokens),
        "empty_plain_subcalls": float(empty_plain),
        "error_like_plain_subcalls": float(error_like_plain),
    }


def _manifest(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _frame(label: str, path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, record in enumerate(_records(path)):
        reward_correctness = _metric(record, "reward_correctness")
        reward = _number(record.get("reward"))
        correct = reward_correctness if not math.isnan(reward_correctness) else (1.0 if reward > 0.0 else 0.0)
        stats = _segment_stats(record)
        rows.append(
            {
                "model": label,
                "source_id": _source_id(record),
                "row_order": index,
                "reward": reward,
                "correct": correct,
                "reward_correctness": reward_correctness,
                "judge_score": _metric(record, "judge_score"),
                "oolong_pairs_precision": _metric(record, "oolong_pairs_precision"),
                "oolong_pairs_recall": _metric(record, "oolong_pairs_recall"),
                "oolong_pairs_f1": _metric(record, "oolong_pairs_f1"),
                "oolong_pairs_predicted_count": _metric(record, "oolong_pairs_predicted_count"),
                "oolong_pairs_expected_count": _metric(record, "oolong_pairs_expected_count"),
                "used_repl": _metric(record, "used_repl"),
                "used_llm_subcalls": _metric(record, "used_llm_subcalls"),
                "used_rlm_subcalls": _metric(record, "used_rlm_subcalls"),
                "num_llm_subcalls": _metric(record, "num_llm_subcalls"),
                "num_rlm_subcalls": _metric(record, "num_rlm_subcalls"),
                "max_depth_reached": _metric(record, "max_depth_reached"),
                "cost_total_tokens": _metric(record, "cost_total_tokens"),
                "cost_trainable_tokens": _metric(record, "cost_trainable_tokens"),
                "cost_plain_subcall_tokens": _metric(record, "cost_plain_subcall_tokens"),
                "missing_final": _metric(record, "missing_final"),
                "hit_max_turn_without_final": _metric(record, "hit_max_turn_without_final"),
                "used_forced_finalize_prompt": _metric(record, "used_forced_finalize_prompt"),
                "stop_condition": record.get("stop_condition"),
                "has_error": bool(record.get("has_error") or record.get("error")),
                "is_truncated": bool(record.get("is_truncated")),
                "result_path": record.get("_result_path"),
                **stats,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No result rows for {label}: {path}")
    return frame.merge(manifest.drop(columns=["metadata"], errors="ignore"), on="source_id", how="left", validate="many_to_one")


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else math.nan


def _summaries(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key, strict=False))
        row.update(
            {
                "examples": int(group["source_id"].nunique()),
                "rollouts": int(len(group)),
                "pass@1": _mean(group["correct"]),
                "primary_score": _mean(
                    group["oolong_pairs_f1"]
                    if "dataset" in group and str(group["dataset"].iloc[0]) == "oolong_pairs"
                    else group["correct"]
                ),
                "avg_reward": _mean(group["reward"]),
                "judge_score_mean": _mean(group["judge_score"]),
                "oolong_pairs_f1": _mean(group["oolong_pairs_f1"]),
                "oolong_pairs_precision": _mean(group["oolong_pairs_precision"]),
                "oolong_pairs_recall": _mean(group["oolong_pairs_recall"]),
                "mean_oolong_pairs_predicted_count": _mean(group["oolong_pairs_predicted_count"]),
                "mean_oolong_pairs_expected_count": _mean(group["oolong_pairs_expected_count"]),
                "llm_subcall_usage_rate": _mean(group["used_llm_subcalls"]),
                "rlm_subcall_usage_rate": _mean(group["used_rlm_subcalls"]),
                "mean_num_llm_subcalls": _mean(group["num_llm_subcalls"]),
                "mean_num_rlm_subcalls": _mean(group["num_rlm_subcalls"]),
                "mean_max_depth_reached": _mean(group["max_depth_reached"]),
                "used_repl_rate": _mean(group["used_repl"]),
                "missing_final_rate": _mean(group["missing_final"]),
                "max_turn_without_final_rate": _mean(group["hit_max_turn_without_final"]),
                "forced_finalize_rate": _mean(group["used_forced_finalize_prompt"]),
                "error_rate": _mean(group["has_error"].astype(float)),
                "truncated_rate": _mean(group["is_truncated"].astype(float)),
                "mean_total_tokens": _mean(group["cost_total_tokens"].fillna(group["segment_total_tokens"])),
                "mean_trainable_tokens": _mean(group["cost_trainable_tokens"].fillna(group["segment_trainable_tokens"])),
                "mean_plain_subcall_tokens": _mean(group["cost_plain_subcall_tokens"].fillna(group["segment_plain_subcall_tokens"])),
                "empty_plain_subcall_rate": float(group["empty_plain_subcalls"].sum() / max(group["num_llm_subcalls"].sum(), 1.0)),
                "error_like_plain_subcall_rate": float(
                    group["error_like_plain_subcalls"].sum() / max(group["num_llm_subcalls"].sum(), 1.0)
                ),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-result", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-examples", type=int, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(args.manifest)
    frames = []
    model_paths = {}
    for item in args.model_result:
        label, path = item.split("=", 1)
        model_paths[label] = path
        frames.append(_frame(label, Path(path), manifest))
    samples = pd.concat(frames, ignore_index=True)

    for model, group in samples.groupby("model"):
        if len(group) != args.expected_examples:
            raise RuntimeError(f"{model}: expected {args.expected_examples} rows, got {len(group)}")
        bad = group.groupby("source_id").size()
        bad = bad[bad != 1]
        if len(bad):
            raise RuntimeError(f"{model}: expected one rollout per source; bad={bad.head(10).to_dict()}")

    aggregate = _summaries(samples, ["model"])
    by_benchmark = _summaries(samples, ["model", "dataset"])
    by_task = _summaries(samples, ["model", "dataset", "task"])
    by_stop = samples.groupby(["model", "stop_condition"], dropna=False).size().reset_index(name="count")

    samples.to_csv(args.out_dir / "per_rollout.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate.csv", index=False)
    by_benchmark.to_csv(args.out_dir / "by_benchmark.csv", index=False)
    by_task.to_csv(args.out_dir / "by_task.csv", index=False)
    by_stop.to_csv(args.out_dir / "by_stop_condition.csv", index=False)
    payload = {
        "expected_examples": args.expected_examples,
        "model_paths": model_paths,
        "aggregate": aggregate.to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report = "\n\n".join(
        [
            "# RLM-Evals Paper-Style Pass@1 Comparison",
            "## Aggregate",
            _markdown_table(aggregate),
            "## By Benchmark",
            _markdown_table(by_benchmark),
            "## Stop Conditions",
            _markdown_table(by_stop),
        ]
    )
    (args.out_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.out_dir / "summary.json"), "models": list(model_paths)}, indent=2))


if __name__ == "__main__":
    main()
