from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log running prime eval results.jsonl files to W&B.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--project", default="rlm-rlvr-evals")
    parser.add_argument("--name", required=True)
    parser.add_argument("--expected-rollouts", type=int, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--started-after", type=float, default=0.0)
    parser.add_argument(
        "--aggregate-files",
        action="store_true",
        help="Aggregate every results.jsonl under results-root modified after started-after.",
    )
    return parser.parse_args()


def _number(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _metric(record: dict[str, Any], key: str) -> float:
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for candidate in (key, f"rlm_rlvr/{key}", f"eval/{key}", f"{key}_metric"):
            if candidate in metrics:
                return _number(metrics[candidate])
    return _number(record.get(key))


def _result_files(root: Path, started_after: float) -> list[Path]:
    candidates = [
        path
        for path in root.rglob("results.jsonl")
        if path.is_file() and path.stat().st_mtime >= started_after
    ]
    return sorted(candidates, key=lambda path: str(path))


def _latest_results_file(root: Path, started_after: float) -> Path | None:
    candidates = _result_files(root, started_after)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except FileNotFoundError:
        pass
    return rows


def _read_many_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_jsonl(path):
            row["_result_path"] = str(path)
            rows.append(row)
    return rows


def _json_loads(value: Any) -> Any:
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _source_key(record: dict[str, Any]) -> str:
    info = _json_loads(record.get("info"))
    if isinstance(info, dict):
        metadata = info.get("metadata")
        if isinstance(metadata, str):
            metadata = _json_loads(metadata)
        if isinstance(metadata, dict):
            for key in ("original_source_id", "source_id"):
                value = metadata.get(key)
                if value not in (None, ""):
                    return str(value)
        value = info.get("source_id")
        if value not in (None, ""):
            source_id = str(value)
            if "__rollout_" in source_id:
                return source_id.split("__rollout_", 1)[0]
            return source_id
    for key in ("example_id", "source_id", "id"):
        value = record.get(key)
        if value not in (None, ""):
            source_id = str(value)
            if "__rollout_" in source_id:
                return source_id.split("__rollout_", 1)[0]
            return source_id
    return ""


def _summarize(rows: list[dict[str, Any]], expected_rollouts: int) -> dict[str, float]:
    rewards = [_number(row.get("reward")) for row in rows]
    rewards = [value for value in rewards if not math.isnan(value)]
    completed = len(rows)
    groups: dict[str, list[float]] = {}
    for row in rows:
        key = _source_key(row)
        if not key:
            continue
        reward = _number(row.get("reward"))
        if not math.isnan(reward):
            groups.setdefault(key, []).append(reward)

    pass_at_1_values = [1.0 if values and values[0] > 0.0 else 0.0 for values in groups.values()]
    pass_at_k_values = [1.0 if any(value > 0.0 for value in values) else 0.0 for values in groups.values()]
    target_rollouts = max(1, int(expected_rollouts / max(len(groups), 1))) if groups else 1
    completed_examples = sum(1 for values in groups.values() if len(values) >= target_rollouts)

    def mean_metric(key: str) -> float:
        values = [_metric(row, key) for row in rows]
        values = [value for value in values if not math.isnan(value)]
        return float(sum(values) / len(values)) if values else math.nan

    return {
        "completed_rollouts": float(completed),
        "expected_rollouts": float(expected_rollouts),
        "progress": float(completed / expected_rollouts) if expected_rollouts else 0.0,
        "seen_examples": float(len(groups)),
        "completed_examples": float(completed_examples),
        "avg_reward": float(sum(rewards) / len(rewards)) if rewards else math.nan,
        "pass_at_1_so_far": float(sum(pass_at_1_values) / len(pass_at_1_values)) if pass_at_1_values else math.nan,
        "pass_at_k_so_far": float(sum(pass_at_k_values) / len(pass_at_k_values)) if pass_at_k_values else math.nan,
        "used_llm_subcalls": mean_metric("used_llm_subcalls"),
        "num_llm_subcalls": mean_metric("num_llm_subcalls"),
        "cost_total_tokens": mean_metric("cost_total_tokens"),
        "error_rate": mean_metric("has_error"),
    }


def main() -> None:
    args = _parse_args()
    import wandb

    run = wandb.init(
        project=args.project,
        name=args.name,
        config={
            "run_label": args.run_label,
            "expected_rollouts": args.expected_rollouts,
            "results_root": str(args.results_root),
        },
    )
    step = 0
    last_completed = -1.0
    try:
        while True:
            if args.aggregate_files:
                result_files = _result_files(args.results_root, args.started_after)
                rows = _read_many_jsonl(result_files)
                result_file_label = ",".join(str(path) for path in result_files[-8:])
            else:
                results_file = _latest_results_file(args.results_root, args.started_after)
                rows = _read_jsonl(results_file) if results_file is not None else []
                result_file_label = str(results_file) if results_file is not None else ""
            if rows:
                metrics = _summarize(rows, args.expected_rollouts)
                metrics = {f"{args.run_label}/{key}": value for key, value in metrics.items()}
                metrics[f"{args.run_label}/result_files"] = result_file_label
                completed = metrics[f"{args.run_label}/completed_rollouts"]
                if completed != last_completed:
                    run.log(metrics, step=step)
                    last_completed = completed
                    step += 1
            if args.stop_file.exists():
                break
            time.sleep(args.interval_seconds)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
