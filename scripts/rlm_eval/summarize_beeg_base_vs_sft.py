from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired BEEG evals with pass@1/pass@k.")
    parser.add_argument("--base", type=Path, required=True, help="Base results.jsonl or run directory.")
    parser.add_argument("--sft", type=Path, required=True, help="SFT results.jsonl or run directory.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default="lsteno/BEEG-agents")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollouts-per-example", type=int, default=5)
    return parser.parse_args()


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
    candidates = sorted(path.rglob("results.jsonl"), key=lambda candidate: candidate.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No results.jsonl found under {path}")
    return candidates


def _records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    result_paths = _result_files(path)
    for result_path in result_paths:
        with result_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    value["_result_path"] = str(result_path)
                    rows.append(value)
    if not rows:
        raise ValueError(f"No result rows under {path}")
    return rows


def _record_info(record: dict[str, Any]) -> dict[str, Any]:
    info = _json_loads(record.get("info"))
    return info if isinstance(info, dict) else {}


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    info = _record_info(record)
    metadata = info.get("metadata")
    metadata = _json_loads(metadata)
    return metadata if isinstance(metadata, dict) else {}


def _record_source_id(record: dict[str, Any]) -> str:
    metadata = _record_metadata(record)
    for key in ("original_source_id", "source_id"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)

    info = _record_info(record)
    value = info.get("source_id")
    if value not in (None, ""):
        source_id = str(value)
        if "__rollout_" in source_id:
            return source_id.split("__rollout_", 1)[0]
        return source_id

    value = record.get("source_id") or record.get("id") or record.get("example_id")
    if value not in (None, ""):
        source_id = str(value)
        if "__rollout_" in source_id:
            return source_id.split("__rollout_", 1)[0]
        return source_id
    return ""


def _record_rollout_index(record: dict[str, Any], row_order: int, rollouts_per_example: int) -> int:
    metadata = _record_metadata(record)
    for key in ("original_rollout_index", "rollout_index"):
        value = metadata.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    for key in ("rollout_index",):
        value = record.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return -1


def _load_manifest(dataset_id: str, split: str, seed: int) -> pd.DataFrame:
    dataset = load_dataset(dataset_id, split=split)
    if seed >= 0:
        dataset = dataset.shuffle(seed=seed)
    rows = []
    for example_id, row in enumerate(dataset):
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            metadata = _json_loads(metadata)
        if not isinstance(metadata, dict):
            metadata = {}
        rows.append(
            {
                "example_id": str(example_id),
                "source_id": str(row.get("id") or row.get("example_id") or example_id),
                "dataset": row.get("dataset"),
                "task": row.get("task"),
                "answer_type": row.get("answer_type"),
                "context_token_count": row.get("context_token_count"),
                "source_dataset": metadata.get("source_dataset"),
                "task_group": metadata.get("task_group"),
                "reasoning_types": json.dumps(metadata.get("reasoning_types"), ensure_ascii=False)
                if metadata.get("reasoning_types") is not None
                else None,
                "repo": metadata.get("repo"),
            }
        )
    return pd.DataFrame(rows)


def _segment_count(segment: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = segment.get(key)
        if value not in (None, ""):
            return int(value)
    return 0


def _segments(record: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("rlm_segments", "segments"):
        value = _json_loads(record.get(key))
        if isinstance(value, list):
            return [segment for segment in value if isinstance(segment, dict)]
    return []


def _segment_stats(record: dict[str, Any]) -> dict[str, float]:
    segments = _segments(record)
    prompt = 0
    completion = 0
    trainable = 0
    plain = 0
    empty_plain = 0
    error_like_plain = 0
    for segment in segments:
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
            if not text or "NOT_FOUND" in text or "429" in text or "RESOURCE_EXHAUSTED" in text:
                error_like_plain += 1
    return {
        "segment_prompt_tokens": float(prompt),
        "segment_completion_tokens": float(completion),
        "segment_total_tokens": float(prompt + completion),
        "segment_trainable_tokens": float(trainable),
        "segment_plain_subcall_tokens": float(plain),
        "empty_plain_subcalls": float(empty_plain),
        "error_like_plain_subcalls": float(error_like_plain),
    }


def _frame(label: str, path: Path, manifest: pd.DataFrame, rollouts_per_example: int) -> pd.DataFrame:
    rows = []
    for idx, record in enumerate(_records(path)):
        example_id = str(record.get("example_id") if record.get("example_id") is not None else "")
        if not example_id:
            example_id = str(idx // rollouts_per_example)
        source_id = _record_source_id(record)
        rollout_index = _record_rollout_index(record, idx, rollouts_per_example)
        stats = _segment_stats(record)
        rows.append(
            {
                "model": label,
                "example_id": example_id,
                "source_id": source_id,
                "rollout_index": rollout_index,
                "row_order": idx,
                "reward": _number(record.get("reward")),
                "correct": 1.0 if _number(record.get("reward")) > 0.0 else 0.0,
                "has_error": bool(record.get("has_error") or record.get("error")),
                "is_truncated": bool(record.get("is_truncated")),
                "completion_len": _number(record.get("completion_len")),
                "used_repl": _metric(record, "used_repl"),
                "used_recursion": _metric(record, "used_recursion"),
                "used_llm_subcalls": _metric(record, "used_llm_subcalls"),
                "used_rlm_subcalls": _metric(record, "used_rlm_subcalls"),
                "num_subcalls": _metric(record, "num_subcalls"),
                "num_llm_subcalls": _metric(record, "num_llm_subcalls"),
                "num_rlm_subcalls": _metric(record, "num_rlm_subcalls"),
                "max_depth_reached": _metric(record, "max_depth_reached"),
                "cost_prompt_tokens": _metric(record, "cost_prompt_tokens"),
                "cost_completion_tokens": _metric(record, "cost_completion_tokens"),
                "cost_total_tokens": _metric(record, "cost_total_tokens"),
                "cost_trainable_tokens": _metric(record, "cost_trainable_tokens"),
                "cost_plain_subcall_tokens": _metric(record, "cost_plain_subcall_tokens"),
                **stats,
                "result_path": record.get("_result_path"),
            }
        )
    frame = pd.DataFrame(rows)
    missing_source = frame["source_id"].isna() | (frame["source_id"] == "")
    if missing_source.any():
        frame.loc[missing_source, "source_id"] = frame.loc[missing_source, "example_id"].map(str)
    missing_rollout = frame["rollout_index"] < 0
    if missing_rollout.any():
        frame = frame.sort_values(["source_id", "row_order"]).reset_index(drop=True)
        fallback_rollout_index = frame.groupby("source_id").cumcount()
        frame.loc[missing_rollout, "rollout_index"] = fallback_rollout_index[missing_rollout]
    return frame.merge(manifest.drop(columns=["example_id"]), on="source_id", how="left", validate="many_to_one")


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else math.nan


def _summaries(frame: pd.DataFrame, group_cols: list[str], rollouts_per_example: int) -> pd.DataFrame:
    group_rows = []
    for key, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        ordered = group.sort_values(["source_id", "rollout_index", "row_order"])
        per_example = ordered.groupby("source_id", dropna=False)["correct"].agg(list)
        pass1 = per_example.map(lambda values: float(bool(values and values[0] > 0))).mean()
        passk = per_example.map(lambda values: float(any(value > 0 for value in values[:rollouts_per_example]))).mean()
        complete_per_example = per_example.map(lambda values: len(values) >= rollouts_per_example)
        row = dict(zip(group_cols, key, strict=False))
        row.update(
            {
                "examples": int(per_example.shape[0]),
                "complete_examples": int(complete_per_example.sum()),
                "rollouts": int(group.shape[0]),
                "pass@1": float(pass1),
                "avg@1": _mean(group["correct"]),
                f"pass@{rollouts_per_example}": float(passk),
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
                "mean_num_rlm_subcalls": _mean(group["num_rlm_subcalls"]),
                "truncated_rate": _mean(group["is_truncated"].astype(float)),
                "error_rate": _mean(group["has_error"].astype(float)),
            }
        )
        group_rows.append(row)
    return pd.DataFrame(group_rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(value))
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(args.dataset_id, args.split, args.seed)
    base = _frame("base", args.base, manifest, args.rollouts_per_example)
    sft = _frame("sft", args.sft, manifest, args.rollouts_per_example)
    samples = pd.concat([base, sft], ignore_index=True)

    aggregate = _summaries(samples, ["model"], args.rollouts_per_example)
    by_dataset = _summaries(samples, ["model", "dataset"], args.rollouts_per_example)
    by_task = _summaries(samples, ["model", "dataset", "task"], args.rollouts_per_example)

    aggregate.to_csv(args.out_dir / "aggregate.csv", index=False)
    by_dataset.to_csv(args.out_dir / "by_dataset.csv", index=False)
    by_task.to_csv(args.out_dir / "by_task.csv", index=False)
    samples.to_csv(args.out_dir / "per_rollout.csv", index=False)

    pivot = aggregate.set_index("model")
    deltas = {}
    if {"base", "sft"}.issubset(set(pivot.index)):
        for metric in ("pass@1", "avg@1", f"pass@{args.rollouts_per_example}", "mean_num_llm_subcalls"):
            deltas[f"delta_{metric}"] = float(pivot.loc["sft", metric] - pivot.loc["base", metric])
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "split": args.split,
                "seed": args.seed,
                "rollouts_per_example": args.rollouts_per_example,
                "deltas": deltas,
                "aggregate": aggregate.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    lines = [
        "# BEEG Base vs SFT Depth-1 Eval",
        "",
        "## Aggregate",
        "",
        _markdown_table(aggregate),
        "",
        "## Delta",
        "",
        json.dumps(deltas, indent=2),
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
    (args.out_dir / "summary.md").write_text("\n".join(lines))
    print((args.out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
