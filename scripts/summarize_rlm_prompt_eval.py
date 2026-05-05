from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "reward",
    "used_recursion",
    "used_llm_subcalls",
    "used_rlm_subcalls",
    "num_subcalls",
    "num_llm_subcalls",
    "num_rlm_subcalls",
    "max_depth_reached",
    "prompt_tokens",
    "completion_tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired RLM prompt-eval runs against a fixed manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="PROMPT=PATH",
        help="Prompt variant and result directory or file. Repeat once per prompt.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/rlm_prompt_eval_summary"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Run spec must be PROMPT=PATH, got {spec!r}.")
    prompt, path = spec.split("=", 1)
    prompt = prompt.strip()
    if not prompt:
        raise ValueError(f"Run spec has empty prompt name: {spec!r}.")
    return prompt, Path(path).expanduser()


def json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".json" else []
    trace_files = sorted(path.glob("traces/step_*.json"))
    if trace_files:
        return trace_files
    step_files = sorted(path.rglob("step_*.json"))
    if step_files:
        return step_files
    return sorted(path.rglob("*.json"))


def number_or_nan(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def metric_value(record: dict[str, Any], name: str) -> float:
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for key in (name, f"rlm_rlvr/{name}", f"eval/{name}"):
            if key in metrics:
                return number_or_nan(metrics[key])
    if name in record:
        return number_or_nan(record[name])
    return math.nan


def metric_value_any(record: dict[str, Any], names: tuple[str, ...]) -> float:
    for name in names:
        value = metric_value(record, name)
        if not math.isnan(value):
            return value
    return math.nan


def parse_maybe_json(value: Any) -> Any:
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def segment_count(segment: dict[str, Any], count_keys: tuple[str, ...], token_id_keys: tuple[str, ...]) -> int:
    for key in count_keys:
        value = segment.get(key)
        if value not in (None, ""):
            return int(value)
    for key in token_id_keys:
        value = segment.get(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            return len(value)
    return 0


def token_totals_from_segments(segments: Any) -> tuple[float, float]:
    segments = parse_maybe_json(segments)
    if not isinstance(segments, (list, tuple, np.ndarray)):
        return math.nan, math.nan
    prompt_tokens = 0
    completion_tokens = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        prompt_tokens += segment_count(
            segment,
            ("prompt_token_count", "prompt_tokens"),
            ("prompt_ids",),
        )
        if segment.get("trainable_token_count") not in (None, "") or segment.get("trainable_completion_tokens") not in (None, ""):
            completion_tokens += segment_count(
                segment,
                ("trainable_token_count", "trainable_completion_tokens"),
                ("completion_ids",),
            )
        elif isinstance(segment.get("completion_mask"), (list, tuple, np.ndarray)):
            completion_tokens += sum(1 for item in segment["completion_mask"] if item)
        else:
            completion_tokens += segment_count(
                segment,
                ("completion_token_count", "completion_tokens"),
                ("completion_ids",),
            )
    return float(prompt_tokens), float(completion_tokens)


def token_totals(record: dict[str, Any]) -> tuple[float, float]:
    for key in ("segments", "rlm_segments"):
        prompt_tokens, completion_tokens = token_totals_from_segments(record.get(key))
        if not math.isnan(prompt_tokens) or not math.isnan(completion_tokens):
            return prompt_tokens, completion_tokens
    return math.nan, math.nan


def record_source_id(record: dict[str, Any]) -> str:
    for key in ("source_id", "id", "example_id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    sample = record.get("sample")
    if isinstance(sample, dict):
        value = sample.get("source_id")
        if value not in (None, ""):
            return str(value)
    metadata = record.get("sample_metadata")
    if isinstance(metadata, dict):
        value = metadata.get("source_id")
        if value not in (None, ""):
            return str(value)
    return ""


def live_trace_record(prompt: str, payload: dict[str, Any], json_path: Path) -> dict[str, Any] | None:
    if "segments" not in payload or "sample" not in payload:
        return None
    prompt_tokens, completion_tokens = token_totals(payload)
    status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
    return {
        "prompt_variant": str(payload.get("prompt_variant") or prompt),
        "source_id": record_source_id(payload),
        "reward": math.nan,
        "used_recursion": number_or_nan(status.get("used_recursion")),
        "used_llm_subcalls": number_or_nan(status.get("used_llm_subcalls")),
        "used_rlm_subcalls": number_or_nan(status.get("used_rlm_subcalls")),
        "num_subcalls": number_or_nan(status.get("num_subcalls")),
        "num_llm_subcalls": number_or_nan(status.get("num_llm_subcalls")),
        "num_rlm_subcalls": number_or_nan(status.get("num_rlm_subcalls")),
        "max_depth_reached": number_or_nan(status.get("max_depth_reached")),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "result_path": str(json_path),
        "rollout_index": 0,
        "error": None,
    }


def load_trace_records(prompt: str, path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for json_path in json_files(path):
        with open(json_path) as handle:
            payload = json.load(handle)
        rollouts = payload.get("rollouts") if isinstance(payload, dict) else None
        if not isinstance(rollouts, list):
            live_record = live_trace_record(prompt, payload, json_path) if isinstance(payload, dict) else None
            if live_record is not None:
                records.append(live_record)
            continue
        for rollout_index, record in enumerate(rollouts):
            if not isinstance(record, dict):
                continue
            source_id = record_source_id(record)
            prompt_tokens, completion_tokens = token_totals(record)
            records.append(
                {
                    "prompt_variant": prompt,
                    "source_id": source_id,
                    "reward": number_or_nan(record.get("reward")),
                    "used_recursion": metric_value(record, "used_recursion"),
                    "used_llm_subcalls": metric_value(record, "used_llm_subcalls"),
                    "used_rlm_subcalls": metric_value(record, "used_rlm_subcalls"),
                    "num_subcalls": metric_value(record, "num_subcalls"),
                    "num_llm_subcalls": metric_value(record, "num_llm_subcalls"),
                    "num_rlm_subcalls": metric_value(record, "num_rlm_subcalls"),
                    "max_depth_reached": metric_value_any(record, ("max_depth_reached", "max_depth")),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "result_path": str(json_path),
                    "rollout_index": rollout_index,
                    "error": record.get("error"),
                }
            )
    return records


def load_table_records(prompt: str, path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix == ".jsonl":
        frame = pd.read_json(path, lines=True)
    else:
        return []

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        source_id = str(row.get("source_id") or row.get("id") or row.get("example_id") or "")
        prompt_tokens = number_or_nan(row.get("prompt_tokens"))
        completion_tokens = number_or_nan(row.get("completion_tokens"))
        if math.isnan(prompt_tokens) or math.isnan(completion_tokens):
            segment_prompt_tokens, segment_completion_tokens = token_totals(dict(row))
            if math.isnan(prompt_tokens):
                prompt_tokens = segment_prompt_tokens
            if math.isnan(completion_tokens):
                completion_tokens = segment_completion_tokens
        rows.append(
            {
                "prompt_variant": prompt,
                "source_id": source_id,
                "reward": number_or_nan(row.get("reward")),
                "used_recursion": number_or_nan(row.get("used_recursion")),
                "used_llm_subcalls": number_or_nan(row.get("used_llm_subcalls")),
                "used_rlm_subcalls": number_or_nan(row.get("used_rlm_subcalls")),
                "num_subcalls": number_or_nan(row.get("num_subcalls")),
                "num_llm_subcalls": number_or_nan(row.get("num_llm_subcalls")),
                "num_rlm_subcalls": number_or_nan(row.get("num_rlm_subcalls")),
                "max_depth_reached": number_or_nan(row.get("max_depth_reached")),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "result_path": str(path),
                "rollout_index": int(row.get("rollout_index") or 0),
                "error": row.get("error"),
            }
        )
    return rows


def load_run(prompt: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Result path does not exist for {prompt}: {path}")
    records = load_table_records(prompt, path) if path.is_file() else []
    if not records:
        records = load_trace_records(prompt, path)
    if not records:
        raise ValueError(f"No supported result records found for {prompt}: {path}")
    return pd.DataFrame.from_records(records)


def paired_bootstrap(
    frame: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    pivot = frame.pivot_table(
        index="source_id",
        columns="prompt_variant",
        values="reward",
        aggfunc="mean",
    ).dropna(subset=[baseline, candidate])
    if pivot.empty:
        return {"n": 0, "mean_delta": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    deltas = (pivot[candidate] - pivot[baseline]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        draw = rng.choice(deltas, size=len(deltas), replace=True)
        means.append(float(np.mean(draw)))
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n": int(len(deltas)),
        "mean_delta": float(np.mean(deltas)),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def grouped_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(group_cols, dropna=False)[METRIC_COLUMNS]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(group_cols)
    )


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(display.columns)
    rows = display.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown(
    *,
    out_path: Path,
    aggregate: pd.DataFrame,
    paired: list[dict[str, float | str]],
    by_dataset: pd.DataFrame,
    by_task: pd.DataFrame,
) -> None:
    lines = [
        "# RLM Prompt Eval Summary",
        "",
        "## Aggregate",
        "",
        markdown_table(aggregate),
        "",
        "## Paired Delta vs default",
        "",
        markdown_table(pd.DataFrame(paired)),
        "",
        "## By Dataset",
        "",
        markdown_table(by_dataset),
        "",
        "## By Task",
        "",
        markdown_table(by_task),
        "",
    ]
    out_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if not args.run:
        raise ValueError("Provide at least one --run PROMPT=PATH argument.")

    manifest = pd.read_parquet(args.manifest)
    manifest["source_id"] = manifest["id"].astype(str)
    manifest_cols = [
        "source_id",
        "dataset",
        "task",
        "answer_type",
        "context_token_count",
        "source_dataset",
        "task_group",
        "reasoning_types",
        "repo",
    ]

    run_frames = [load_run(prompt, path) for prompt, path in map(parse_run_spec, args.run)]
    results = pd.concat(run_frames, ignore_index=True)
    results["source_id"] = results["source_id"].astype(str)
    merged = results.merge(manifest[manifest_cols], on="source_id", how="left", validate="many_to_one")

    missing = merged["dataset"].isna().sum()
    if missing:
        raise ValueError(f"{missing} result rows did not match manifest source_id values.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "samples.csv", index=False)

    aggregate = grouped_summary(merged, ["prompt_variant"])
    by_dataset = grouped_summary(merged, ["prompt_variant", "dataset"])
    by_task = grouped_summary(merged, ["prompt_variant", "dataset", "task"])
    by_answer_type = grouped_summary(merged, ["prompt_variant", "answer_type"])

    baseline = "default"
    paired = []
    for prompt in sorted(merged["prompt_variant"].unique()):
        if prompt == baseline:
            continue
        stats = paired_bootstrap(
            merged,
            baseline=baseline,
            candidate=prompt,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
        paired.append({"prompt_variant": prompt, **stats})

    aggregate.to_csv(args.out_dir / "aggregate.csv", index=False)
    by_dataset.to_csv(args.out_dir / "by_dataset.csv", index=False)
    by_task.to_csv(args.out_dir / "by_task.csv", index=False)
    by_answer_type.to_csv(args.out_dir / "by_answer_type.csv", index=False)
    pd.DataFrame(paired).to_csv(args.out_dir / "paired_deltas.csv", index=False)
    write_markdown(
        out_path=args.out_dir / "summary.md",
        aggregate=aggregate,
        paired=paired,
        by_dataset=by_dataset,
        by_task=by_task,
    )

    print(f"Wrote summary files to {args.out_dir}")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
