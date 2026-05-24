#!/usr/bin/env python3
"""Compute example-level uncertainty for BEEG pass@10 evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import socket
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


METRICS = ("pass@1", "pass@5", "pass@10", "avg_correct")
MODEL_LABELS = {
    "base_qwen4b": "Base Qwen3 4B",
    "lora_r4_lr1em4": "LoRA r4, lr 1e-4",
    "lora_r16_lr1em4": "LoRA r16, lr 1e-4",
    "lora_r64_lr1em5": "LoRA r64, lr 1e-5",
    "fullft": "Full fine-tune, lr 5e-6",
}
DATASET_LABELS = {
    "frames": "Deep search",
    "oolong": "Aggregation",
    "longcodeu": "Long code understanding",
}


def git_sha(repo: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short=7", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-rollout-csv", type=Path, required=True)
    parser.add_argument("--aggregate-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="base_qwen4b")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_523)
    parser.add_argument("--expected-examples", type=int, default=452)
    parser.add_argument("--rollouts-per-example", type=int, default=10)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args()


def as_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def pct(value: float) -> str:
    return "" if math.isnan(value) else f"{100.0 * value:.1f}"


def pp(value: float) -> str:
    return "" if math.isnan(value) else f"{100.0 * value:.1f}"


def compact_tokens(value: float) -> str:
    if math.isnan(value):
        return ""
    if abs(value) >= 1000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def percentile_ci(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return quantile(ordered, 0.025), quantile(ordered, 0.975)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def load_rows(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row["model"]
            source_id = row["source_id"]
            rows[model][source_id].append(
                {
                    "rollout_index": int(row["rollout_index"]),
                    "row_order": int(row["row_order"]),
                    "correct": as_float(row["correct"]),
                    "cost_total_tokens": as_float(row.get("cost_total_tokens") or row.get("segment_total_tokens")),
                    "num_llm_subcalls": as_float(row["num_llm_subcalls"]),
                    "used_llm_subcalls": as_float(row["used_llm_subcalls"]),
                    "is_truncated": 1.0 if str(row["is_truncated"]).lower() == "true" else 0.0,
                    "dataset": row["dataset"],
                    "task": row["task"],
                }
            )
    return rows


def validate_rows(rows: dict[str, dict[str, list[dict[str, Any]]]], args: argparse.Namespace) -> list[str]:
    models = sorted(rows)
    if args.baseline not in rows:
        raise RuntimeError(f"Baseline {args.baseline!r} not found. Models: {models}")
    baseline_ids = set(rows[args.baseline])
    if len(baseline_ids) != args.expected_examples:
        raise RuntimeError(f"{args.baseline}: expected {args.expected_examples} examples, got {len(baseline_ids)}")
    for model in models:
        ids = set(rows[model])
        if ids != baseline_ids:
            missing = sorted(baseline_ids - ids)[:5]
            extra = sorted(ids - baseline_ids)[:5]
            raise RuntimeError(f"{model}: source_id mismatch; missing={missing}, extra={extra}")
        for source_id, rollouts in rows[model].items():
            if len(rollouts) != args.rollouts_per_example:
                raise RuntimeError(f"{model}/{source_id}: expected {args.rollouts_per_example} rollouts, got {len(rollouts)}")
    return models


def build_example_metrics(
    rows: dict[str, dict[str, list[dict[str, Any]]]], rollouts_per_example: int
) -> dict[str, dict[str, dict[str, Any]]]:
    per_model: dict[str, dict[str, dict[str, Any]]] = {}
    for model, by_source in rows.items():
        per_model[model] = {}
        for source_id, rollouts in by_source.items():
            ordered = sorted(rollouts, key=lambda item: (item["rollout_index"], item["row_order"]))
            correctness = [float(item["correct"]) for item in ordered]
            tokens = [float(item["cost_total_tokens"]) for item in ordered]
            llm_calls = [float(item["num_llm_subcalls"]) for item in ordered]
            per_model[model][source_id] = {
                "dataset": ordered[0]["dataset"],
                "pass@1": 1.0 if any(value > 0 for value in correctness[:1]) else 0.0,
                "pass@5": 1.0 if any(value > 0 for value in correctness[: min(5, rollouts_per_example)]) else 0.0,
                "pass@10": 1.0 if any(value > 0 for value in correctness[:rollouts_per_example]) else 0.0,
                "avg_correct": mean(correctness[:rollouts_per_example]),
                "mean_tokens": mean(tokens),
                "mean_llm_calls": mean(llm_calls),
            }
    return per_model


def stratified_ids(per_model: dict[str, dict[str, dict[str, Any]]], baseline: str) -> dict[str, list[str]]:
    strata: dict[str, list[str]] = defaultdict(list)
    for source_id, metrics in per_model[baseline].items():
        strata[metrics["dataset"]].append(source_id)
    return {dataset: sorted(ids) for dataset, ids in sorted(strata.items())}


def bootstrap_draws(strata: dict[str, list[str]], samples: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    draws: list[list[str]] = []
    for _ in range(samples):
        sample: list[str] = []
        for ids in strata.values():
            sample.extend(ids[rng.randrange(len(ids))] for _ in ids)
        draws.append(sample)
    return draws


def metric_mean(per_model: dict[str, dict[str, dict[str, Any]]], model: str, ids: list[str], metric: str) -> float:
    return mean([float(per_model[model][source_id][metric]) for source_id in ids])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_uncertainty(
    per_model: dict[str, dict[str, dict[str, Any]]], models: list[str], all_ids: list[str], draws: list[list[str]]
) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        for metric in METRICS:
            boots = [metric_mean(per_model, model, draw, metric) for draw in draws]
            low, high = percentile_ci(boots)
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "metric": metric,
                    "estimate": metric_mean(per_model, model, all_ids, metric),
                    "ci_low": low,
                    "ci_high": high,
                    "estimate_pct": pct(metric_mean(per_model, model, all_ids, metric)),
                    "ci_low_pct": pct(low),
                    "ci_high_pct": pct(high),
                    "n_examples": len(all_ids),
                    "bootstrap_samples": len(draws),
                }
            )
    return rows


def paired_deltas(
    per_model: dict[str, dict[str, dict[str, Any]]],
    models: list[str],
    baseline: str,
    all_ids: list[str],
    draws: list[list[str]],
) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        if model == baseline:
            continue
        for metric in METRICS:
            point = mean([per_model[model][source_id][metric] - per_model[baseline][source_id][metric] for source_id in all_ids])
            boots = [
                mean([per_model[model][source_id][metric] - per_model[baseline][source_id][metric] for source_id in draw])
                for draw in draws
            ]
            low, high = percentile_ci(boots)
            row = {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "baseline": baseline,
                "metric": metric,
                "delta": point,
                "ci_low": low,
                "ci_high": high,
                "delta_pp": pp(point),
                "ci_low_pp": pp(low),
                "ci_high_pp": pp(high),
                "n_examples": len(all_ids),
            }
            if metric.startswith("pass@"):
                candidate_only = baseline_only = both_correct = both_wrong = 0
                for source_id in all_ids:
                    candidate = bool(per_model[model][source_id][metric])
                    base = bool(per_model[baseline][source_id][metric])
                    if candidate and base:
                        both_correct += 1
                    elif candidate and not base:
                        candidate_only += 1
                    elif base and not candidate:
                        baseline_only += 1
                    else:
                        both_wrong += 1
                row.update(
                    {
                        "candidate_only": candidate_only,
                        "baseline_only": baseline_only,
                        "both_correct": both_correct,
                        "both_wrong": both_wrong,
                    }
                )
            else:
                row.update({"candidate_only": "", "baseline_only": "", "both_correct": "", "both_wrong": ""})
            rows.append(row)
    return rows


def trained_pair_deltas(
    per_model: dict[str, dict[str, dict[str, Any]]], models: list[str], all_ids: list[str], draws: list[list[str]]
) -> list[dict[str, Any]]:
    pairs = [
        ("lora_r64_lr1em5", "lora_r4_lr1em4"),
        ("lora_r64_lr1em5", "fullft"),
        ("lora_r4_lr1em4", "fullft"),
    ]
    rows = []
    for left, right in pairs:
        if left not in models or right not in models:
            continue
        for metric in METRICS:
            point = mean([per_model[left][source_id][metric] - per_model[right][source_id][metric] for source_id in all_ids])
            boots = [mean([per_model[left][source_id][metric] - per_model[right][source_id][metric] for source_id in draw]) for draw in draws]
            low, high = percentile_ci(boots)
            rows.append(
                {
                    "left_model": left,
                    "right_model": right,
                    "metric": metric,
                    "delta_left_minus_right": point,
                    "ci_low": low,
                    "ci_high": high,
                    "delta_pp": pp(point),
                    "ci_low_pp": pp(low),
                    "ci_high_pp": pp(high),
                    "n_examples": len(all_ids),
                }
            )
    return rows


def task_family_uncertainty(
    per_model: dict[str, dict[str, dict[str, Any]]], models: list[str], strata: dict[str, list[str]], samples: int, seed: int
) -> list[dict[str, Any]]:
    rows = []
    for offset, (dataset, ids) in enumerate(strata.items()):
        draws = bootstrap_draws({dataset: ids}, samples, seed + offset + 1000)
        for model in models:
            for metric in METRICS:
                boots = [metric_mean(per_model, model, draw, metric) for draw in draws]
                low, high = percentile_ci(boots)
                rows.append(
                    {
                        "dataset": dataset,
                        "task_family": DATASET_LABELS.get(dataset, dataset),
                        "model": model,
                        "model_label": MODEL_LABELS.get(model, model),
                        "metric": metric,
                        "estimate": metric_mean(per_model, model, ids, metric),
                        "ci_low": low,
                        "ci_high": high,
                        "estimate_pct": pct(metric_mean(per_model, model, ids, metric)),
                        "ci_low_pct": pct(low),
                        "ci_high_pct": pct(high),
                        "n_examples": len(ids),
                    }
                )
    return rows


def token_distribution(raw_rows: dict[str, dict[str, list[dict[str, Any]]]], models: list[str]) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        by_dataset: dict[str, list[float]] = defaultdict(list)
        by_dataset["overall"] = []
        for rollouts in raw_rows[model].values():
            for rollout in rollouts:
                dataset = rollout["dataset"]
                tokens = float(rollout["cost_total_tokens"])
                by_dataset["overall"].append(tokens)
                by_dataset[dataset].append(tokens)
        for dataset, values in sorted(by_dataset.items()):
            ordered = sorted(values)
            nonzero = [value for value in ordered if value > 0]
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "dataset": dataset,
                    "task_family": "Overall" if dataset == "overall" else DATASET_LABELS.get(dataset, dataset),
                    "n_rollouts": len(ordered),
                    "mean_tokens": mean(ordered),
                    "median_tokens": median(ordered) if ordered else math.nan,
                    "q1_tokens": quantile(ordered, 0.25),
                    "q3_tokens": quantile(ordered, 0.75),
                    "p90_tokens": quantile(ordered, 0.90),
                    "p95_tokens": quantile(ordered, 0.95),
                    "p99_tokens": quantile(ordered, 0.99),
                    "max_tokens": ordered[-1] if ordered else math.nan,
                    "nonzero_rate": len(nonzero) / len(ordered) if ordered else math.nan,
                    "mean_tokens_fmt": compact_tokens(mean(ordered)),
                    "median_tokens_fmt": compact_tokens(median(ordered) if ordered else math.nan),
                    "iqr_tokens_fmt": f"{compact_tokens(quantile(ordered, 0.25))}--{compact_tokens(quantile(ordered, 0.75))}",
                    "p95_tokens_fmt": compact_tokens(quantile(ordered, 0.95)),
                }
            )
    return rows


def check_against_aggregate(path: Path | None, aggregate_rows: list[dict[str, Any]]) -> list[str]:
    if path is None:
        return []
    observed = {(row["model"], row["metric"]): float(row["estimate"]) for row in aggregate_rows}
    warnings = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            for metric in METRICS:
                reported = float(row[metric])
                computed = observed[(model, metric)]
                if abs(reported - computed) > 1e-12:
                    warnings.append(f"{model}/{metric}: aggregate={reported}, recomputed={computed}")
    return warnings


def markdown_summary(
    aggregate_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
) -> str:
    def row_for(rows: list[dict[str, Any]], **match: str) -> dict[str, Any]:
        for row in rows:
            if all(row[key] == value for key, value in match.items()):
                return row
        raise KeyError(match)

    models = ["base_qwen4b", "lora_r4_lr1em4", "lora_r16_lr1em4", "lora_r64_lr1em5", "fullft"]
    lines = [
        "# BEEG Pass@10 Uncertainty Analysis",
        "",
        "All intervals are 95% percentile bootstrap intervals over held-out examples.",
        "",
        "## Main Accuracy",
        "",
        "| Model | p@1 | p@5 | p@10 | Avg. correct |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        cells = []
        for metric in METRICS:
            row = row_for(aggregate_rows, model=model, metric=metric)
            cells.append(f"{row['estimate_pct']} [{row['ci_low_pct']}, {row['ci_high_pct']}]")
        lines.append(f"| {MODEL_LABELS.get(model, model)} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Paired Delta vs Base",
            "",
            "| Model | Δp@1 | Δp@5 | Δp@10 | Δavg. correct |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in [m for m in models if m != "base_qwen4b"]:
        cells = []
        for metric in METRICS:
            row = row_for(delta_rows, model=model, metric=metric)
            cells.append(f"{row['delta_pp']} [{row['ci_low_pp']}, {row['ci_high_pp']}]")
        lines.append(f"| {MODEL_LABELS.get(model, model)} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Token Cost Distribution",
            "",
            "| Model | Mean | Median [IQR] | p95 | Nonzero rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in models:
        row = row_for(token_rows, model=model, dataset="overall")
        lines.append(
            f"| {MODEL_LABELS.get(model, model)} | {row['mean_tokens_fmt']} | "
            f"{row['median_tokens_fmt']} [{row['iqr_tokens_fmt']}] | {row['p95_tokens_fmt']} | "
            f"{100.0 * row['nonzero_rate']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Task-Family p@10",
            "",
            "| Family | Base | LoRA r4 | LoRA r64 | Full FT |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset in ["frames", "oolong", "longcodeu"]:
        cells = []
        for model in ["base_qwen4b", "lora_r4_lr1em4", "lora_r64_lr1em5", "fullft"]:
            row = row_for(task_rows, dataset=dataset, model=model, metric="pass@10")
            cells.append(f"{row['estimate_pct']} [{row['ci_low_pct']}, {row['ci_high_pct']}]")
        lines.append(f"| {DATASET_LABELS.get(dataset, dataset)} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "script": str(Path(__file__).resolve()),
        "repo_root": str(args.repo_root.resolve()),
        "git_sha": git_sha(args.repo_root),
        "hostname": socket.gethostname(),
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "params": {
            "per_rollout_csv": str(args.per_rollout_csv.resolve()),
            "aggregate_csv": str(args.aggregate_csv.resolve()) if args.aggregate_csv else None,
            "baseline": args.baseline,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "expected_examples": args.expected_examples,
            "rollouts_per_example": args.rollouts_per_example,
        },
        "completed": False,
    }
    meta_path = args.out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    raw_rows = load_rows(args.per_rollout_csv)
    models = validate_rows(raw_rows, args)
    model_order = [model for model in ["base_qwen4b", "lora_r4_lr1em4", "lora_r16_lr1em4", "lora_r64_lr1em5", "fullft"] if model in models]
    model_order.extend(model for model in models if model not in model_order)
    per_model = build_example_metrics(raw_rows, args.rollouts_per_example)
    strata = stratified_ids(per_model, args.baseline)
    all_ids = [source_id for ids in strata.values() for source_id in ids]
    draws = bootstrap_draws(strata, args.bootstrap_samples, args.seed)

    aggregate_rows = aggregate_uncertainty(per_model, model_order, all_ids, draws)
    warnings = check_against_aggregate(args.aggregate_csv, aggregate_rows)
    if warnings:
        raise RuntimeError("Aggregate validation failed:\n" + "\n".join(warnings))
    delta_rows = paired_deltas(per_model, model_order, args.baseline, all_ids, draws)
    trained_delta_rows = trained_pair_deltas(per_model, model_order, all_ids, draws)
    task_rows = task_family_uncertainty(per_model, model_order, strata, args.bootstrap_samples, args.seed)
    token_rows = token_distribution(raw_rows, model_order)

    write_csv(
        args.out_dir / "main_metrics_ci.csv",
        aggregate_rows,
        ["model", "model_label", "metric", "estimate", "ci_low", "ci_high", "estimate_pct", "ci_low_pct", "ci_high_pct", "n_examples", "bootstrap_samples"],
    )
    write_csv(
        args.out_dir / "paired_deltas_vs_base.csv",
        delta_rows,
        [
            "model",
            "model_label",
            "baseline",
            "metric",
            "delta",
            "ci_low",
            "ci_high",
            "delta_pp",
            "ci_low_pp",
            "ci_high_pp",
            "n_examples",
            "candidate_only",
            "baseline_only",
            "both_correct",
            "both_wrong",
        ],
    )
    write_csv(
        args.out_dir / "trained_pair_deltas.csv",
        trained_delta_rows,
        ["left_model", "right_model", "metric", "delta_left_minus_right", "ci_low", "ci_high", "delta_pp", "ci_low_pp", "ci_high_pp", "n_examples"],
    )
    write_csv(
        args.out_dir / "task_family_metrics_ci.csv",
        task_rows,
        ["dataset", "task_family", "model", "model_label", "metric", "estimate", "ci_low", "ci_high", "estimate_pct", "ci_low_pct", "ci_high_pct", "n_examples"],
    )
    write_csv(
        args.out_dir / "token_distribution.csv",
        token_rows,
        [
            "model",
            "model_label",
            "dataset",
            "task_family",
            "n_rollouts",
            "mean_tokens",
            "median_tokens",
            "q1_tokens",
            "q3_tokens",
            "p90_tokens",
            "p95_tokens",
            "p99_tokens",
            "max_tokens",
            "nonzero_rate",
            "mean_tokens_fmt",
            "median_tokens_fmt",
            "iqr_tokens_fmt",
            "p95_tokens_fmt",
        ],
    )
    (args.out_dir / "summary.md").write_text(
        markdown_summary(aggregate_rows, delta_rows, token_rows, task_rows),
        encoding="utf-8",
    )

    meta["completed"] = True
    meta["duration_s"] = round(time.time() - t0, 2)
    meta["outputs"] = sorted(path.name for path in args.out_dir.iterdir())
    meta["models"] = model_order
    meta["strata"] = {dataset: len(ids) for dataset, ids in strata.items()}
    meta["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote uncertainty analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
