#!/usr/bin/env python3
"""Summarize rank/LR ablation outputs and optional eval result files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def number(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def metric(record: dict[str, Any], key: str) -> float:
    for source in (record, record.get("state"), record.get("info")):
        if isinstance(source, dict) and key in source:
            return number(source[key])
    return math.nan


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return sum(clean) / len(clean) if clean else math.nan


def summarize_eval(path: Path) -> dict[str, float | int]:
    rows = read_jsonl(path)
    rewards = [number(row.get("reward")) for row in rows]
    correct = [1.0 if reward > 0.0 else 0.0 for reward in rewards if not math.isnan(reward)]
    return {
        "eval_rows": len(rows),
        "avg_reward": mean(rewards),
        "avg_correct": mean(correct),
        "mean_num_llm_subcalls": mean([metric(row, "num_llm_subcalls") for row in rows]),
        "llm_subcall_usage_rate": mean([metric(row, "used_llm_subcalls") for row in rows]),
        "mean_total_tokens": mean([metric(row, "cost_total_tokens") for row in rows]),
        "mean_plain_subcall_tokens": mean([metric(row, "cost_plain_subcall_tokens") for row in rows]),
    }


def latest_step(path: Path) -> int | None:
    candidates = []
    for parent_name in ("checkpoints", "weights"):
        parent = path / parent_name
        if not parent.exists():
            continue
        for child in parent.iterdir():
            if child.is_dir() and child.name.startswith("step_"):
                try:
                    candidates.append(int(child.name.removeprefix("step_")))
                except ValueError:
                    pass
    return max(candidates) if candidates else None


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def find_eval_files(eval_root: Path | None, run_id: str, output_dir: Path) -> list[Path]:
    roots = []
    if eval_root is not None:
        roots.append(eval_root / run_id)
        roots.append(eval_root)
    roots.append(output_dir)
    seen: set[Path] = set()
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("results.jsonl")):
            if path not in seen:
                seen.add(path)
                files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/rlm_rlvr/ablation_rank_lr/manifest.csv"))
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/rlm_rank_lr_ablation_summary/summary.csv"))
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in manifest:
        output_dir = Path(item["output_dir"])
        eval_files = find_eval_files(args.eval_root, item["run_id"], output_dir)
        eval_summary: dict[str, float | int] = defaultdict(lambda: math.nan)
        if eval_files:
            eval_summary = summarize_eval(eval_files[-1])
            eval_summary["eval_file"] = eval_files[-1].as_posix()
        else:
            eval_summary["eval_file"] = ""
            eval_summary["eval_rows"] = 0
        rows.append(
            {
                "index": item["index"],
                "run_id": item["run_id"],
                "rank": item["rank"],
                "alpha": item["alpha"],
                "lr": item["lr"],
                "latest_step": latest_step(output_dir),
                **eval_summary,
            }
        )

    fieldnames = [
        "index",
        "run_id",
        "rank",
        "alpha",
        "lr",
        "latest_step",
        "eval_rows",
        "avg_reward",
        "avg_correct",
        "mean_num_llm_subcalls",
        "llm_subcall_usage_rate",
        "mean_total_tokens",
        "mean_plain_subcall_tokens",
        "eval_file",
    ]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
