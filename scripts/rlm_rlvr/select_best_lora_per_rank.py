#!/usr/bin/env python3
"""Select one completed LoRA adapter per rank from the RLM rank/LR sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


RANK_RE = re.compile(r"-r(?P<rank>\d+)-a(?P<alpha>\d+)-lr(?P<lr>.+?)-s(?P<steps>\d+)")


def _number(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _adapter_step(output_dir: Path, expected_step: int) -> Path | None:
    candidate = output_dir / "run_default" / "broadcasts" / f"step_{expected_step}"
    required = ("STABLE", "adapter_config.json", "adapter_model.safetensors")
    if all((candidate / name).is_file() for name in required):
        return candidate
    return None


def _final_summary_metrics(output_dir: Path) -> dict[str, float | str]:
    summaries = sorted(output_dir.rglob("final_summary.json"), key=lambda path: path.stat().st_mtime)
    best: dict[str, float | str] = {
        "selection_metric": math.nan,
        "selection_metric_name": "eval/rlm_rlvr_eval/pass@1",
        "final_summary_path": "",
    }
    for path in summaries:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pass1 = _number(data.get("eval/rlm_rlvr_eval/pass@1"))
        avg1 = _number(data.get("eval/rlm_rlvr_eval/avg@1"))
        reward = _number(data.get("reward/all/mean"))
        candidates = [
            ("eval/rlm_rlvr_eval/pass@1", pass1),
            ("eval/rlm_rlvr_eval/avg@1", avg1),
            ("reward/all/mean", reward),
        ]
        for name, value in candidates:
            if not math.isnan(value):
                if math.isnan(float(best["selection_metric"])) or value > float(best["selection_metric"]):
                    best = {
                        "selection_metric": value,
                        "selection_metric_name": name,
                        "final_summary_path": str(path),
                    }
                break
    return best


def _parse_run_id(run_id: str) -> dict[str, Any]:
    match = RANK_RE.search(run_id)
    if match is None:
        raise ValueError(f"Could not parse rank/alpha/lr/steps from run_id={run_id!r}")
    return {
        "rank": int(match.group("rank")),
        "alpha": int(match.group("alpha")),
        "lr": match.group("lr"),
        "steps": int(match.group("steps")),
    }


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_overrides(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(rank): str(run_id) for rank, run_id in payload.items()}


def select_models(
    manifest_path: Path,
    *,
    root_dir: Path,
    output_path: Path,
    expected_step: int,
    repo_prefix: str,
    overrides_path: Path | None = None,
) -> dict[str, Any]:
    rows = _load_manifest(manifest_path)
    overrides = _load_overrides(overrides_path)
    candidates_by_rank: dict[int, list[dict[str, Any]]] = {}
    all_candidates: list[dict[str, Any]] = []

    for row in rows:
        run_id = row.get("run_id", "")
        if not run_id or "bal35f40v1" not in run_id:
            continue
        parsed = _parse_run_id(run_id)
        output_dir = root_dir / row["output_dir"]
        adapter_path = _adapter_step(output_dir, expected_step)
        if adapter_path is None:
            continue
        metrics = _final_summary_metrics(output_dir)
        candidate = {
            **parsed,
            "run_id": run_id,
            "output_dir": str(output_dir),
            "adapter_path": str(adapter_path),
            "adapter_step": expected_step,
            "repo_id": f"{repo_prefix}-r{parsed['rank']}-a{parsed['alpha']}-lr{parsed['lr']}-s{expected_step}-bal35f40v1-lora",
            **metrics,
        }
        candidates_by_rank.setdefault(parsed["rank"], []).append(candidate)
        all_candidates.append(candidate)

    selected: list[dict[str, Any]] = []
    missing: list[int] = []
    for rank in (4, 16, 64):
        options = candidates_by_rank.get(rank, [])
        if rank in overrides:
            override = overrides[rank]
            matches = [option for option in options if option["run_id"] == override]
            if not matches:
                raise RuntimeError(f"Override for rank {rank} has no verified step_{expected_step} adapter: {override}")
            chosen = matches[0]
        elif options:
            chosen = max(
                options,
                key=lambda item: (
                    -1.0 if math.isnan(float(item["selection_metric"])) else float(item["selection_metric"]),
                    item["lr"],
                ),
            )
        else:
            missing.append(rank)
            continue
        selected.append(chosen)

    if missing:
        raise RuntimeError(f"No verified final step_{expected_step} LoRA adapter for rank(s): {missing}")

    payload = {
        "expected_step": expected_step,
        "selection_rule": "best final completed adapter per rank by eval/rlm_rlvr_eval/pass@1; fallback avg@1 then reward/all/mean",
        "selected": selected,
        "candidates": all_candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/rlm_rlvr/ablation_rank_lr/manifest.csv"))
    parser.add_argument("--root-dir", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=150)
    parser.add_argument("--repo-prefix", default="lsteno/qwen3-rlm-depth1")
    parser.add_argument("--overrides", type=Path, default=None)
    args = parser.parse_args()
    payload = select_models(
        args.manifest,
        root_dir=args.root_dir,
        output_path=args.out,
        expected_step=args.expected_step,
        repo_prefix=args.repo_prefix,
        overrides_path=args.overrides,
    )
    print(json.dumps({"selected": payload["selected"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
