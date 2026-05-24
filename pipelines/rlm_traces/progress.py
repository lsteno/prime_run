from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path("outputs/rlm_traces/prime-gpt54-vertex-flash-lite-sft-full")
DEFAULT_VARIANT = "sanjaya_text_depth1_llm_only_v1"
DEFAULT_TOTAL = 302


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show progress and ETA for an rlm_traces generation run.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Trace run output directory.")
    parser.add_argument("--variant", default=None, help="Prompt variant subdirectory. Defaults to run_config.json's first variant.")
    parser.add_argument("--total", type=int, default=None, help="Total examples. Defaults to run_config.json num_examples.")
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Do not write/read a small progress snapshot for recent-rate ETA.",
    )
    parser.add_argument(
        "--process-filter",
        default="pipelines/rlm_traces/run.py",
        help="Substring used to detect active trace-generation processes.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def fmt_rate(records: int, seconds: float | None) -> str:
    if not seconds or seconds <= 0 or records <= 0:
        return "unknown"
    per_hour = records / seconds * 3600.0
    per_record = seconds / records
    return f"{per_hour:.1f} records/hour ({fmt_duration(per_record)} per record)"


def active_processes(process_filter: str) -> list[str]:
    try:
        result = subprocess.run(
            ["pgrep", "-af", process_filter],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return []
    this_pid = str(subprocess.run(["sh", "-c", "echo $$"], text=True, capture_output=True).stdout).strip()
    lines = []
    for line in result.stdout.splitlines():
        if "progress.py" in line:
            continue
        if this_pid and line.startswith(this_pid + " "):
            continue
        lines.append(line)
    return lines


def load_previous_state(path: Path) -> dict[str, Any]:
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(path: Path, *, now: float, records: int) -> None:
    path.write_text(json.dumps({"time": now, "records": records}, indent=2))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    run_config = read_json(run_dir / "run_config.json")
    variant = args.variant or (run_config.get("prompt_variants") or [DEFAULT_VARIANT])[0]
    total = int(args.total or run_config.get("num_examples") or DEFAULT_TOTAL)

    variant_dir = run_dir / variant
    records_path = variant_dir / "records.jsonl"
    successful_path = variant_dir / "successful_records.jsonl"
    summary_path = variant_dir / "summary.json"
    state_path = variant_dir / ".progress_state.json"

    records = read_jsonl(records_path)
    successful = read_jsonl(successful_path)
    summary = read_json(summary_path)
    now = time.time()

    start_time = None
    if (run_dir / "run_config.json").exists():
        start_time = (run_dir / "run_config.json").stat().st_mtime
    elif records_path.exists():
        start_time = records_path.stat().st_mtime
    elapsed = (now - start_time) if start_time is not None else None

    completed = len(records)
    remaining = max(0, total - completed)
    active = active_processes(args.process_filter)
    is_active = bool(active)
    resumed_without_new_records = (
        is_active
        and start_time is not None
        and records_path.exists()
        and records_path.stat().st_mtime < start_time
        and completed > 0
    )
    average_eta = (
        elapsed / completed * remaining
        if is_active and not resumed_without_new_records and elapsed and completed
        else None
    )
    percent = (completed / total * 100.0) if total else 0.0

    previous = {} if args.no_state else load_previous_state(state_path)
    recent_eta = None
    recent_rate_text = "unknown"
    if previous:
        previous_records = int(previous.get("records") or 0)
        previous_time = float(previous.get("time") or 0.0)
        delta_records = completed - previous_records
        delta_seconds = now - previous_time
        if delta_records > 0 and delta_seconds > 0:
            recent_eta = delta_seconds / delta_records * remaining if is_active else None
            recent_rate_text = fmt_rate(delta_records, delta_seconds)
        elif delta_records == 0 and delta_seconds > 0:
            recent_rate_text = f"no new records in {fmt_duration(delta_seconds)}"
    if not args.no_state:
        write_state(state_path, now=now, records=completed)

    errors = [record for record in records if record.get("error")]
    exact = [record for record in records if record.get("exact_match")]
    judged = [
        record
        for record in records
        if record.get("exact_match") or record.get("judge_score") is not None or record.get("judge_raw_response") is not None
    ]
    accepted = [
        record
        for record in records
        if not record.get("error") and (record.get("exact_match") or record.get("judge_score") == 1.0)
    ]
    subcalls = [int(record.get("num_llm_subcalls") or record.get("num_subcalls") or 0) for record in records]
    tokens = [float(record.get("total_rollout_tokens") or record.get("total_model_tokens") or 0.0) for record in records]
    print(f"Run: {run_dir}")
    print(f"Variant: {variant}")
    print(f"Status: {'active' if is_active else 'stopped / no matching process found'}")
    if active:
        print("Process:")
        for line in active[:5]:
            print(f"  {line}")
    print()
    print(f"Progress: {completed}/{total} ({percent:.1f}%), remaining {remaining}")
    print(f"Elapsed: {fmt_duration(elapsed)}")
    if resumed_without_new_records:
        print("Average rate: waiting for a new resumed record")
    else:
        print(f"Average rate: {fmt_rate(completed, elapsed)}")
    print(f"Average ETA: {fmt_duration(average_eta)}")
    print(f"Recent rate: {recent_rate_text}")
    print(f"Recent ETA: {fmt_duration(recent_eta)}")
    if not is_active:
        print("ETA note: unavailable while stopped; restart/resume the run for a meaningful ETA.")
    elif resumed_without_new_records and not previous:
        print("ETA note: run resumed with existing records; run this script again after a new record lands for recent ETA.")
    if average_eta is not None:
        finish = datetime.fromtimestamp(now + average_eta, tz=timezone.utc).astimezone()
        print(f"Average finish estimate: {finish:%Y-%m-%d %H:%M:%S %Z}")
    print()
    print(f"Successful records: {len(successful)}")
    print(f"Errors: {len(errors)}")
    print(f"Judge completed: {len(judged)}/{completed}")
    print(f"Accepted exact-or-judge: {len(accepted)}/{completed}")
    print(f"Exact matches: {len(exact)}/{completed}")
    if subcalls:
        print(
            "LLM subcalls: "
            f"total {sum(subcalls)}, min {min(subcalls)}, mean {sum(subcalls) / len(subcalls):.2f}, max {max(subcalls)}"
        )
    if tokens:
        print(f"Mean rollout tokens: {sum(tokens) / len(tokens):.1f}")
    if records:
        print("Last records:")
        for record in records[-5:]:
            source_id = record.get("source_id", record.get("example_id"))
            status = "error" if record.get("error") else "ok"
            accepted_flag = bool(not record.get("error") and (record.get("exact_match") or record.get("judge_score") == 1.0))
            print(
                f"  {source_id}: {status}, accepted={accepted_flag}, "
                f"exact={bool(record.get('exact_match'))}, judge={record.get('judge_score')}, "
                f"subcalls={int(record.get('num_llm_subcalls') or record.get('num_subcalls') or 0)}"
            )
    if summary:
        print()
        print(
            "Summary file: "
            f"records={summary.get('num_records')}, "
            f"exact_rate={summary.get('exact_match_rate')}, "
            f"mean_judge={summary.get('mean_judge_score')}, "
            f"errors={summary.get('num_errors')}"
        )


if __name__ == "__main__":
    main()
