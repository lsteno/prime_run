#!/usr/bin/env python3
"""Poll Prime Intellect GPU availability and alert when target pods appear.

This script only checks availability. It never creates or reserves pods.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_GPU_TYPES = ("A100", "H100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll Prime Intellect availability for target GPU nodes and alert when any are in stock.",
    )
    parser.add_argument("--gpu-count", type=int, default=4, help="Required GPU count. Default: 4.")
    parser.add_argument(
        "--gpu-types",
        default=",".join(DEFAULT_GPU_TYPES),
        help="Comma-separated GPU type substrings to match. Default: A100,H100.",
    )
    parser.add_argument(
        "--min-gpu-memory-gb",
        type=int,
        default=40,
        help="Minimum per-GPU memory in GB. Default: 40.",
    )
    parser.add_argument("--regions", help="Optional Prime region filter, e.g. united_states.")
    parser.add_argument("--provider", help="Optional Prime provider filter, e.g. aws.")
    parser.add_argument("--socket", help="Optional socket filter, e.g. SXM5 or SXM4.")
    parser.add_argument("--interval-seconds", type=int, default=60, help="Poll interval. Default: 60.")
    parser.add_argument("--max-attempts", type=int, help="Stop after this many polls. Default: run forever.")
    parser.add_argument(
        "--state-file",
        default="outputs/prime_gpu_availability/latest_match.json",
        help="Where to write the latest matching resources.",
    )
    parser.add_argument(
        "--notify-command",
        help=(
            "Optional shell command to run when matches are found. "
            "Environment variables PRIME_GPU_MATCH_COUNT and PRIME_GPU_MATCH_FILE are set."
        ),
    )
    parser.add_argument(
        "--keep-polling",
        action="store_true",
        help="Continue polling after a match. By default the script exits after alerting once.",
    )
    return parser.parse_args()


def run_availability(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    cmd = [
        "prime",
        "--plain",
        "availability",
        "list",
        "--gpu-count",
        str(args.gpu_count),
        "--output",
        "json",
    ]
    if args.regions:
        cmd.extend(["--regions", args.regions])
    if args.provider:
        cmd.extend(["--provider", args.provider])
    if args.socket:
        cmd.extend(["--socket", args.socket])

    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = completed.stdout.strip()
    json_start = output.find("{")
    if json_start < 0:
        raise RuntimeError(f"Prime CLI did not return JSON. Output:\n{output}")
    payload = json.loads(output[json_start:])
    return payload, output[:json_start].strip()


def parse_memory_gb(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return int(value)
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


def is_available(resource: dict[str, Any]) -> bool:
    status = str(resource.get("stock_status", "")).lower()
    if status in {"", "none", "n/a"}:
        return True
    unavailable_markers = ("unavailable", "out_of_stock", "out-of-stock", "sold out", "none")
    if any(marker in status for marker in unavailable_markers):
        return False
    return True


def matches(resource: dict[str, Any], *, gpu_types: list[str], gpu_count: int, min_gpu_memory_gb: int) -> bool:
    gpu_type = str(resource.get("gpu_type", ""))
    if not any(target.lower() in gpu_type.lower() for target in gpu_types):
        return False
    if int(resource.get("gpu_count") or 0) < gpu_count:
        return False
    gpu_memory = parse_memory_gb(resource.get("gpu_memory"))
    if gpu_memory is not None and gpu_memory < min_gpu_memory_gb:
        return False
    return is_available(resource)


def write_matches(path: Path, matches_: list[dict[str, Any]], *, warnings: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "found_at": dt.datetime.now(dt.UTC).isoformat(),
        "match_count": len(matches_),
        "warnings": warnings,
        "gpu_resources": matches_,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def print_matches(matches_: list[dict[str, Any]], state_file: Path) -> None:
    print("\a", end="", flush=True)
    print(f"\nFOUND {len(matches_)} matching Prime GPU resource(s). Details written to {state_file}")
    for item in matches_[:10]:
        print(
            "- "
            f"id={item.get('id')} cloud_id={item.get('cloud_id')} "
            f"type={item.get('gpu_type')} count={item.get('gpu_count')} "
            f"memory={item.get('gpu_memory')} provider={item.get('provider')} "
            f"location={item.get('location')} price={item.get('price_per_hour')}"
        )


def macos_notify(message: str) -> None:
    if sys.platform != "darwin":
        return
    script = f'display notification {json.dumps(message)} with title "Prime GPUs available" sound name "Glass"'
    subprocess.run(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def run_notify_command(command: str, *, match_count: int, state_file: Path) -> None:
    env = {
        **os.environ,
        "PRIME_GPU_MATCH_COUNT": str(match_count),
        "PRIME_GPU_MATCH_FILE": str(state_file),
    }
    subprocess.run(command, shell=True, env=env, check=False)


def main() -> int:
    args = parse_args()
    gpu_types = [item.strip() for item in args.gpu_types.split(",") if item.strip()]
    state_file = Path(args.state_file)
    attempt = 0

    while args.max_attempts is None or attempt < args.max_attempts:
        attempt += 1
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            payload, warnings = run_availability(args)
            resources = payload.get("gpu_resources") or []
            matches_ = [
                item
                for item in resources
                if isinstance(item, dict)
                and matches(
                    item,
                    gpu_types=gpu_types,
                    gpu_count=args.gpu_count,
                    min_gpu_memory_gb=args.min_gpu_memory_gb,
                )
            ]
        except Exception as exc:
            print(f"[{timestamp}] poll failed: {exc}", file=sys.stderr)
            matches_ = []
            warnings = str(exc)

        if matches_:
            write_matches(state_file, matches_, warnings=warnings)
            print_matches(matches_, state_file)
            macos_notify(f"{len(matches_)} matching {args.gpu_count}x {','.join(gpu_types)} resource(s) found.")
            if args.notify_command:
                run_notify_command(args.notify_command, match_count=len(matches_), state_file=state_file)
            if not args.keep_polling:
                return 0
        else:
            print(
                f"[{timestamp}] no matches for {args.gpu_count}x {','.join(gpu_types)} "
                f"with >= {args.min_gpu_memory_gb}GB GPU memory"
            )
            if warnings:
                print(f"[{timestamp}] prime warning: {warnings}", file=sys.stderr)

        time.sleep(max(1, int(args.interval_seconds)))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
