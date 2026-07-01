#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any


def _add_site_packages(repo_root: Path) -> None:
    site_packages = repo_root / "environments" / "rlm_rlvr" / ".venv" / "lib"
    for child in site_packages.glob("python*/site-packages"):
        sys.path.insert(0, str(child))


def _parse_info(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _source_id_from_input(rollout_input: dict[str, Any]) -> str:
    info = _parse_info(rollout_input.get("info"))
    metadata = _parse_info(info.get("metadata"))
    for value in (
        metadata.get("original_source_id"),
        info.get("source_id"),
        rollout_input.get("example_id"),
    ):
        if value is not None and str(value):
            return str(value).split("__rollout_", 1)[0]
    return str(rollout_input.get("example_id", "unknown"))


def _work_id_from_input(rollout_input: dict[str, Any]) -> str:
    info = _parse_info(rollout_input.get("info"))
    metadata = _parse_info(info.get("metadata"))
    for value in (
        rollout_input.get("dynamic_eval_work_id"),
        metadata.get("original_work_id"),
        metadata.get("work_id"),
        info.get("source_id"),
        rollout_input.get("example_id"),
    ):
        if value is not None and str(value):
            return str(value)
    return _source_id_from_input(rollout_input)


def _source_id_from_output(output: dict[str, Any]) -> str:
    info = _parse_info(output.get("info"))
    metadata = _parse_info(info.get("metadata"))
    for value in (
        metadata.get("original_source_id"),
        info.get("source_id"),
        output.get("example_id"),
        output.get("id"),
    ):
        if value is not None and str(value):
            return str(value).split("__rollout_", 1)[0]
    return "unknown"


def _work_id_from_output(output: dict[str, Any]) -> str:
    info = _parse_info(output.get("info"))
    metadata = _parse_info(info.get("metadata"))
    for value in (
        metadata.get("original_work_id"),
        metadata.get("work_id"),
        info.get("source_id"),
        output.get("example_id"),
        output.get("id"),
    ):
        if value is not None and str(value):
            return str(value)
    return _source_id_from_output(output)


def _load_completed_sources(result_root: Path) -> set[str]:
    completed: set[str] = set()
    for path in sorted(result_root.rglob("results.jsonl")):
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        completed.add(_work_id_from_output(json.loads(line)))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            continue
    completed.discard("unknown")
    return completed


def _timeout_output(
    rollout_input: dict[str, Any],
    *,
    source_id: str,
    attempt_number: int,
    timeout_seconds: float,
    duration_s: float,
    worker_id: int,
    dp_rank: int,
    state_columns: list[str],
) -> dict[str, Any]:
    metrics = {
        "used_repl": 0.0,
        "used_recursion": 0.0,
        "used_llm_subcalls": 0.0,
        "used_rlm_subcalls": 0.0,
        "num_subcalls": 0.0,
        "num_llm_subcalls": 0.0,
        "num_rlm_subcalls": 0.0,
        "max_depth_reached": 0.0,
    }
    output: dict[str, Any] = {
        "prompt": rollout_input.get("prompt") or [],
        "completion": [],
        "answer": rollout_input.get("answer", ""),
        "task": rollout_input.get("task", ""),
        "info": rollout_input.get("info", {}),
        "reward": 0.0,
        "metrics": metrics,
        "example_id": rollout_input.get("example_id", -1),
        "is_truncated": False,
        "stop_condition": "rollout_timeout",
        "error": {
            "error_chain_str": f"rollout_timeout after {timeout_seconds:.1f}s",
            "type": "TimeoutError",
        },
        "timing": {
            "generation_ms": duration_s * 1000.0,
            "scoring_ms": 0.0,
            "total_ms": duration_s * 1000.0,
        },
    }
    extras = {
        "source_id": source_id,
        "attempt_number": attempt_number,
        "timeout": True,
        "timeout_seconds": timeout_seconds,
        "duration_s": duration_s,
        "worker_id": worker_id,
        "dp_rank": dp_rank,
        "final_answer": None,
        "missing_final": True,
        "hit_max_turn_without_final": False,
    }
    output.update({key: extras.get(key) for key in state_columns if key in extras})
    return output


def _compact_trace_stats(output: dict[str, Any], state_columns: list[str]) -> None:
    segments = output.get("rlm_segments")
    if isinstance(segments, str) and segments.strip():
        try:
            segments = json.loads(segments)
        except json.JSONDecodeError:
            segments = []
    prompt = completion = trainable = plain_tokens = 0
    empty_plain = error_like_plain = 0
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
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
    compact = {
        "segment_total_tokens": float(prompt + completion),
        "segment_trainable_tokens": float(trainable),
        "segment_plain_subcall_tokens": float(plain_tokens),
        "empty_plain_subcalls": float(empty_plain),
        "error_like_plain_subcalls": float(error_like_plain),
        "rlm_trace_dropped_from_results": True,
        "rlm_segments_dropped_from_results": True,
    }
    for key, value in compact.items():
        if key not in output and (key in state_columns or key.startswith(("segment_", "empty_", "error_", "rlm_"))):
            output[key] = value
    output.pop("rlm_trace", None)
    output.pop("rlm_segments", None)


def _worker_main(
    worker_id: int,
    dp_rank: int,
    repo_root: str,
    env_args: dict[str, Any],
    model: str,
    api_base_url: str,
    api_key_var: str,
    sampling_args: dict[str, Any],
    state_columns: list[str],
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    max_retries: int,
    drop_heavy_state_columns: bool,
) -> None:
    def _handle_sigterm(*_: object) -> None:
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    root = Path(repo_root)
    _add_site_packages(root)
    import verifiers as vf
    from verifiers.types import ClientConfig

    env = vf.load_environment(env_id="rlm_rlvr", **env_args)
    client = ClientConfig(
        client_type="openai_chat_completions",
        api_key_var=api_key_var,
        api_base_url=api_base_url,
        timeout=3600.0,
        connect_timeout=5.0,
        max_connections=4096,
        max_keepalive_connections=4096,
        max_retries=10,
        extra_headers={"X-data-parallel-rank": str(dp_rank)},
    )
    result_queue.put(
        {
            "event": "ready",
            "worker_id": worker_id,
            "dp_rank": dp_rank,
            "time": time.time(),
        }
    )
    while True:
        item = task_queue.get()
        if item is None:
            return
        task_id = item["task_id"]
        rollout_input = item["input"]
        attempt_number = item["attempt_number"]
        source_id = item["source_id"]
        start = time.time()
        result_queue.put(
            {
                "event": "started",
                "task_id": task_id,
                "source_id": source_id,
                "worker_id": worker_id,
                "dp_rank": dp_rank,
                "attempt_number": attempt_number,
                "start_time": start,
            }
        )
        try:
            output = asyncio.run(
                env.run_rollout(
                    rollout_input,
                    client,
                    model,
                    sampling_args,
                    max_retries=max_retries,
                    state_columns=state_columns,
                )
            )
            duration_s = time.time() - start
            output["dynamic_eval_worker_id"] = worker_id
            output["dynamic_eval_dp_rank"] = dp_rank
            output["dynamic_eval_work_id"] = source_id
            output["dynamic_eval_attempt_number"] = attempt_number
            output["dynamic_eval_duration_s"] = duration_s
            if drop_heavy_state_columns:
                _compact_trace_stats(output, state_columns)
            result_queue.put(
                {
                    "event": "done",
                    "task_id": task_id,
                    "source_id": source_id,
                    "worker_id": worker_id,
                    "dp_rank": dp_rank,
                    "attempt_number": attempt_number,
                    "duration_s": duration_s,
                    "output": output,
                }
            )
            result_queue.put(
                {
                    "event": "idle",
                    "worker_id": worker_id,
                    "dp_rank": dp_rank,
                    "source_id": source_id,
                    "task_id": task_id,
                    "time": time.time(),
                }
            )
        except BaseException as exc:
            duration_s = time.time() - start
            result_queue.put(
                {
                    "event": "error",
                    "task_id": task_id,
                    "source_id": source_id,
                    "worker_id": worker_id,
                    "dp_rank": dp_rank,
                    "attempt_number": attempt_number,
                    "duration_s": duration_s,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=20),
                }
            )
            result_queue.put(
                {
                    "event": "idle",
                    "worker_id": worker_id,
                    "dp_rank": dp_rank,
                    "source_id": source_id,
                    "task_id": task_id,
                    "time": time.time(),
                }
            )


def _load_inputs(repo_root: Path, env_args: dict[str, Any], rollouts_per_example: int) -> list[dict[str, Any]]:
    _add_site_packages(repo_root)
    import verifiers as vf

    env = vf.load_environment(env_id="rlm_rlvr", **env_args)
    inputs = env._get_eval_inputs(-1, rollouts_per_example)  # noqa: SLF001
    return [dict(item) for item in inputs]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _attempt_log_event(event: dict[str, Any]) -> dict[str, Any]:
    """Keep attempt logs compact; full rollouts are written to results.jsonl."""
    compact = {key: value for key, value in event.items() if key != "output"}
    output = event.get("output")
    if isinstance(output, dict):
        compact["stop_condition"] = output.get("stop_condition")
        compact["reward"] = output.get("reward")
        compact["is_completed"] = output.get("is_completed")
        compact["is_truncated"] = output.get("is_truncated")
        compact["final_answer_present"] = bool(output.get("final_answer"))
        metrics = output.get("metrics")
        if isinstance(metrics, dict):
            for key in (
                "correctness_metric",
                "judge_score_metric",
                "used_repl_metric",
                "used_llm_subcalls_metric",
                "used_rlm_subcalls_metric",
                "num_llm_subcalls_metric",
                "num_rlm_subcalls_metric",
                "max_depth_metric",
            ):
                if key in metrics:
                    compact[key] = metrics[key]
        token_usage = output.get("token_usage")
        if isinstance(token_usage, dict):
            compact["input_tokens"] = token_usage.get("input_tokens")
            compact["output_tokens"] = token_usage.get("output_tokens")
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic work-stealing RLM eval runner.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--api-key-var", default="LOCAL_VLLM_API_KEY")
    parser.add_argument("--env-args-file", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--dp-size", type=int, default=8)
    parser.add_argument("--per-rank-cap", type=int, default=4)
    parser.add_argument("--rollout-timeout-seconds", type=float, default=400.0)
    parser.add_argument("--worker-ready-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--assigned-start-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--rollouts-per-example", type=int, default=1)
    parser.add_argument("--state-columns", default="")
    parser.add_argument("--sampling-args", default="{}")
    parser.add_argument("--drop-heavy-state-columns", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mp-start-method",
        choices=("fork", "spawn", "forkserver"),
        default="fork" if sys.platform.startswith("linux") else "spawn",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    state_columns = [item for item in args.state_columns.split(",") if item]
    env_args = json.loads(args.env_args_file.read_text(encoding="utf-8"))
    sampling_args = json.loads(args.sampling_args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = output_dir / "attempts.jsonl"

    inputs = _load_inputs(repo_root, env_args, args.rollouts_per_example)
    by_source: dict[str, dict[str, Any]] = {}
    seen_work_ids: dict[str, int] = {}
    for item in inputs:
        base_work_id = _work_id_from_input(item)
        occurrence = seen_work_ids.get(base_work_id, 0)
        seen_work_ids[base_work_id] = occurrence + 1
        work_id = base_work_id if occurrence == 0 else f"{base_work_id}__repeat_{occurrence}"
        by_source[work_id] = item
    completed = _load_completed_sources(args.result_root)
    pending_sources = [source_id for source_id in by_source if source_id not in completed]
    total_expected = len(by_source)

    _add_site_packages(repo_root)
    from verifiers.types import ClientConfig
    from verifiers.utils.save_utils import GenerateOutputsBuilder, save_metadata, save_new_outputs

    builder = GenerateOutputsBuilder(
        env_id="rlm_rlvr",
        env_args=env_args,
        model=args.model,
        client=ClientConfig(
            client_type="openai_chat_completions",
            api_key_var=args.api_key_var,
            api_base_url=args.api_base_url,
        ),
        num_examples=total_expected,
        rollouts_per_example=args.rollouts_per_example,
        state_columns=state_columns,
        sampling_args=sampling_args,
        results_path=output_dir,
        pass_threshold=0.5,
    )

    try:
        ctx = mp.get_context(args.mp_start_method)
    except ValueError:
        ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    workers: dict[int, mp.Process] = {}
    worker_queues: dict[int, mp.Queue] = {}
    worker_current: dict[int, dict[str, Any] | None] = {}
    worker_ready: dict[int, bool] = {}
    attempts: dict[str, int] = {source_id: 0 for source_id in by_source}
    exhausted: set[str] = set()
    pending_queue = deque(pending_sources)

    def spawn_worker(worker_id: int) -> None:
        dp_rank = worker_id % args.dp_size
        task_queue: mp.Queue = ctx.Queue()
        worker_queues[worker_id] = task_queue
        proc = ctx.Process(
            target=_worker_main,
            args=(
                worker_id,
                dp_rank,
                str(repo_root),
                env_args,
                args.model,
                args.api_base_url,
                args.api_key_var,
                sampling_args,
                state_columns,
                task_queue,
                result_queue,
                args.max_retries,
                args.drop_heavy_state_columns,
            ),
            daemon=True,
        )
        proc.start()
        workers[worker_id] = proc
        worker_ready[worker_id] = False
        worker_current[worker_id] = {
            "event": "starting",
            "worker_id": worker_id,
            "dp_rank": dp_rank,
            "spawn_time": time.time(),
        }

    def stop_worker(worker_id: int) -> None:
        proc = workers.get(worker_id)
        if proc is None:
            return
        worker_ready[worker_id] = False
        worker_current[worker_id] = None
        try:
            worker_queues[worker_id].put(None)
        except Exception:
            pass
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)

    worker_count = min(args.workers, args.dp_size * args.per_rank_cap)
    for worker_id in range(worker_count):
        spawn_worker(worker_id)

    task_counter = 0
    in_queue: set[str] = set()

    def enqueue_to_worker(worker_id: int, source_id: str) -> None:
        nonlocal task_counter
        if source_id in completed or source_id in exhausted or source_id in in_queue:
            return
        attempts[source_id] = attempts.get(source_id, 0) + 1
        task_counter += 1
        in_queue.add(source_id)
        dp_rank = worker_id % args.dp_size
        worker_current[worker_id] = {
            "event": "assigned",
            "task_id": task_counter,
            "source_id": source_id,
            "worker_id": worker_id,
            "dp_rank": dp_rank,
            "attempt_number": attempts[source_id],
            "assigned_time": time.time(),
        }
        worker_queues[worker_id].put(
            {
                "task_id": task_counter,
                "source_id": source_id,
                "input": by_source[source_id],
                "attempt_number": attempts[source_id],
            }
        )

    def dispatch_idle_workers() -> None:
        for worker_id in range(worker_count):
            if not worker_ready.get(worker_id, False):
                continue
            if worker_current.get(worker_id) is not None:
                continue
            while pending_queue:
                source_id = pending_queue.popleft()
                if source_id in completed or source_id in exhausted or source_id in in_queue:
                    continue
                enqueue_to_worker(worker_id, source_id)
                break

    def release_worker_for_event(event: dict[str, Any]) -> None:
        worker_id = int(event.get("worker_id", -1))
        if worker_id < 0:
            return
        source_id = str(event.get("source_id", ""))
        task_id = event.get("task_id")
        current = worker_current.get(worker_id)
        if isinstance(current, dict):
            current_source = str(current.get("source_id", ""))
            current_task = current.get("task_id")
            if current_source == source_id and (task_id is None or current_task == task_id):
                worker_current[worker_id] = None
        in_queue.discard(source_id)

    dispatch_idle_workers()

    last_log = 0.0
    last_flush = time.time()
    pending_save_outputs: list[dict[str, Any]] = []

    def flush_outputs(force: bool = False) -> None:
        nonlocal last_flush
        if not pending_save_outputs:
            return
        now = time.time()
        if not force and len(pending_save_outputs) < 50 and now - last_flush < 30.0:
            return
        save_new_outputs(list(pending_save_outputs), output_dir)
        pending_save_outputs.clear()
        save_metadata(builder.build_metadata(), output_dir)
        last_flush = now

    try:
        while len(completed) < total_expected:
            now = time.time()
            if now - last_log >= 15.0:
                active_by_rank: dict[int, int] = {}
                active = 0
                assigned = 0
                starting = 0
                for info in worker_current.values():
                    if info is None:
                        continue
                    if info.get("event") == "assigned":
                        assigned += 1
                    elif info.get("event") == "starting":
                        starting += 1
                    else:
                        active += 1
                    active_by_rank[info["dp_rank"]] = active_by_rank.get(info["dp_rank"], 0) + 1
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "label": args.label,
                            "completed": len(completed),
                            "exhausted": len(exhausted),
                            "expected": total_expected,
                            "active": active,
                            "assigned_not_started": assigned,
                            "starting": starting,
                            "ready_workers": sum(1 for value in worker_ready.values() if value),
                            "queued": len(pending_queue) + len(in_queue),
                            "submitted_not_started": len(in_queue),
                            "active_by_rank": active_by_rank,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                last_log = now

            for worker_id, info in list(worker_current.items()):
                if info is None:
                    continue
                if info.get("event") == "starting":
                    start_duration = now - float(info.get("spawn_time") or now)
                    proc = workers.get(worker_id)
                    if start_duration <= args.worker_ready_timeout_seconds and proc is not None and proc.is_alive():
                        continue
                    _append_jsonl(
                        attempts_path,
                        {
                            "event": "worker_ready_timeout",
                            "worker_id": worker_id,
                            "dp_rank": info.get("dp_rank"),
                            "duration_s": start_duration,
                            "worker_alive": bool(proc is not None and proc.is_alive()),
                        },
                    )
                    stop_worker(worker_id)
                    spawn_worker(worker_id)
                    continue
                if info.get("event") == "assigned":
                    assigned_duration = now - float(info.get("assigned_time") or now)
                    proc = workers.get(worker_id)
                    if assigned_duration <= args.assigned_start_timeout_seconds and proc is not None and proc.is_alive():
                        continue
                    source_id = info["source_id"]
                    _append_jsonl(
                        attempts_path,
                        {
                            "event": "assigned_start_timeout",
                            "source_id": source_id,
                            "worker_id": worker_id,
                            "dp_rank": info["dp_rank"],
                            "attempt_number": info["attempt_number"],
                            "duration_s": assigned_duration,
                            "worker_alive": bool(proc is not None and proc.is_alive()),
                        },
                    )
                    worker_current[worker_id] = None
                    in_queue.discard(source_id)
                    stop_worker(worker_id)
                    spawn_worker(worker_id)
                    if attempts.get(source_id, 0) < args.max_attempts:
                        pending_queue.appendleft(source_id)
                    else:
                        exhausted.add(source_id)
                        output = _timeout_output(
                            by_source[source_id],
                            source_id=source_id,
                            attempt_number=attempts[source_id],
                            timeout_seconds=args.assigned_start_timeout_seconds,
                            duration_s=assigned_duration,
                            worker_id=worker_id,
                            dp_rank=info["dp_rank"],
                            state_columns=state_columns,
                        )
                        output["stop_condition"] = "worker_start_timeout"
                        output["error"] = {
                            "error_chain_str": (
                                f"worker_start_timeout after {args.assigned_start_timeout_seconds:.1f}s"
                            ),
                            "type": "WorkerStartTimeout",
                        }
                        completed.add(source_id)
                        builder.add_outputs([output])
                        save_new_outputs([output], output_dir)
                        save_metadata(builder.build_metadata(), output_dir)
                    dispatch_idle_workers()
                    continue
                if now - float(info["start_time"]) <= args.rollout_timeout_seconds:
                    continue
                proc = workers[worker_id]
                source_id = info["source_id"]
                _append_jsonl(
                    attempts_path,
                    {
                        "event": "timeout",
                        "source_id": source_id,
                        "worker_id": worker_id,
                        "dp_rank": info["dp_rank"],
                        "attempt_number": info["attempt_number"],
                        "duration_s": now - float(info["start_time"]),
                    },
                )
                worker_current[worker_id] = None
                in_queue.discard(source_id)
                stop_worker(worker_id)
                spawn_worker(worker_id)
                if attempts.get(source_id, 0) < args.max_attempts:
                    pending_queue.append(source_id)
                else:
                    exhausted.add(source_id)
                    output = _timeout_output(
                        by_source[source_id],
                        source_id=source_id,
                        attempt_number=attempts[source_id],
                        timeout_seconds=args.rollout_timeout_seconds,
                        duration_s=now - float(info["start_time"]),
                        worker_id=worker_id,
                        dp_rank=info["dp_rank"],
                        state_columns=state_columns,
                    )
                    completed.add(source_id)
                    builder.add_outputs([output])
                    save_new_outputs([output], output_dir)
                    save_metadata(builder.build_metadata(), output_dir)
                dispatch_idle_workers()

            try:
                event = result_queue.get(timeout=args.poll_seconds)
            except queue.Empty:
                dispatch_idle_workers()
                continue
            _append_jsonl(attempts_path, _attempt_log_event(event))
            event_type = event.get("event")
            worker_id = int(event.get("worker_id", -1))
            source_id = str(event.get("source_id", ""))
            if event_type == "ready":
                worker_ready[worker_id] = True
                current = worker_current.get(worker_id)
                if isinstance(current, dict) and current.get("event") == "starting":
                    worker_current[worker_id] = None
                dispatch_idle_workers()
                continue
            if event_type == "started":
                in_queue.discard(source_id)
                worker_current[worker_id] = event
                dispatch_idle_workers()
                continue
            if event_type == "idle":
                current = worker_current.get(worker_id)
                if isinstance(current, dict) and current.get("source_id") == source_id:
                    worker_current[worker_id] = None
                elif worker_ready.get(worker_id, False) and current is None:
                    worker_current[worker_id] = None
                dispatch_idle_workers()
                continue
            if source_id in completed:
                release_worker_for_event(event)
                dispatch_idle_workers()
                flush_outputs()
                continue
            if event_type == "done":
                output = event["output"]
                completed.add(source_id)
                builder.add_outputs([output])
                pending_save_outputs.append(output)
                flush_outputs()
                dispatch_idle_workers()
            elif event_type == "error":
                if attempts.get(source_id, 0) < args.max_attempts:
                    pending_queue.append(source_id)
                else:
                    exhausted.add(source_id)
                    output = _timeout_output(
                        by_source[source_id],
                        source_id=source_id,
                        attempt_number=attempts[source_id],
                        timeout_seconds=args.rollout_timeout_seconds,
                        duration_s=float(event.get("duration_s") or 0.0),
                        worker_id=worker_id,
                        dp_rank=int(event.get("dp_rank") or 0),
                        state_columns=state_columns,
                    )
                    output["error"] = {
                        "error_chain_str": str(event.get("error") or "rollout_error"),
                        "type": "RolloutError",
                    }
                    completed.add(source_id)
                    builder.add_outputs([output])
                    pending_save_outputs.append(output)
                    flush_outputs()
                dispatch_idle_workers()
    finally:
        flush_outputs(force=True)
        for worker_id in list(workers):
            stop_worker(worker_id)
        save_metadata(builder.build_metadata(), output_dir)

    print(
        json.dumps(
            {
                "event": "finished",
                "label": args.label,
                "completed": len(completed),
                "exhausted": len(exhausted),
                "expected": total_expected,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
