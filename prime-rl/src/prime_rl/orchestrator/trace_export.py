from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import verifiers as vf


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_safe(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _segment_summary(segment: dict[str, Any]) -> dict[str, Any]:
    completion_mask = [bool(value) for value in segment.get("completion_mask", [])]
    return {
        "order": int(segment.get("order", 0)),
        "depth": int(segment.get("depth", 0)),
        "kind": segment.get("kind"),
        "temperature": segment.get("temperature"),
        "response_text": segment.get("response_text"),
        "prompt_token_count": len(segment.get("prompt_ids", [])),
        "completion_token_count": len(segment.get("completion_ids", [])),
        "trainable_token_count": sum(completion_mask),
    }


def _rollout_answer(rollout: vf.RolloutOutput) -> str | None:
    final_answer = rollout.get("final_answer")
    if final_answer not in (None, ""):
        return str(final_answer)

    completion = rollout.get("completion") or []
    if completion:
        last_message = completion[-1]
        if isinstance(last_message, dict):
            content = last_message.get("content")
            if content not in (None, ""):
                return str(content)

    return None


def _rollout_record(rollout: vf.RolloutOutput, include_segments: bool, include_metrics: bool) -> dict[str, Any]:
    record = {
        "example_id": rollout.get("example_id"),
        "task": rollout.get("task"),
        "answer": _json_safe(rollout.get("answer")),
        "rlm_answer": _json_safe(_rollout_answer(rollout)),
        "reward": rollout.get("reward"),
        "error": _json_safe(rollout.get("error")),
        "final_answer": rollout.get("final_answer"),
        "is_truncated": rollout.get("is_truncated"),
        "stop_condition": rollout.get("stop_condition"),
        "sampling_args": _json_safe(rollout.get("sampling_args") or {}),
        "timing": _json_safe(rollout.get("timing") or {}),
        "trace": _json_safe(rollout.get("rlm_trace") or []),
    }
    if include_metrics:
        record["metrics"] = _json_safe(rollout.get("metrics") or {})
    if include_segments:
        record["segments"] = [_segment_summary(segment) for segment in rollout.get("rlm_segments") or []]
    return record


def export_rollout_traces(
    *,
    rollouts: list[vf.RolloutOutput],
    step: int,
    output_dir: Path,
    include_segments: bool = True,
    include_metrics: bool = True,
) -> Path:
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"step_{step:06d}.json"
    payload = {
        "step": step,
        "num_rollouts": len(rollouts),
        "rollouts": [_rollout_record(rollout, include_segments, include_metrics) for rollout in rollouts],
    }
    with open(trace_path, "w") as f:
        json.dump(payload, f, indent=2)
    return trace_path