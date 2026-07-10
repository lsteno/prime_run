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
    is_trainable = bool(segment.get("is_trainable_rlm_turn", False))
    prompt_token_count = segment.get("prompt_token_count")
    completion_token_count = segment.get("completion_token_count")
    summary = {
        "order": int(segment.get("order", 0)),
        "call_id": segment.get("call_id"),
        "parent_call_id": segment.get("parent_call_id"),
        "depth": int(segment.get("depth", 0)),
        "turn_index": segment.get("turn_index"),
        "kind": segment.get("kind"),
        "train_scope": segment.get("train_scope"),
        "is_trainable_rlm_turn": bool(segment.get("is_trainable_rlm_turn", False)),
        "response_source": segment.get("response_source"),
        "prompt_fingerprint": segment.get("prompt_fingerprint"),
        "prompt_message_count": segment.get("prompt_message_count"),
        "prompt_char_count": segment.get("prompt_char_count"),
        "temperature": segment.get("temperature"),
        "response_text": segment.get("response_text"),
        "prompt_token_count": int(prompt_token_count)
        if prompt_token_count not in (None, "")
        else len(segment.get("prompt_ids", [])),
        "completion_token_count": int(completion_token_count)
        if completion_token_count not in (None, "")
        else len(segment.get("completion_ids", [])),
        "trainable_token_count": sum(completion_mask) if is_trainable else 0,
    }
    semantic_keys = (
        "semantic_recognized_record_count",
        "semantic_recognized_chunk_ids",
        "semantic_primary_chunk_id",
        "semantic_full_chunk",
        "semantic_input_verified",
        "semantic_input_rejection_reason",
        "semantic_local_signal",
        "semantic_local_contract",
        "semantic_local_accuracy",
        "semantic_local_coverage",
        "semantic_local_schema_valid",
        "semantic_local_advantage",
    )
    summary.update({key: segment[key] for key in semantic_keys if segment.get(key) is not None})
    return summary


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


def _rollout_debug(rollout: vf.RolloutOutput) -> dict[str, Any]:
    trajectory = rollout.get("trajectory") or []
    if not trajectory:
        return {}
    last_step = trajectory[-1]
    extras = last_step.get("extras") or {}
    return extras.get("rlm_debug") or {}


def _rollout_record(rollout: vf.RolloutOutput, include_segments: bool, include_metrics: bool) -> dict[str, Any]:
    debug = _rollout_debug(rollout)
    sample_metadata = debug.get("sample_metadata") or {}
    trace = rollout.get("rlm_trace")
    if not trace:
        trace = debug.get("trace") or []

    segments = rollout.get("rlm_segments")
    if not segments:
        segments = debug.get("segments") or []

    record = {
        "example_id": rollout.get("example_id"),
        "task": rollout.get("task"),
        "answer": _json_safe(rollout.get("answer")),
        "expected_answers": _json_safe(debug.get("expected_answers")),
        "source_id": _json_safe(sample_metadata.get("source_id")),
        "dataset_name": _json_safe(sample_metadata.get("dataset_name")),
        "source_task": _json_safe(sample_metadata.get("source_task")),
        "answer_type": _json_safe(sample_metadata.get("answer_type")),
        "context_token_count": _json_safe(sample_metadata.get("context_token_count")),
        "sample_metadata": _json_safe(sample_metadata.get("metadata") or {}),
        "rlm_answer": _json_safe(_rollout_answer(rollout)),
        "judge_score": _json_safe(debug.get("judge_score")),
        "judge_raw_response": _json_safe(debug.get("judge_raw_response")),
        "judge_parse_error": _json_safe(debug.get("judge_parse_error")),
        "reward": rollout.get("reward"),
        "error": _json_safe(rollout.get("error")),
        "final_answer": rollout.get("final_answer"),
        "is_truncated": rollout.get("is_truncated"),
        "stop_condition": rollout.get("stop_condition"),
        "sampling_args": _json_safe(rollout.get("sampling_args") or {}),
        "timing": _json_safe(rollout.get("timing") or {}),
        "trace": _json_safe(trace),
    }
    if include_metrics:
        record["metrics"] = _json_safe(rollout.get("metrics") or {})
    if include_segments:
        record["segments"] = [_segment_summary(segment) for segment in segments]
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
