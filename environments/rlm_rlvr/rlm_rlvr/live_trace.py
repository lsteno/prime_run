from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _safe_name(value: str, *, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not cleaned:
        cleaned = "sample"
    return cleaned[:max_len]


def _sample_name(state: dict[str, Any]) -> str:
    info = state.get("info")
    if not isinstance(info, dict):
        info = {}

    source_id = info.get("source_id")
    if source_id:
        return _safe_name(str(source_id))

    example_id = state.get("example_id")
    if example_id is not None:
        return _safe_name(f"example_{example_id}")

    question = str(info.get("question", ""))
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    return _safe_name(f"sample_{digest}")


def _trace_path(state: dict[str, Any]) -> Path | None:
    existing = state.get("_live_trace_path")
    if existing:
        return Path(str(existing))

    trace_dir = state.get("live_trace_dir")
    if not trace_dir:
        return None

    variant = _safe_name(str(state.get("prompt_variant") or "default"))
    sample = _sample_name(state)
    path = Path(str(trace_dir)).expanduser() / variant / f"{sample}.json"
    state["_live_trace_path"] = str(path)
    return path


def _compact_segment(segment: dict[str, Any]) -> dict[str, Any]:
    completion_mask = segment.get("completion_mask") or []
    prompt_ids = segment.get("prompt_ids") or []
    completion_ids = segment.get("completion_ids") or []
    is_trainable = bool(segment.get("is_trainable_rlm_turn", False))
    return {
        "order": segment.get("order"),
        "call_id": segment.get("call_id"),
        "parent_call_id": segment.get("parent_call_id"),
        "depth": segment.get("depth"),
        "turn_index": segment.get("turn_index"),
        "kind": segment.get("kind"),
        "train_scope": segment.get("train_scope"),
        "is_trainable_rlm_turn": bool(segment.get("is_trainable_rlm_turn", False)),
        "response_source": segment.get("response_source"),
        "prompt_fingerprint": segment.get("prompt_fingerprint"),
        "prompt_message_count": segment.get("prompt_message_count"),
        "prompt_char_count": segment.get("prompt_char_count"),
        "prompt_tokens": len(prompt_ids) if isinstance(prompt_ids, list) else None,
        "completion_tokens": len(completion_ids) if isinstance(completion_ids, list) else None,
        "trainable_completion_tokens": (
            sum(1 for item in completion_mask if item) if is_trainable and isinstance(completion_mask, list) else 0
        )
        if isinstance(completion_mask, list)
        else None,
        "temperature": segment.get("temperature"),
        "response_text": segment.get("response_text"),
    }


def _sample_metadata(state: dict[str, Any]) -> dict[str, Any]:
    info = state.get("info")
    if not isinstance(info, dict):
        info = {}
    metadata = info.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    selected_metadata = {
        key: metadata.get(key)
        for key in (
            "source_dataset",
            "task_group",
            "reasoning_types",
            "repo",
            "language",
            "n_docs",
            "n_wiki",
        )
        if key in metadata
    }
    return {
        "source_id": info.get("source_id"),
        "dataset_name": info.get("dataset_name"),
        "source_task": info.get("source_task"),
        "answer_type": info.get("answer_type"),
        "context_token_count": info.get("context_token_count"),
        "metadata": selected_metadata,
    }


def write_live_trace(
    state: dict[str, Any],
    *,
    event: str,
    active_trace: dict[str, Any] | None = None,
) -> None:
    path = _trace_path(state)
    if path is None:
        return

    traces = list(state.get("rlm_trace") or [])
    if active_trace is not None and active_trace not in traces:
        traces.append(active_trace)

    payload = {
        "updated_at": time.time(),
        "event": event,
        "prompt_variant": state.get("prompt_variant"),
        "sample": _sample_metadata(state),
        "status": {
            "used_repl": bool(state.get("used_repl", False)),
            "used_recursion": bool(state.get("used_recursion", False)),
            "used_llm_subcalls": bool(state.get("used_llm_subcalls", False)),
            "used_rlm_subcalls": bool(state.get("used_rlm_subcalls", False)),
            "max_depth_reached": int(state.get("max_depth_reached", 0)),
            "num_subcalls": int(state.get("num_subcalls", 0)),
            "num_llm_subcalls": int(state.get("num_llm_subcalls", 0)),
            "num_rlm_subcalls": int(state.get("num_rlm_subcalls", 0)),
            "final_answer": state.get("final_answer"),
            "total_model_tokens": float(state.get("total_model_tokens", 0.0)),
            "total_env_tokens": float(state.get("total_env_tokens", 0.0)),
            "total_prompt_tokens": float(state.get("total_prompt_tokens", 0.0)),
            "total_completion_tokens": float(state.get("total_completion_tokens", 0.0)),
            "total_rollout_tokens": float(state.get("total_rollout_tokens", 0.0)),
        },
        "traces": traces,
        "segments": [_compact_segment(segment) for segment in state.get("rlm_segments") or []],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False)
    tmp_path.replace(path)
