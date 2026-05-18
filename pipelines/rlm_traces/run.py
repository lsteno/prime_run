from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
import json
import multiprocessing
import os
import random
import statistics
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RLM_ENV_ROOT = REPO_ROOT / "environments" / "rlm_rlvr"
if str(RLM_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(RLM_ENV_ROOT))


JUDGE_PROMPT = """You are grading whether a model answer is semantically correct.

Question:
{question}

Reference answer(s):
{expected_answers}

Model answer:
{predicted_answer}

Scoring rules:
- Return 1 if the model answer is mostly correct in meaning.
- Semantic correctness matters more than exact wording or format.
- Return 1 if the answer contains the correct fact, entity, or number even if formatting is imperfect.
- Return 1 if the answer adds extra harmless text but still clearly gives the correct answer.
- Return 0 only if the answer is completely incorrect, missing the core correct information, contradictory on the final answer, or gives no answer.
- The only valid outputs are 0 or 1.

Return exactly one character: 0 or 1.
"""

JUDGE_SYSTEM_PROMPT = "Return exactly one character: 0 or 1. Never return JSON, tool calls, or explanations."


@dataclass
class EndpointConfig:
    endpoint_id: str | None
    model: str
    url: str
    api_key_env: str | None
    api_key: str
    endpoint_type: str = "openai_chat_completions"
    extra_body: dict[str, Any] | None = None


@dataclass
class JudgeConfig:
    provider: str
    model: str
    endpoint: EndpointConfig | None = None
    vertex_project_env: str = "GOOGLE_CLOUD_PROJECT"
    vertex_project: str | None = None
    vertex_location: str = "global"
    thinking_level: str | None = "medium"
    max_attempts: int = 6
    base_retry_seconds: float = 1.0
    max_retry_seconds: float = 30.0


@dataclass
class PlainSubcallConfig:
    provider: str
    model: str
    endpoint: EndpointConfig | None = None
    vertex_project_env: str = "GOOGLE_CLOUD_PROJECT"
    vertex_project: str | None = None
    vertex_location: str = "global"
    thinking_level: str | None = "medium"
    empty_response_max_attempts: int = 1
    empty_response_base_retry_seconds: float = 1.0
    empty_response_max_retry_seconds: float = 30.0


@dataclass
class Example:
    example_id: int
    source_id: str
    question: str
    answer: str
    acceptable_answers: list[str]
    context: str
    info: dict[str, Any]


def audit_jsonable(value: Any, *, _seen: set[int] | None = None) -> Any:
    """Convert REPL internals to untruncated JSON-serializable audit data."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if _seen is None:
        _seen = set()
    value_id = id(value)
    if value_id in _seen:
        return f"<recursive {type(value).__name__}>"
    _seen.add(value_id)
    try:
        if isinstance(value, dict):
            return {str(key): audit_jsonable(item, _seen=_seen) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [audit_jsonable(item, _seen=_seen) for item in value]
        if isinstance(value, set | frozenset):
            return [audit_jsonable(item, _seen=_seen) for item in sorted(value, key=repr)]
        if hasattr(value, "__dict__"):
            return {
                str(key): audit_jsonable(item, _seen=_seen)
                for key, item in vars(value).items()
                if not str(key).startswith("_")
            }
        return repr(value)
    finally:
        _seen.discard(value_id)


def audit_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
        for message in messages
    ]


def audit_local_value(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, str):
        summary["char_count"] = len(value)
        if len(value) <= 500:
            summary["value"] = value
        else:
            summary["omitted_value"] = "large string"
        return summary
    if isinstance(value, dict):
        summary["length"] = len(value)
        return summary
    if isinstance(value, list | tuple | set | frozenset):
        summary["length"] = len(value)
        return summary
    if isinstance(value, int | float | bool) or value is None:
        summary["value"] = value
        return summary
    text = repr(value)
    if len(text) <= 500:
        summary["repr"] = text
    else:
        summary["repr_char_count"] = len(text)
        summary["omitted_repr"] = "large repr"
    return summary


def audit_repl_execution(code: str, execution: Any) -> dict[str, Any]:
    calls = getattr(execution, "rlm_calls", None)
    if calls is None:
        calls = getattr(execution, "llm_calls", None)
    local_values = getattr(execution, "locals", {}) or {}
    audited_locals: dict[str, Any] = {}
    if isinstance(local_values, dict):
        for name, value in local_values.items():
            if name in {"context", "context_0"}:
                audited_locals[str(name)] = {
                    "__omitted__": "REPL context variable",
                    "type": type(value).__name__,
                    "char_count": len(str(value)),
                }
            else:
                audited_locals[str(name)] = audit_local_value(value)
    else:
        audited_locals = audit_local_value(local_values)
    return {
        "code": code,
        "stdout": str(getattr(execution, "stdout", "")),
        "stderr": str(getattr(execution, "stderr", "")),
        "final_answer": getattr(execution, "final_answer", None),
        "execution_time": getattr(execution, "execution_time", None),
        "locals": audited_locals,
        "llm_calls": audit_jsonable(calls or []),
    }


@lru_cache(maxsize=1)
def load_dataset_fn():
    from datasets import load_dataset

    return load_dataset


@lru_cache(maxsize=1)
def load_openai_client():
    from openai import OpenAI

    return OpenAI


@lru_cache(maxsize=1)
def load_google_genai():
    from google import genai
    from google.genai import types

    return genai, types


@lru_cache(maxsize=1)
def load_rlm_modules() -> dict[str, Any]:
    from rlm_rlvr.external_rlm import (
        CodeBlock,
        QueryMetadata,
        RLMIteration,
        build_system_prompt,
        build_user_prompt,
        find_code_blocks,
        find_final_answer,
        make_feedback_messages,
    )
    from rlm_rlvr.parsing import normalize_text, parse_answer_candidates
    from rlm_rlvr.repl import create_repl
    from rlm_rlvr.runtime import RecursiveRuntime, RuntimeConfig, SyncInferenceSession, VertexGeminiSession
    from rlm_rlvr.trace import make_segment

    return {
        "CodeBlock": CodeBlock,
        "QueryMetadata": QueryMetadata,
        "RLMIteration": RLMIteration,
        "RecursiveRuntime": RecursiveRuntime,
        "RuntimeConfig": RuntimeConfig,
        "SyncInferenceSession": SyncInferenceSession,
        "VertexGeminiSession": VertexGeminiSession,
        "build_system_prompt": build_system_prompt,
        "build_user_prompt": build_user_prompt,
        "create_repl": create_repl,
        "find_code_blocks": find_code_blocks,
        "find_final_answer": find_final_answer,
        "make_feedback_messages": make_feedback_messages,
        "make_segment": make_segment,
        "normalize_text": normalize_text,
        "parse_answer_candidates": parse_answer_candidates,
    }


def build_context_metadata_message(context_payload: str) -> dict[str, str]:
    QueryMetadata = load_rlm_modules()["QueryMetadata"]
    context_metadata = QueryMetadata(context_payload)
    return {
        "role": "user",
        "content": (
            f"Your context is a {context_metadata.context_type} with "
            f"{context_metadata.context_total_length} total characters, and is broken up into chunks "
            f"of char lengths: {context_metadata.context_lengths}."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RLM traces and compare prompt variants.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the pipeline TOML config.")
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_path(path_str: str, *, relative_to: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        local_path = (relative_to / path).resolve()
        repo_path = (REPO_ROOT / path).resolve()
        path = local_path if local_path.exists() else repo_path
    return path


def load_endpoints(path: Path) -> dict[str, dict[str, Any]]:
    config = load_toml(path)
    endpoints = {}
    for record in config.get("endpoint", []):
        endpoint_id = str(record["endpoint_id"])
        endpoints[endpoint_id] = record
    return endpoints


def resolve_endpoint(spec: dict[str, Any], *, endpoints_path: Path) -> EndpointConfig:
    endpoint_id = spec.get("endpoint_id")
    if endpoint_id is not None:
        endpoints = load_endpoints(endpoints_path)
        if endpoint_id not in endpoints:
            raise ValueError(f"Unknown endpoint_id: {endpoint_id}")
        record = endpoints[endpoint_id]
        endpoint_type = record.get("type")
        if endpoint_type != "openai_chat_completions":
            raise ValueError(
                f"Endpoint {endpoint_id!r} has unsupported type {endpoint_type!r}. "
                "This pipeline currently supports only openai_chat_completions endpoints."
            )
        model = str(spec.get("model") or record["model"])
        url = str(spec.get("url") or record["url"])
        api_key_env = str(spec.get("api_key_env") or record.get("key") or "")
        api_key = os.environ.get(api_key_env, "EMPTY") if api_key_env else "EMPTY"
        return EndpointConfig(
            endpoint_id=str(endpoint_id),
            model=model,
            url=url,
            api_key_env=api_key_env or None,
            api_key=api_key,
            endpoint_type=endpoint_type,
            extra_body=spec.get("extra_body"),
        )

    model = str(spec["model"])
    url = str(spec["url"])
    api_key_env = spec.get("api_key_env")
    api_key = os.environ.get(str(api_key_env), "EMPTY") if api_key_env else "EMPTY"
    return EndpointConfig(
        endpoint_id=None,
        model=model,
        url=url,
        api_key_env=str(api_key_env) if api_key_env else None,
        api_key=api_key,
        endpoint_type=str(spec.get("type", "openai_chat_completions")),
        extra_body=spec.get("extra_body"),
    )


def resolve_judge(config: dict[str, Any], *, endpoints_path: Path) -> JudgeConfig | None:
    judge_cfg = config.get("judge")
    if not judge_cfg or not bool(judge_cfg.get("enabled", False)):
        return None

    provider = str(judge_cfg.get("provider", "openai_compatible"))
    if provider == "vertex":
        project_env = str(judge_cfg.get("vertex_project_env", "GOOGLE_CLOUD_PROJECT"))
        return JudgeConfig(
            provider="vertex",
            model=str(judge_cfg.get("model", "gemini-3-flash-preview")),
            vertex_project_env=project_env,
            vertex_project=os.environ.get(project_env),
            vertex_location=str(judge_cfg.get("vertex_location", os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))),
            thinking_level=judge_cfg.get("thinking_level", "medium"),
            max_attempts=int(judge_cfg.get("max_attempts", 6)),
            base_retry_seconds=float(judge_cfg.get("base_retry_seconds", 1.0)),
            max_retry_seconds=float(judge_cfg.get("max_retry_seconds", 30.0)),
        )

    if provider != "openai_compatible":
        raise ValueError("judge.provider must be either 'openai_compatible' or 'vertex'")

    return JudgeConfig(
        provider="openai_compatible",
        model=str(judge_cfg.get("model") or ""),
        endpoint=resolve_endpoint(judge_cfg, endpoints_path=endpoints_path),
        max_attempts=int(judge_cfg.get("max_attempts", 6)),
        base_retry_seconds=float(judge_cfg.get("base_retry_seconds", 1.0)),
        max_retry_seconds=float(judge_cfg.get("max_retry_seconds", 30.0)),
    )


def resolve_plain_subcall(config: dict[str, Any], *, endpoints_path: Path) -> PlainSubcallConfig | None:
    subcall_cfg = config.get("llm_subcall")
    if not subcall_cfg:
        return None

    provider = str(subcall_cfg.get("provider", "openai_compatible"))
    if provider == "vertex":
        project_env = str(subcall_cfg.get("vertex_project_env", "GOOGLE_CLOUD_PROJECT"))
        return PlainSubcallConfig(
            provider="vertex",
            model=str(subcall_cfg.get("model", "gemini-3.1-flash-lite")),
            vertex_project_env=project_env,
            vertex_project=os.environ.get(project_env),
            vertex_location=str(subcall_cfg.get("vertex_location", os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))),
            thinking_level=subcall_cfg.get("thinking_level", "medium"),
            empty_response_max_attempts=int(subcall_cfg.get("empty_response_max_attempts", 1)),
            empty_response_base_retry_seconds=float(subcall_cfg.get("empty_response_base_retry_seconds", 1.0)),
            empty_response_max_retry_seconds=float(subcall_cfg.get("empty_response_max_retry_seconds", 30.0)),
        )

    if provider != "openai_compatible":
        raise ValueError("llm_subcall.provider must be either 'openai_compatible' or 'vertex'")

    endpoint = resolve_endpoint(subcall_cfg, endpoints_path=endpoints_path)
    return PlainSubcallConfig(provider="openai_compatible", model=endpoint.model, endpoint=endpoint)


def _extract_by_path(row: dict[str, Any], path: str) -> object:
    current: object = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _pick_first(row: dict[str, Any], keys: list[str]) -> object:
    for key in keys:
        value = _extract_by_path(row, key)
        if value is not None:
            return value
    return None


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def load_examples(dataset_cfg: dict[str, Any]) -> list[Example]:
    load_dataset = load_dataset_fn()
    parse_answer_candidates = load_rlm_modules()["parse_answer_candidates"]
    dataset_id = str(dataset_cfg["dataset_id"])
    split = str(dataset_cfg.get("split", "eval"))
    dataset_config = dataset_cfg.get("dataset_config")
    revision = dataset_cfg.get("dataset_revision")
    max_examples = int(dataset_cfg.get("max_examples", -1))
    seed = int(dataset_cfg.get("seed", 42))

    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision

    if dataset_config:
        dataset = load_dataset(dataset_id, dataset_config, **kwargs)
    else:
        dataset = load_dataset(dataset_id, **kwargs)

    row_count = len(dataset)
    row_indices = list(range(row_count))
    requested_source_ids = [str(item) for item in dataset_cfg.get("source_ids", [])]
    hard_subset = str(dataset_cfg.get("hard_subset", "")).strip()
    if requested_source_ids:
        source_index_by_id: dict[str, int] = {}
        for index in row_indices:
            row = dataset[int(index)]
            source_id = _stringify(_pick_first(row, ["id", "example_id", "uid", "source_id"])).strip()
            if not source_id:
                source_id = str(index)
            source_index_by_id[source_id] = index
        missing_ids = [source_id for source_id in requested_source_ids if source_id not in source_index_by_id]
        if missing_ids:
            raise ValueError(f"Requested source_ids were not found in the dataset: {missing_ids}")
        row_indices = [source_index_by_id[source_id] for source_id in requested_source_ids]
    elif hard_subset:
        if hard_subset != "frames_multireasoning":
            raise ValueError(f"Unsupported dataset hard_subset: {hard_subset!r}")

        def hard_score(index: int) -> tuple[int, int, int]:
            row = dataset[int(index)]
            if str(row.get("dataset", "")) != "frames":
                return (-1, -1, -1)
            metadata = row.get("metadata")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            reasoning = str(metadata.get("reasoning_types") or "")
            reasoning_count = len([part for part in reasoning.split("|") if part.strip()])
            n_docs = int(metadata.get("n_docs") or 0)
            context_tokens = int(row.get("context_token_count") or metadata.get("context_tokens_est") or 0)
            return (reasoning_count, n_docs, context_tokens)

        row_indices = [index for index in row_indices if hard_score(index)[0] >= 3]
        row_indices.sort(key=lambda index: hard_score(index), reverse=True)
        if not row_indices:
            raise ValueError(f"No examples matched hard_subset={hard_subset!r}.")

    rng = random.Random(seed)
    if not hard_subset and not requested_source_ids:
        rng.shuffle(row_indices)
    if max_examples > 0 and not requested_source_ids:
        row_indices = row_indices[:max_examples]

    selected_dataset = dataset.select(row_indices)
    rows = zip(row_indices, selected_dataset, strict=True)

    examples: list[Example] = []
    for index, (source_index, row) in enumerate(rows):
        question = _stringify(_pick_first(row, ["question", "prompt", "query", "instruction", "task"])).strip()
        if not question:
            continue
        acceptable_answers = parse_answer_candidates(
            _pick_first(
                row,
                [
                    "acceptable_answers",
                    "answers",
                    "answer",
                    "target",
                    "expected_answer",
                    "solution",
                ],
            )
        )
        if not acceptable_answers:
            continue
        context = _stringify(
            _pick_first(row, ["context", "context_payload", "input", "document", "passage", "metadata.context"])
        )
        source_id = _stringify(_pick_first(row, ["id", "example_id", "uid", "source_id"])).strip()
        if not source_id:
            source_id = str(source_index)
        examples.append(
            Example(
                example_id=index,
                source_id=source_id,
                question=question,
                answer=acceptable_answers[0],
                acceptable_answers=acceptable_answers,
                context=context,
                info=dict(row),
            )
        )
    if not examples:
        raise ValueError("No usable examples were loaded from the dataset.")
    return examples


def is_exact_match(predicted_answer: str, expected_answers: list[str]) -> bool:
    normalize_text = load_rlm_modules()["normalize_text"]
    predicted = normalize_text(predicted_answer)
    return any(predicted == normalize_text(answer) for answer in expected_answers if str(answer).strip())


def format_expected_answers(answers: list[str]) -> str:
    return "\n".join(f"- {answer}" for answer in answers)


def parse_binary_judge_score(raw_text: str) -> float:
    text = raw_text.strip()
    if text in {"0", "1"}:
        return float(text)
    if "1" in text and "0" not in text:
        return 1.0
    if "0" in text:
        return 0.0
    raise ValueError(f"Judge response did not contain a valid binary score: {raw_text!r}")


def _exception_status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            value = getattr(response, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
    return None


def _is_retryable_exception(exc: BaseException) -> bool:
    status_code = _exception_status_code(exc)
    if status_code == 429 or status_code in {500, 502, 503, 504}:
        return True
    marker = f"{type(exc).__name__}: {exc}".upper()
    return any(
        token in marker
        for token in (
            "429",
            "RATE_LIMIT",
            "RESOURCE_EXHAUSTED",
            "TOO MANY REQUESTS",
            "UNAVAILABLE",
            "SERVICE UNAVAILABLE",
            "DEADLINE_EXCEEDED",
            "INTERNAL",
        )
    )


def _sleep_before_retry(*, attempt: int, base_seconds: float, max_seconds: float) -> None:
    delay = min(max_seconds, base_seconds * (2**attempt))
    jitter = random.uniform(0.0, min(1.0, delay * 0.25))
    time.sleep(delay + jitter)


def call_with_retries(request, *, max_attempts: int = 6, base_seconds: float = 1.0, max_seconds: float = 30.0):
    for attempt in range(max_attempts):
        try:
            return request()
        except Exception as exc:
            if attempt == max_attempts - 1 or not _is_retryable_exception(exc):
                raise
            _sleep_before_retry(attempt=attempt, base_seconds=base_seconds, max_seconds=max_seconds)
    raise RuntimeError("unreachable retry state")


def _judge_answer_openai(
    *,
    judge_config: JudgeConfig,
    question: str,
    expected_answers: list[str],
    predicted_answer: str,
) -> tuple[float, str]:
    assert judge_config.endpoint is not None
    OpenAI = load_openai_client()
    judge_endpoint = judge_config.endpoint
    client = OpenAI(base_url=judge_endpoint.url, api_key=judge_endpoint.api_key or "EMPTY")
    judge_prompt = JUDGE_PROMPT.format(
        question=question,
        expected_answers=format_expected_answers(expected_answers),
        predicted_answer=predicted_answer,
    )
    response = call_with_retries(
        lambda: client.chat.completions.create(
            model=judge_endpoint.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
            max_tokens=8,
            extra_body={"reasoning": {"enabled": False}},
        ),
        max_attempts=judge_config.max_attempts,
        base_seconds=judge_config.base_retry_seconds,
        max_seconds=judge_config.max_retry_seconds,
    )
    raw = (response.choices[0].message.content or "").strip()
    return parse_binary_judge_score(raw), raw


def _normalise_vertex_model_name(model_name: str) -> str:
    if model_name.startswith("google/"):
        model_name = model_name.removeprefix("google/")
    if model_name == "gemini-3-flash":
        return "gemini-3-flash-preview"
    return model_name


def _judge_answer_vertex(
    *,
    judge_config: JudgeConfig,
    question: str,
    expected_answers: list[str],
    predicted_answer: str,
) -> tuple[float, str]:
    if not judge_config.vertex_project:
        raise ValueError(f"{judge_config.vertex_project_env} must be set for Vertex judging.")
    genai, types = load_google_genai()
    client = genai.Client(
        vertexai=True,
        project=judge_config.vertex_project,
        location=judge_config.vertex_location,
    )
    config_kwargs: dict[str, Any] = {
        "system_instruction": JUDGE_SYSTEM_PROMPT,
        "temperature": 0,
        "max_output_tokens": 1024,
    }
    if judge_config.thinking_level:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=judge_config.thinking_level)
    judge_prompt = JUDGE_PROMPT.format(
        question=question,
        expected_answers=format_expected_answers(expected_answers),
        predicted_answer=predicted_answer,
    )
    response = call_with_retries(
        lambda: client.models.generate_content(
            model=_normalise_vertex_model_name(judge_config.model),
            contents=judge_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        ),
        max_attempts=judge_config.max_attempts,
        base_seconds=judge_config.base_retry_seconds,
        max_seconds=judge_config.max_retry_seconds,
    )
    raw = str(getattr(response, "text", "") or "").strip()
    return parse_binary_judge_score(raw), raw


def judge_answer(
    *,
    judge_config: JudgeConfig,
    question: str,
    expected_answers: list[str],
    predicted_answer: str,
) -> tuple[float, str]:
    if judge_config.provider == "vertex":
        return _judge_answer_vertex(
            judge_config=judge_config,
            question=question,
            expected_answers=expected_answers,
            predicted_answer=predicted_answer,
        )
    return _judge_answer_openai(
        judge_config=judge_config,
        question=question,
        expected_answers=expected_answers,
        predicted_answer=predicted_answer,
    )


def init_state(
    *,
    endpoint: EndpointConfig,
    runtime_config: Any,
    plain_subcall_config: PlainSubcallConfig | None = None,
    disable_recursive_subcalls: bool = False,
) -> tuple[dict[str, Any], Any, Any, Any]:
    modules = load_rlm_modules()
    SyncInferenceSession = modules["SyncInferenceSession"]
    RecursiveRuntime = modules["RecursiveRuntime"]
    create_repl = modules["create_repl"]
    state: dict[str, Any] = {
        "used_repl": False,
        "used_recursion": False,
        "used_llm_subcalls": False,
        "used_rlm_subcalls": False,
        "max_depth_reached": 0,
        "num_subcalls": 0,
        "num_llm_subcalls": 0,
        "num_rlm_subcalls": 0,
        "total_model_tokens": 0.0,
        "total_env_tokens": 0.0,
        "total_prompt_tokens": 0.0,
        "total_completion_tokens": 0.0,
        "total_rollout_tokens": 0.0,
        "rlm_segments": [],
        "rlm_trace": [],
        "rlm_segment_counter": 0,
        "rlm_call_counter": 1,
        "current_call_depth": 0,
        "current_call_id": 0,
        "current_parent_call_id": None,
        "current_branch_max_depth": runtime_config.max_depth,
        "subcall_budget_enabled": bool(runtime_config.subcall_budget_enabled),
        "subcall_budget_total": int(runtime_config.max_total_subcalls),
        "subcall_budget_remaining": int(runtime_config.max_total_subcalls),
        "subcall_budget_exhausted": False,
        "final_answer": None,
        "sampling_temperature": runtime_config.temperature,
    }
    session = SyncInferenceSession(
        base_url=endpoint.url,
        api_key=endpoint.api_key,
        default_headers=None,
        model_name=endpoint.model,
        tokenizer_name=runtime_config.tokenizer_name,
        max_prompt_tokens=runtime_config.max_prompt_tokens,
        request_logprobs=False,
        retry_transient_errors=True,
        openai_extra_body=endpoint.extra_body,
        enable_token_accounting=False,
    )
    state["_sync_session"] = session
    if plain_subcall_config is not None:
        if plain_subcall_config.provider == "vertex":
            VertexGeminiSession = modules["VertexGeminiSession"]
            if not plain_subcall_config.vertex_project:
                raise ValueError(f"{plain_subcall_config.vertex_project_env} must be set for Vertex llm_subcall.")
            state["_plain_llm_session"] = VertexGeminiSession(
                model_name=plain_subcall_config.model,
                project=plain_subcall_config.vertex_project,
                location=plain_subcall_config.vertex_location,
                tokenizer_name=runtime_config.tokenizer_name,
                max_prompt_tokens=runtime_config.max_prompt_tokens,
                thinking_level=plain_subcall_config.thinking_level,
                empty_response_max_attempts=plain_subcall_config.empty_response_max_attempts,
                empty_response_base_retry_seconds=plain_subcall_config.empty_response_base_retry_seconds,
                empty_response_max_retry_seconds=plain_subcall_config.empty_response_max_retry_seconds,
            )
        else:
            if plain_subcall_config.endpoint is None:
                raise ValueError("OpenAI-compatible llm_subcall requires an endpoint config.")
            state["_plain_llm_session"] = SyncInferenceSession(
                base_url=plain_subcall_config.endpoint.url,
                api_key=plain_subcall_config.endpoint.api_key,
                default_headers=None,
                model_name=plain_subcall_config.endpoint.model,
                tokenizer_name=runtime_config.tokenizer_name,
                max_prompt_tokens=runtime_config.max_prompt_tokens,
                request_logprobs=False,
                retry_transient_errors=True,
                openai_extra_body=plain_subcall_config.endpoint.extra_body,
                enable_token_accounting=False,
            )
    runtime = RecursiveRuntime(state, runtime_config)
    state["_runtime"] = runtime
    if disable_recursive_subcalls:
        recursive_fn = runtime._plain_query
        recursive_batch_fn = lambda prompts, model, max_depth, max_workers: runtime.run_plain_query_batch(
            prompts,
            model=model,
            max_workers=max_workers,
        )
    else:
        recursive_fn = runtime._recursive_query
        recursive_batch_fn = lambda prompts, model, max_depth, max_workers: runtime.run_recursive_query_batch(
            prompts,
            model=model,
            max_depth=max_depth,
            max_workers=max_workers,
        )
    repl = create_repl(
        backend=runtime_config.repl_backend,
        backend_kwargs=runtime_config.repl_backend_kwargs,
        context_payload="",
        llm_query_fn=runtime._plain_query,
        rlm_query_fn=recursive_fn,
        llm_query_batch_fn=lambda prompts, model, max_workers: runtime.run_plain_query_batch(
            prompts,
            model=model,
            max_workers=max_workers,
        ),
        rlm_query_batch_fn=recursive_batch_fn,
    )
    state["_root_repl"] = repl
    return state, session, runtime, repl


def run_rollout(
    *,
    example: Example,
    endpoint: EndpointConfig,
    runtime_config: Any,
    prompt_variant: str,
    judge_config: JudgeConfig | None,
    plain_subcall_config: PlainSubcallConfig | None = None,
    disable_recursive_subcalls: bool = False,
) -> dict[str, Any]:
    modules = load_rlm_modules()
    CodeBlock = modules["CodeBlock"]
    RLMIteration = modules["RLMIteration"]
    build_system_prompt = modules["build_system_prompt"]
    build_user_prompt = modules["build_user_prompt"]
    create_repl = modules["create_repl"]
    find_code_blocks = modules["find_code_blocks"]
    find_final_answer = modules["find_final_answer"]
    make_feedback_messages = modules["make_feedback_messages"]
    state, session, runtime, repl = init_state(
        endpoint=endpoint,
        runtime_config=runtime_config,
        plain_subcall_config=plain_subcall_config,
        disable_recursive_subcalls=disable_recursive_subcalls,
    )
    state["_root_context"] = example.context
    if hasattr(repl, "_env"):
        pass
    if disable_recursive_subcalls:
        recursive_fn = runtime._plain_query
        recursive_batch_fn = lambda prompts, model, max_depth, max_workers: runtime.run_plain_query_batch(
            prompts,
            model=model,
            max_workers=max_workers,
        )
    else:
        recursive_fn = runtime._recursive_query
        recursive_batch_fn = lambda prompts, model, max_depth, max_workers: runtime.run_recursive_query_batch(
            prompts,
            model=model,
            max_depth=max_depth,
            max_workers=max_workers,
        )

    state["_root_repl"] = create_repl(
        backend=runtime_config.repl_backend,
        backend_kwargs=runtime_config.repl_backend_kwargs,
        context_payload=example.context,
        llm_query_fn=runtime._plain_query,
        rlm_query_fn=recursive_fn,
        llm_query_batch_fn=lambda prompts, model, max_workers: runtime.run_plain_query_batch(
            prompts,
            model=model,
            max_workers=max_workers,
        ),
        rlm_query_batch_fn=recursive_batch_fn,
    )
    repl = state["_root_repl"]

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_prompt(
                depth=0,
                max_depth=runtime_config.max_depth,
                prompt_variant=prompt_variant,
                max_prompt_tokens=runtime_config.max_prompt_tokens,
                turn_max_tokens=runtime_config.turn_max_tokens,
                subcall_max_tokens=runtime_config.subcall_max_tokens,
                subcall_budget_enabled=runtime_config.subcall_budget_enabled,
                max_total_subcalls=runtime_config.max_total_subcalls,
                max_batched_subcalls=runtime_config.max_batched_subcalls,
                include_budget_reminder=runtime_config.include_budget_reminder,
            ),
        },
        build_context_metadata_message(example.context),
        {"role": "user", "content": example.question},
    ]
    root_steps: list[dict[str, Any]] = []
    forced_finalize = False
    error: str | None = None
    start_time = time.perf_counter()

    try:
        for assistant_turn in range(runtime_config.max_iterations + 1):
            assistant_text, payload = session.generate(
                messages=messages,
                max_tokens=runtime_config.turn_max_tokens,
                temperature=runtime_config.temperature,
                top_p=runtime_config.top_p,
            )

            segment_kind = "root_finalize_turn" if forced_finalize else "root_turn"
            runtime._append_segment(
                payload=payload,
                depth=0,
                turn_index=assistant_turn,
                kind=segment_kind,
                train_scope="root_turn",
                is_trainable_rlm_turn=True,
                response_source="root",
                response_text=assistant_text,
                messages=messages,
                call_id=0,
                parent_call_id=None,
            )

            code_block_strs = [block.strip() for block in find_code_blocks(assistant_text)]
            code_blocks: list[CodeBlock] = []
            execution_results: list[dict[str, Any]] = []
            if code_block_strs:
                state["used_repl"] = True

            for code in code_block_strs:
                execution = repl.execute_code(code)
                code_blocks.append(CodeBlock(code=code, result=execution))
                execution_results.append(audit_repl_execution(code, execution))
                for child_call in execution.rlm_calls:
                    metadata = child_call.metadata or {}
                    if metadata.get("kind") == "recursive_query":
                        state["used_recursion"] = True
                if execution.final_answer is not None and state.get("final_answer") is None:
                    state["final_answer"] = execution.final_answer

            if state.get("final_answer") is None:
                found = find_final_answer(assistant_text, environment=repl)
                if found is not None:
                    state["final_answer"] = found

            iteration = RLMIteration(prompt=messages, response=assistant_text, code_blocks=code_blocks)
            feedback_messages = make_feedback_messages(
                iteration,
                max_chars=runtime_config.execution_output_char_limit,
            )

            root_steps.append(
                {
                    "prompt_messages": audit_messages(messages),
                    "request": {
                        "model": session.model_name,
                        "max_tokens": runtime_config.turn_max_tokens,
                        "temperature": runtime_config.temperature,
                        "top_p": runtime_config.top_p,
                    },
                    "assistant": assistant_text,
                    "code_blocks": code_block_strs,
                    "execution_results": execution_results,
                    "feedback": [message["content"] for message in feedback_messages],
                    "feedback_messages": audit_messages(feedback_messages),
                    "final_answer": state.get("final_answer"),
                    "forced_finalize": forced_finalize,
                }
            )

            if state.get("final_answer") is not None:
                break

            if forced_finalize:
                state["final_answer"] = assistant_text.strip()
                root_steps[-1]["final_answer"] = state["final_answer"]
                break

            next_iteration = assistant_turn + 1
            if next_iteration >= runtime_config.max_iterations:
                next_message = {"role": "user", "content": runtime.build_finalize_message()}
                forced_finalize = True
            else:
                next_message = build_user_prompt(
                    root_prompt=example.question,
                    iteration=next_iteration,
                    context_count=int(repl.get_context_count()),
                    history_count=int(repl.get_history_count()),
                )

            response_messages = [*feedback_messages, next_message]
            state["total_env_tokens"] += float(
                sum(session.count_text_tokens(message["content"]) for message in response_messages)
            )
            messages.append({"role": "assistant", "content": assistant_text})
            messages.extend(response_messages)
        else:
            state["final_answer"] = ""
    except Exception as exc:  # pragma: no cover - defensive capture for real runs
        error = str(exc)
    finally:
        close = getattr(repl, "close", None)
        if callable(close):
            close()

    final_answer = "" if state.get("final_answer") is None else str(state["final_answer"]).strip()
    exact_match = is_exact_match(final_answer, example.acceptable_answers) if final_answer else False
    judge_score = None
    judge_raw_response = None
    if final_answer and judge_config is not None and not exact_match:
        try:
            judge_score, judge_raw_response = judge_answer(
                judge_config=judge_config,
                question=example.question,
                expected_answers=example.acceptable_answers,
                predicted_answer=final_answer,
            )
        except Exception as exc:  # pragma: no cover - best-effort judging
            judge_raw_response = f"[judge_error] {exc}"

    return {
        "example_id": example.example_id,
        "source_id": example.source_id,
        "question": example.question,
        "acceptable_answers": example.acceptable_answers,
        "answer": example.answer,
        "context_length": len(example.context),
        "prompt_variant": prompt_variant,
        "endpoint": redacted_dataclass_dict(endpoint),
        "final_answer": final_answer,
        "exact_match": exact_match,
        "judge_score": judge_score,
        "judge_raw_response": judge_raw_response,
        "used_repl": bool(state["used_repl"]),
        "used_recursion": bool(state["used_recursion"]),
        "used_llm_subcalls": bool(state["used_llm_subcalls"]),
        "used_rlm_subcalls": bool(state["used_rlm_subcalls"]),
        "num_subcalls": int(state["num_subcalls"]),
        "num_llm_subcalls": int(state["num_llm_subcalls"]),
        "num_rlm_subcalls": int(state["num_rlm_subcalls"]),
        "max_depth_reached": int(state["max_depth_reached"]),
        "total_model_tokens": float(state["total_model_tokens"]),
        "total_env_tokens": float(state["total_env_tokens"]),
        "total_prompt_tokens": float(state.get("total_prompt_tokens", 0.0)),
        "total_completion_tokens": float(state.get("total_completion_tokens", 0.0)),
        "total_rollout_tokens": float(state.get("total_rollout_tokens", 0.0)),
        "root_steps": root_steps,
        "trace": state["rlm_trace"],
        "segments": state["rlm_segments"],
        "error": error,
        "elapsed_seconds": time.perf_counter() - start_time,
    }


def make_error_record(
    *,
    example: Example,
    endpoint: EndpointConfig,
    prompt_variant: str,
    error: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "source_id": example.source_id,
        "question": example.question,
        "acceptable_answers": example.acceptable_answers,
        "answer": example.answer,
        "context_length": len(example.context),
        "prompt_variant": prompt_variant,
        "endpoint": redacted_dataclass_dict(endpoint),
        "final_answer": "",
        "exact_match": False,
        "judge_score": None,
        "judge_raw_response": None,
        "used_repl": False,
        "used_recursion": False,
        "used_llm_subcalls": False,
        "used_rlm_subcalls": False,
        "num_subcalls": 0,
        "num_llm_subcalls": 0,
        "num_rlm_subcalls": 0,
        "max_depth_reached": 0,
        "total_model_tokens": 0.0,
        "total_env_tokens": 0.0,
        "total_prompt_tokens": 0.0,
        "total_completion_tokens": 0.0,
        "total_rollout_tokens": 0.0,
        "root_steps": [],
        "trace": [],
        "segments": [],
        "error": error,
        "elapsed_seconds": elapsed_seconds,
    }


def run_rollout_from_payload(
    *,
    example: Example,
    endpoint: EndpointConfig,
    runtime_config_payload: dict[str, Any],
    prompt_variant: str,
    judge_config: JudgeConfig | None,
    plain_subcall_config: PlainSubcallConfig | None = None,
    disable_recursive_subcalls: bool = False,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    try:
        RuntimeConfig = load_rlm_modules()["RuntimeConfig"]
        runtime_config = RuntimeConfig(**runtime_config_payload)
        return run_rollout(
            example=example,
            endpoint=endpoint,
            runtime_config=runtime_config,
            prompt_variant=prompt_variant,
            judge_config=judge_config,
            plain_subcall_config=plain_subcall_config,
            disable_recursive_subcalls=disable_recursive_subcalls,
        )
    except Exception as exc:  # pragma: no cover - defensive capture for production process workers
        return make_error_record(
            example=example,
            endpoint=endpoint,
            prompt_variant=prompt_variant,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - start_time,
        )


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    exact_matches = [1.0 if record["exact_match"] else 0.0 for record in records]
    token_counts = [float(record["total_model_tokens"]) for record in records]
    subcalls = [int(record["num_subcalls"]) for record in records]
    recursion = [1.0 if record["used_recursion"] else 0.0 for record in records]
    judge_scores = [record["judge_score"] for record in records if record["judge_score"] is not None]

    summary = {
        "num_records": len(records),
        "exact_match_rate": statistics.mean(exact_matches) if exact_matches else 0.0,
        "mean_total_model_tokens": statistics.mean(token_counts) if token_counts else 0.0,
        "mean_num_subcalls": statistics.mean(subcalls) if subcalls else 0.0,
        "used_recursion_rate": statistics.mean(recursion) if recursion else 0.0,
        "num_errors": sum(1 for record in records if record["error"]),
    }
    if judge_scores:
        summary["mean_judge_score"] = statistics.mean(judge_scores)
    return summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def redacted_dataclass_dict(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    if "api_key" in payload:
        payload["api_key"] = "<redacted>"
    endpoint = payload.get("endpoint")
    if isinstance(endpoint, dict) and "api_key" in endpoint:
        endpoint["api_key"] = "<redacted>"
    return payload


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def record_is_accepted(record: dict[str, Any]) -> bool:
    return not record.get("error") and (bool(record.get("exact_match")) or record.get("judge_score") == 1.0)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def render_progress_bar(completed: int, total: int, *, width: int = 28) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = min(width, max(0, int(round(width * completed / total))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_progress_line(
    *,
    label: str,
    records: list[dict[str, Any]],
    total: int,
    initial_completed: int,
    started_at: float,
    final: bool = False,
) -> None:
    completed = len(records)
    new_completed = max(0, completed - initial_completed)
    elapsed = max(0.0, time.perf_counter() - started_at)
    remaining = max(0, total - completed)
    percent = (completed / total * 100.0) if total else 100.0
    rate = (new_completed / elapsed) if elapsed > 0 and new_completed > 0 else 0.0
    eta = (remaining / rate) if rate > 0 and remaining > 0 else (0.0 if remaining == 0 else None)
    accepted = sum(1 for record in records if record_is_accepted(record))
    errors = sum(1 for record in records if record.get("error"))
    subcalls = sum(int(record.get("num_llm_subcalls") or 0) for record in records)
    bar = render_progress_bar(completed, total)
    suffix = "\n" if final else ""
    line = (
        f"\r{label} {bar} {completed}/{total} {percent:5.1f}% "
        f"accepted={accepted} errors={errors} llm_subcalls={subcalls} "
        f"rate={rate * 60.0:5.1f}/min eta={fmt_duration(eta)}"
    )
    print(line, end=suffix, file=sys.stderr, flush=True)


def render_comparison_markdown(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Prompt Comparison",
        "",
        "| Prompt Variant | Exact Match | Judge Score | Mean Tokens | Mean Subcalls | Recursion Rate | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for prompt_variant, summary in sorted(results.items()):
        lines.append(
            "| {name} | {exact:.3f} | {judge} | {tokens:.1f} | {subcalls:.2f} | {recursion:.3f} | {errors} |".format(
                name=prompt_variant,
                exact=summary["exact_match_rate"],
                judge=f"{summary['mean_judge_score']:.3f}" if "mean_judge_score" in summary else "-",
                tokens=summary["mean_total_model_tokens"],
                subcalls=summary["mean_num_subcalls"],
                recursion=summary["used_recursion_rate"],
                errors=summary["num_errors"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_toml(config_path)
    RuntimeConfig = load_rlm_modules()["RuntimeConfig"]

    endpoints_path = resolve_path(
        str(config.get("endpoints_path", "configs/endpoints.toml")),
        relative_to=config_path.parent,
    )
    output_root = resolve_path(str(config.get("output_dir", "outputs/rlm_traces")), relative_to=config_path.parent)
    run_name = str(config.get("run_name", config_path.stem))
    run_dir = output_root / run_name

    model_endpoint = resolve_endpoint(config["model"], endpoints_path=endpoints_path)
    judge_config = resolve_judge(config, endpoints_path=endpoints_path)
    plain_subcall_config = resolve_plain_subcall(config, endpoints_path=endpoints_path)

    rollout_cfg = config["rollout"]
    disable_recursive_subcalls = bool(rollout_cfg.get("disable_recursive_subcalls", False))
    configured_max_depth = int(rollout_cfg.get("max_depth", 2))
    runtime_max_depth = max(0, configured_max_depth - 1) if disable_recursive_subcalls else configured_max_depth
    runtime_config = RuntimeConfig(
        max_depth=runtime_max_depth,
        max_iterations=int(rollout_cfg.get("max_iterations", 4)),
        turn_max_tokens=int(rollout_cfg.get("turn_max_tokens", 192)),
        subcall_max_tokens=int(rollout_cfg.get("subcall_max_tokens", 128)),
        max_prompt_tokens=(
            int(rollout_cfg["max_prompt_tokens"]) if rollout_cfg.get("max_prompt_tokens") is not None else None
        ),
        temperature=float(rollout_cfg.get("temperature", 1.0)),
        top_p=float(rollout_cfg.get("top_p", 1.0)),
        tokenizer_name=rollout_cfg.get("tokenizer_name"),
        inference_mode="local",
        inference_base_url=model_endpoint.url,
        inference_api_key=model_endpoint.api_key,
        llm_subcall_empty_response_max_attempts=int(
            config.get("llm_subcall", {}).get("empty_response_max_attempts", 1)
        ),
        llm_subcall_empty_response_base_retry_seconds=float(
            config.get("llm_subcall", {}).get("empty_response_base_retry_seconds", 1.0)
        ),
        llm_subcall_empty_response_max_retry_seconds=float(
            config.get("llm_subcall", {}).get("empty_response_max_retry_seconds", 30.0)
        ),
        repl_backend=str(rollout_cfg.get("repl_backend", "local")),
        repl_timeout_seconds=rollout_cfg.get("repl_timeout_seconds"),
        repl_fast_timeout_seconds=rollout_cfg.get("repl_fast_timeout_seconds"),
        prompt_variant=str((rollout_cfg.get("prompt_variants") or ["sanjaya_text_v1"])[0]),
        subcall_budget_enabled=bool(rollout_cfg.get("subcall_budget_enabled", True)),
        max_total_subcalls=int(rollout_cfg.get("max_total_subcalls", 60)),
        max_batched_subcalls=int(rollout_cfg.get("max_batched_subcalls", rollout_cfg.get("max_total_subcalls", 60))),
        capture_prompt_messages=True,
        include_budget_reminder=bool(rollout_cfg.get("include_budget_reminder", True)),
        recursive_rlm_batch_mode=str(rollout_cfg.get("recursive_rlm_batch_mode", "serial")),
        live_trace_dir=rollout_cfg.get("live_trace_dir"),
    )
    prompt_variants = [str(item) for item in rollout_cfg.get("prompt_variants", ["sanjaya_text_v1"])]
    max_workers = max(1, int(config.get("max_workers", rollout_cfg.get("max_workers", 1))))
    worker_backend = str(config.get("worker_backend", rollout_cfg.get("worker_backend", "process"))).lower()
    if max_workers == 1:
        worker_backend = "serial"
    if worker_backend not in {"serial", "process"}:
        raise ValueError("worker_backend must be either 'serial' or 'process'.")
    resume = bool(config.get("resume", True))
    progress_enabled = bool(config.get("progress", True))

    examples = load_examples(config["dataset"])
    write_json(
        run_dir / "run_config.json",
        {
            "config_path": str(config_path),
            "model_endpoint": redacted_dataclass_dict(model_endpoint),
            "llm_subcall": redacted_dataclass_dict(plain_subcall_config) if plain_subcall_config is not None else None,
            "judge": redacted_dataclass_dict(judge_config) if judge_config is not None else None,
            "prompt_variants": prompt_variants,
            "num_examples": len(examples),
            "configured_max_depth": configured_max_depth,
            "runtime_max_depth": runtime_max_depth,
            "disable_recursive_subcalls": disable_recursive_subcalls,
            "max_workers": max_workers,
            "worker_backend": worker_backend,
        },
    )

    comparison: dict[str, dict[str, Any]] = {}
    for prompt_variant in prompt_variants:
        variant_runtime = RuntimeConfig(**asdict(runtime_config))
        variant_runtime.prompt_variant = prompt_variant
        variant_dir = run_dir / prompt_variant
        records_path = variant_dir / "records.jsonl"
        successful_path = variant_dir / "successful_records.jsonl"
        records: list[dict[str, Any]] = read_jsonl(records_path) if resume else []
        successful: list[dict[str, Any]] = [record for record in records if not record.get("error")]
        if not resume:
            write_jsonl(records_path, [])
            write_jsonl(successful_path, [])
        elif records:
            write_jsonl(successful_path, successful)
        completed_ids = {str(record.get("source_id", record.get("example_id"))) for record in records if not record.get("error")}
        pending_examples = [example for example in examples if str(example.source_id) not in completed_ids]
        progress_started_at = time.perf_counter()
        progress_initial_completed = len(records)
        progress_label = f"{run_name}/{prompt_variant}"

        def _write_progress() -> None:
            summary = summarize(records)
            comparison[prompt_variant] = summary
            write_json(variant_dir / "summary.json", summary)
            write_json(run_dir / "comparison.json", comparison)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "comparison.md").write_text(render_comparison_markdown(comparison))
            if progress_enabled:
                print_progress_line(
                    label=progress_label,
                    records=records,
                    total=len(examples),
                    initial_completed=progress_initial_completed,
                    started_at=progress_started_at,
                )

        _write_progress()
        if max_workers == 1:
            for example in pending_examples:
                record = run_rollout(
                    example=example,
                    endpoint=model_endpoint,
                    runtime_config=variant_runtime,
                    prompt_variant=prompt_variant,
                    judge_config=judge_config,
                    plain_subcall_config=plain_subcall_config,
                    disable_recursive_subcalls=disable_recursive_subcalls,
                )
                records.append(record)
                append_jsonl(records_path, record)
                if not record["error"]:
                    successful.append(record)
                    append_jsonl(successful_path, record)
                _write_progress()
        else:
            if worker_backend != "process":
                raise ValueError("max_workers > 1 requires worker_backend = 'process' for local REPL safety.")

            mp_context = multiprocessing.get_context("spawn")
            pending_iter = iter(pending_examples)
            in_flight: set[Any] = set()
            runtime_config_payload = asdict(variant_runtime)

            def _submit_next(pool: ProcessPoolExecutor) -> bool:
                try:
                    example = next(pending_iter)
                except StopIteration:
                    return False
                in_flight.add(
                    pool.submit(
                        run_rollout_from_payload,
                        example=example,
                        endpoint=model_endpoint,
                        runtime_config_payload=runtime_config_payload,
                        prompt_variant=prompt_variant,
                        judge_config=judge_config,
                        plain_subcall_config=plain_subcall_config,
                        disable_recursive_subcalls=disable_recursive_subcalls,
                    )
                )
                return True

            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as pool:
                for _ in range(max_workers):
                    if not _submit_next(pool):
                        break

                while in_flight:
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        record = future.result()
                        records.append(record)
                        append_jsonl(records_path, record)
                        if not record["error"]:
                            successful.append(record)
                            append_jsonl(successful_path, record)
                        _write_progress()
                        _submit_next(pool)
        if progress_enabled:
            print_progress_line(
                label=progress_label,
                records=records,
                total=len(examples),
                initial_completed=progress_initial_completed,
                started_at=progress_started_at,
                final=True,
            )

    write_json(run_dir / "comparison.json", comparison)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "comparison.md").write_text(render_comparison_markdown(comparison))


if __name__ == "__main__":
    main()
