from __future__ import annotations

import asyncio
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI
import verifiers as vf

JUDGE_PROMPT = """You are a binary grader.

You will be given three things:
1. The dataset task.
2. One or more gold reference answers for that task.
3. The candidate answer produced by the model you are evaluating.

Your job is to judge whether the candidate answer is semantically correct with respect to the dataset task and the gold reference answer(s).

Dataset task:
{question}

Gold reference answer(s):
{expected_answers}

Candidate model answer:
{predicted_answer}

How to score:
- Return 1 if the candidate answer clearly conveys the same final answer as any gold reference answer.
- Semantic correctness matters more than exact wording, formatting, quoting style, or JSON formatting.
- Return 1 if the candidate answer contains the correct fact, entity, string, or number, even if extra harmless text is present.
- Return 0 if the candidate answer is missing the core answer, gives the wrong answer, contradicts the correct answer, is only scratchpad/reasoning/code without a clear final answer, or is empty.
- If the candidate answer is truncated, malformed, or noisy, you should return 1 only if the correct final answer is clearly present.
- The only valid outputs are 0 or 1.

Return exactly one character: 0 or 1.
"""

JUDGE_SYSTEM_PROMPT = (
    "You are a strict binary grader. Return exactly one character: 0 or 1. "
    "Do not return JSON, explanations, tool calls, or any other text."
)

_EFFICIENCY_PENALTY_PER_1K_TOKENS = 1000.0
_JUDGE_RETRY_MAX_ATTEMPTS = 6
_JUDGE_RETRY_BASE_SECONDS = 1.0
_JUDGE_RETRY_MAX_SECONDS = 30.0
_VERTEX_JUDGE_MAX_OUTPUT_TOKENS = 1024
_VALID_EFFICIENCY_PENALTY_MODES = {"static_per_1k", "adaptive_group"}
_VALID_ADAPTIVE_COST_BASES = {"total_tokens", "weighted_turn_tokens"}
_VALID_EFFICIENCY_PENALTY_SCOPES = {"correct_only", "all_rollouts"}
_JUDGE_TRUNCATION_MARKER = "\n\n[truncated before semantic judging]"


@dataclass(frozen=True)
class CorrectnessResult:
    predicted_answer: str
    expected_answers: list[str]
    score: float
    raw_response: str
    parse_error: str | None


@dataclass(frozen=True)
class TokenBreakdown:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    trainable_tokens: int
    rlm_turn_tokens: int
    plain_subcall_tokens: int


def _truncate_for_judge(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep_chars = max(0, max_chars - len(_JUDGE_TRUNCATION_MARKER))
    return text[:keep_chars].rstrip() + _JUDGE_TRUNCATION_MARKER


def _is_oversized_judge_exception(exc: BaseException) -> bool:
    error_text = f"{type(exc).__name__}: {exc}".casefold()
    return any(
        marker in error_text
        for marker in (
            "text fields that are too large",
            "input token count",
            "exceeds",
            "too many tokens",
            "request payload size exceeds",
            "payload too large",
        )
    )


def _get_predicted_answer(state: vf.State, completion) -> str:
    final_answer = state.get("final_answer")
    if final_answer is not None:
        return str(final_answer)
    if completion:
        return str(completion[-1].get("content", ""))
    return ""


def _last_rlm_debug(state: vf.State) -> dict[str, Any]:
    trajectory = state.get("trajectory") or []
    if not trajectory:
        return {}
    last_step = trajectory[-1]
    if not isinstance(last_step, dict):
        return {}
    extras = last_step.get("extras") or {}
    debug = extras.get("rlm_debug") or {}
    return debug if isinstance(debug, dict) else {}


def _state_bool_with_debug_fallback(state: vf.State, key: str) -> bool:
    if key in state:
        return bool(state.get(key))
    return bool(_last_rlm_debug(state).get(key))


def _get_expected_answers(answer: str, info: dict[str, Any] | None) -> list[str]:
    if info and info.get("acceptable_answers"):
        return [str(item) for item in info["acceptable_answers"]]
    return [answer]


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
        text = text[1:-1].strip()
    return text


def _canonicalize_json(value: str) -> str | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonicalize_number(value: str) -> str | None:
    candidate = value.replace(",", "").strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", candidate):
        return None

    try:
        normalized = format(Decimal(candidate).normalize(), "f")
    except InvalidOperation:
        return None

    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _equivalent_forms(value: Any) -> set[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return {""}

    forms = {normalized, normalized.casefold()}
    json_form = _canonicalize_json(normalized)
    if json_form is not None:
        forms.add(json_form)

    number_form = _canonicalize_number(normalized)
    if number_form is not None:
        forms.add(number_form)

    return forms


def _is_exact_match(predicted_answer: str, expected_answers: list[str]) -> bool:
    predicted_forms = _equivalent_forms(predicted_answer)
    for expected_answer in expected_answers:
        if predicted_forms & _equivalent_forms(expected_answer):
            return True
    return False


def _parse_binary_judge_score(raw_text: str) -> float:
    text = raw_text.strip()
    if text in {"0", "1"}:
        return float(text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        score = parsed.get("score")
        if score in {0, 1, 0.0, 1.0, "0", "1"}:
            return float(score)
    elif parsed in {0, 1, 0.0, 1.0, "0", "1"}:
        return float(parsed)

    match = re.search(r"\b([01])\b", text)
    if match is not None:
        return float(match.group(1))
    raise ValueError(f"Judge response did not contain a valid binary score: {raw_text!r}")


def _format_expected_answers(answers: list[str]) -> str:
    return "\n".join(f"- {answer}" for answer in answers)


_PAIR_PATTERN = re.compile(r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)")


def _extract_pair_set(value: Any) -> set[tuple[str, str]]:
    if value is None:
        return set()
    if isinstance(value, list):
        texts = [_normalize_text(item) for item in value]
    else:
        text = _normalize_text(value)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            texts = [_normalize_text(item) for item in parsed]
        else:
            texts = [text]

    pairs: set[tuple[str, str]] = set()
    for text in texts:
        for left, right in _PAIR_PATTERN.findall(text):
            first, second = sorted((left, right), key=lambda item: int(item))
            pairs.add((first, second))
    return pairs


def _is_oolong_pairs_task(info: dict[str, Any] | None) -> bool:
    if not info:
        return False
    dataset_name = str(info.get("dataset_name") or "")
    answer_type = str(info.get("answer_type") or "")
    metadata = info.get("metadata") if isinstance(info.get("metadata"), dict) else {}
    origin = str(metadata.get("benchmark_origin") or metadata.get("original_benchmark") or "")
    return (
        dataset_name == "oolong_pairs"
        or origin == "oolong_pairs"
        or "oolong_pairs" in str(info.get("source_task") or "")
        or answer_type == "list_of_answers"
    )


def _score_oolong_pairs(predicted_answer: str, expected_answers: list[str]) -> tuple[float, str, dict[str, float]]:
    predicted_pairs = _extract_pair_set(predicted_answer)
    expected_pairs: set[tuple[str, str]] = set()
    for answer in expected_answers:
        expected_pairs |= _extract_pair_set(answer)
    if not expected_pairs:
        score = 1.0 if not predicted_pairs else 0.0
        stats = {
            "precision": score,
            "recall": score,
            "f1": score,
            "predicted_pairs": float(len(predicted_pairs)),
            "expected_pairs": 0.0,
        }
        return score, "[oolong_pairs_empty_gold]", stats

    true_positive = len(predicted_pairs & expected_pairs)
    precision = true_positive / len(predicted_pairs) if predicted_pairs else 0.0
    recall = true_positive / len(expected_pairs)
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0.0 else 0.0
    stats = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_pairs": float(len(predicted_pairs)),
        "expected_pairs": float(len(expected_pairs)),
    }
    return f1, "[oolong_pairs_f1]", stats


def _record_oolong_pairs_metrics(state: vf.State, stats: dict[str, float]) -> None:
    state["oolong_pairs_precision"] = stats["precision"]
    state["oolong_pairs_recall"] = stats["recall"]
    state["oolong_pairs_f1"] = stats["f1"]
    state["oolong_pairs_predicted_count"] = stats["predicted_pairs"]
    state["oolong_pairs_expected_count"] = stats["expected_pairs"]


def _is_oolong_semantic_aggregation_task(info: dict[str, Any] | None) -> bool:
    if not info:
        return False
    source_task = str(info.get("source_task") or "")
    metadata = info.get("metadata") if isinstance(info.get("metadata"), dict) else {}
    derived_dataset = str(metadata.get("derived_dataset") or "")
    return derived_dataset == "oolong_semantic_agg_v2" or source_task.startswith("TASK_TYPE.SEMANTIC_")


def _strip_answer_prefix(text: str) -> str:
    text = _normalize_text(text)
    match = re.match(r"^(?:count|label|user|month|answer)\s*:\s*(.+)$", text, flags=re.IGNORECASE)
    if match:
        return _normalize_text(match.group(1))
    return text


def _extract_numeric_answer(text: str) -> str | None:
    stripped = _strip_answer_prefix(text)
    direct = _canonicalize_number(stripped)
    if direct is not None:
        return direct
    matches = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", stripped.replace(",", ""))
    if len(matches) == 1:
        return _canonicalize_number(matches[0])
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = _normalize_text(text)
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _canonical_histogram(value: dict[str, Any]) -> dict[str, int] | None:
    result: dict[str, int] = {}
    for key, raw_count in value.items():
        number = _canonicalize_number(str(raw_count))
        if number is None:
            return None
        try:
            decimal = Decimal(number)
        except InvalidOperation:
            return None
        if decimal != decimal.to_integral_value():
            return None
        result[_normalize_text(key)] = int(decimal)
    return result


def _score_oolong_semantic_aggregation(
    predicted_answer: str,
    expected_answers: list[str],
    *,
    answer_type: str,
) -> tuple[float, str, str | None]:
    if answer_type == "ANSWER_TYPE.JSON":
        predicted_json = _extract_json_object(predicted_answer)
        if predicted_json is None:
            return 0.0, "[oolong_semantic_json_mismatch]", "invalid_json"
        predicted_histogram = _canonical_histogram(predicted_json)
        if predicted_histogram is None:
            return 0.0, "[oolong_semantic_json_mismatch]", "invalid_histogram"
        for expected_answer in expected_answers:
            expected_json = _extract_json_object(expected_answer)
            if expected_json is None:
                continue
            expected_histogram = _canonical_histogram(expected_json)
            if expected_histogram is not None and predicted_histogram == expected_histogram:
                return 1.0, "[oolong_semantic_exact]", None
        return 0.0, "[oolong_semantic_json_mismatch]", None

    if answer_type == "ANSWER_TYPE.NUMERIC":
        predicted_number = _extract_numeric_answer(predicted_answer)
        if predicted_number is None:
            return 0.0, "[oolong_semantic_numeric_mismatch]", "invalid_numeric"
        for expected_answer in expected_answers:
            expected_number = _extract_numeric_answer(expected_answer)
            if expected_number is not None and predicted_number == expected_number:
                return 1.0, "[oolong_semantic_exact]", None
        return 0.0, "[oolong_semantic_numeric_mismatch]", None

    predicted_label = _strip_answer_prefix(predicted_answer).casefold()
    for expected_answer in expected_answers:
        if predicted_label == _strip_answer_prefix(expected_answer).casefold():
            return 1.0, "[oolong_semantic_exact]", None
    return 0.0, "[oolong_semantic_label_mismatch]", None


def _record_oolong_semantic_metrics(state: vf.State, score: float) -> None:
    state["oolong_semantic_score"] = float(score)
    state["oolong_semantic_exact"] = 1.0 if score >= 1.0 else 0.0


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        joined = "".join(parts).strip()
        if joined:
            return joined

    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ""


def _load_google_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Vertex Gemini judging requires the google-genai package. "
            "Install environments/rlm_rlvr with google-genai[aiohttp]>=1.51.0."
        ) from exc
    return genai, types


def _normalise_vertex_model_name(model_name: str) -> str:
    if model_name.startswith("google/"):
        model_name = model_name.removeprefix("google/")
    if model_name == "gemini-3-flash":
        return "gemini-3-flash-preview"
    return model_name


def _exception_status_code(exc: BaseException) -> int | None:
    for attr_name in ("status_code", "status", "code"):
        value = getattr(exc, attr_name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    response = getattr(exc, "response", None)
    if response is not None:
        for attr_name in ("status_code", "status"):
            value = getattr(response, attr_name, None)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
    return None


def _is_retryable_judge_exception(exc: BaseException) -> bool:
    status_code = _exception_status_code(exc)
    if status_code == 429 or status_code in {500, 502, 503, 504}:
        return True

    error_text = f"{type(exc).__name__}: {exc}".upper()
    return any(
        marker in error_text
        for marker in (
            "429",
            "RATE_LIMIT",
            "RESOURCE_EXHAUSTED",
            "TOO MANY REQUESTS",
            "UNAVAILABLE",
            "SERVICE UNAVAILABLE",
            "DEADLINE_EXCEEDED",
            "INTERNAL",
            "ACCESS_TOKEN_TYPE_UNSUPPORTED",
        )
    )


async def _sleep_before_judge_retry(attempt: int) -> None:
    delay = min(_JUDGE_RETRY_MAX_SECONDS, _JUDGE_RETRY_BASE_SECONDS * (2**attempt))
    jitter = random.uniform(0.0, min(1.0, delay * 0.25))
    await asyncio.sleep(delay + jitter)


async def _call_judge_with_retries(request: Callable[[], Awaitable[Any]]) -> Any:
    for attempt in range(_JUDGE_RETRY_MAX_ATTEMPTS):
        try:
            return await request()
        except Exception as exc:
            if attempt == _JUDGE_RETRY_MAX_ATTEMPTS - 1 or not _is_retryable_judge_exception(exc):
                raise
            await _sleep_before_judge_retry(attempt)

    raise RuntimeError("unreachable judge retry state")


def _vertex_client_kwargs(*, project: str, location: str, types: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "vertexai": True,
        "project": project,
        "location": location,
    }
    http_options_type = getattr(types, "HttpOptions", None)
    if http_options_type is not None:
        kwargs["http_options"] = http_options_type(api_version="v1")
    return kwargs


async def _close_vertex_client(client: Any) -> None:
    aio_client = getattr(client, "aio", None)
    aclose = getattr(aio_client, "aclose", None)
    if callable(aclose):
        await aclose()
        return

    close = getattr(client, "close", None)
    if callable(close):
        close()


class _VertexJudgeClientFactory:
    def __init__(self, *, project: str, location: str) -> None:
        self.project = project
        self.location = location

    async def generate_content(self, **kwargs: Any) -> Any:
        genai, types = _load_google_genai()
        client = genai.Client(**_vertex_client_kwargs(project=self.project, location=self.location, types=types))
        try:
            return await client.aio.models.generate_content(**kwargs)
        finally:
            await _close_vertex_client(client)


async def _call_vertex_generate_content(judge_client: Any, **kwargs: Any) -> Any:
    generate_content = getattr(judge_client, "generate_content", None)
    if callable(generate_content):
        return await generate_content(**kwargs)
    return await judge_client.aio.models.generate_content(**kwargs)


def _record_judge_payload(
    state: vf.State,
    *,
    predicted_answer: str,
    expected_answers: list[str],
    score: float,
    raw_response: str,
    parse_error: str | None,
) -> None:
    state["judge_predicted_answer"] = predicted_answer
    state["judge_expected_answers"] = list(expected_answers)
    state["judge_score"] = score
    state["judge_raw_response"] = raw_response
    state["judge_parse_error"] = parse_error

    trajectory = state.get("trajectory") or []
    if not trajectory:
        return

    extras = trajectory[-1].setdefault("extras", {})
    rlm_debug = extras.setdefault("rlm_debug", {})
    rlm_debug.update(
        {
            "predicted_answer": predicted_answer,
            "expected_answers": list(expected_answers),
            "judge_score": score,
            "judge_raw_response": raw_response,
            "judge_parse_error": parse_error,
        }
    )


def _missing_formal_final_at_max_turn(state: vf.State) -> bool:
    if _state_bool_with_debug_fallback(state, "hit_max_turn_without_final"):
        return True
    if _state_bool_with_debug_fallback(state, "missing_final"):
        return True
    return state.get("final_answer") is None and state.get("stop_condition") == "max_turns_reached"


def _max_turn_penalty_from_state(
    state: vf.State,
    *,
    correctness: float,
    max_turn_penalty_enabled: bool,
    max_turn_penalty: float,
) -> float:
    if not max_turn_penalty_enabled or correctness <= 0.0:
        return 0.0
    if _state_bool_with_debug_fallback(state, "finalized_on_forced_prompt"):
        return max(0.0, max_turn_penalty)
    return 0.0


def _segment_token_length(segment: dict[str, Any], *, ids_key: str, count_key: str | None = None) -> int:
    if count_key is not None:
        count_value = segment.get(count_key)
        if count_value not in (None, ""):
            return int(count_value)
    token_ids = segment.get(ids_key)
    if isinstance(token_ids, list):
        return len(token_ids)
    return 0


def _segment_rollout_token_totals(state: vf.State) -> tuple[int, int]:
    breakdown = _segment_rollout_token_breakdown(state)
    return breakdown.prompt_tokens, breakdown.completion_tokens


def _segment_rollout_token_breakdown(state: vf.State) -> TokenBreakdown:
    segments = state.get("rlm_segments")
    if isinstance(segments, list):
        prompt_tokens = 0
        completion_tokens = 0
        trainable_tokens = 0
        rlm_turn_tokens = 0
        plain_subcall_tokens = 0
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment_prompt_tokens = _segment_token_length(
                segment,
                ids_key="prompt_ids",
                count_key="prompt_token_count",
            )
            segment_completion_tokens = _segment_token_length(
                segment,
                ids_key="completion_ids",
                count_key="completion_token_count",
            )
            segment_total_tokens = segment_prompt_tokens + segment_completion_tokens
            prompt_tokens += segment_prompt_tokens
            completion_tokens += segment_completion_tokens
            if bool(segment.get("is_trainable_rlm_turn", False)):
                trainable_tokens += segment_total_tokens
            if segment.get("kind") == "plain_query" or segment.get("train_scope") == "llm_subcall":
                plain_subcall_tokens += segment_total_tokens
            else:
                rlm_turn_tokens += segment_total_tokens
        return TokenBreakdown(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            trainable_tokens=trainable_tokens,
            rlm_turn_tokens=rlm_turn_tokens,
            plain_subcall_tokens=plain_subcall_tokens,
        )

    prompt_tokens = int(float(state.get("total_prompt_tokens", 0.0) or 0.0))
    completion_tokens = int(float(state.get("total_completion_tokens", 0.0) or 0.0))
    return TokenBreakdown(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        trainable_tokens=prompt_tokens + completion_tokens,
        rlm_turn_tokens=prompt_tokens + completion_tokens,
        plain_subcall_tokens=0,
    )


def _efficiency_penalty_from_state(state: vf.State) -> tuple[float, int, int, int]:
    breakdown = _segment_rollout_token_breakdown(state)
    penalty_coef = float(state.get("efficiency_penalty_coef", 0.0) or 0.0)
    if penalty_coef <= 0.0 or breakdown.total_tokens <= 0:
        return 0.0, breakdown.prompt_tokens, breakdown.completion_tokens, breakdown.total_tokens
    penalty = penalty_coef * (float(breakdown.total_tokens) / _EFFICIENCY_PENALTY_PER_1K_TOKENS)
    return penalty, breakdown.prompt_tokens, breakdown.completion_tokens, breakdown.total_tokens


def _record_reward_breakdown(
    state: vf.State,
    *,
    correctness: float,
    efficiency_penalty: float,
    total_reward: float,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    trainable_tokens: int | None = None,
    rlm_turn_tokens: int | None = None,
    plain_subcall_tokens: int | None = None,
    weighted_tokens: float | None = None,
    group_solve_rate: float | None = None,
    adaptive_beta: float | None = None,
    adaptive_normalized_cost: float | None = None,
    adaptive_cost_penalty: float | None = None,
    incorrect_cost_penalty: float = 0.0,
    max_turn_penalty: float = 0.0,
) -> None:
    state["reward_correctness"] = correctness
    state["reward_efficiency_penalty"] = efficiency_penalty
    state["reward_incorrect_cost_penalty"] = incorrect_cost_penalty
    state["reward_max_turn_penalty"] = max_turn_penalty
    state["reward_total"] = total_reward
    state["cost_prompt_tokens"] = float(prompt_tokens)
    state["cost_completion_tokens"] = float(completion_tokens)
    state["cost_total_tokens"] = float(total_tokens)
    if trainable_tokens is not None:
        state["cost_trainable_tokens"] = float(trainable_tokens)
    if rlm_turn_tokens is not None:
        state["cost_rlm_turn_tokens"] = float(rlm_turn_tokens)
    if plain_subcall_tokens is not None:
        state["cost_plain_subcall_tokens"] = float(plain_subcall_tokens)
    if weighted_tokens is not None:
        state["cost_weighted_tokens"] = float(weighted_tokens)
    if group_solve_rate is not None:
        state["reward_group_solve_rate"] = group_solve_rate
    if adaptive_beta is not None:
        state["reward_adaptive_beta"] = adaptive_beta
    if adaptive_normalized_cost is not None:
        state["reward_adaptive_normalized_cost"] = adaptive_normalized_cost
    if adaptive_cost_penalty is not None:
        state["reward_adaptive_cost_penalty"] = adaptive_cost_penalty

    trajectory = state.get("trajectory") or []
    if not trajectory:
        return

    debug_payload = {
        "reward_correctness": correctness,
        "reward_efficiency_penalty": efficiency_penalty,
        "reward_incorrect_cost_penalty": incorrect_cost_penalty,
        "reward_max_turn_penalty": max_turn_penalty,
        "reward_total": total_reward,
        "cost_prompt_tokens": prompt_tokens,
        "cost_completion_tokens": completion_tokens,
        "cost_total_tokens": total_tokens,
    }
    if trainable_tokens is not None:
        debug_payload["cost_trainable_tokens"] = trainable_tokens
    if rlm_turn_tokens is not None:
        debug_payload["cost_rlm_turn_tokens"] = rlm_turn_tokens
    if plain_subcall_tokens is not None:
        debug_payload["cost_plain_subcall_tokens"] = plain_subcall_tokens
    if weighted_tokens is not None:
        debug_payload["cost_weighted_tokens"] = weighted_tokens
    if group_solve_rate is not None:
        debug_payload["reward_group_solve_rate"] = group_solve_rate
    if adaptive_beta is not None:
        debug_payload["reward_adaptive_beta"] = adaptive_beta
    if adaptive_normalized_cost is not None:
        debug_payload["reward_adaptive_normalized_cost"] = adaptive_normalized_cost
    if adaptive_cost_penalty is not None:
        debug_payload["reward_adaptive_cost_penalty"] = adaptive_cost_penalty

    extras = trajectory[-1].setdefault("extras", {})
    rlm_debug = extras.setdefault("rlm_debug", {})
    rlm_debug.update(debug_payload)


async def _call_binary_judge(
    judge_client: AsyncOpenAI,
    *,
    judge_model: str,
    judge_prompt: str,
) -> tuple[float, str, str | None]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_prompt},
    ]

    last_raw_response = ""
    for attempt in range(2):
        judge_response = await _call_judge_with_retries(
            lambda: judge_client.chat.completions.create(
                model=judge_model,
                messages=messages,
                temperature=0,
                max_tokens=8,
                extra_body={"reasoning": {"enabled": False}},
            )
        )
        raw_response = _extract_message_text(judge_response.choices[0].message)
        last_raw_response = raw_response
        try:
            return _parse_binary_judge_score(raw_response), raw_response, None
        except ValueError:
            if attempt == 1:
                break
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"{judge_prompt}\n\n"
                        f'Your previous response was invalid: {raw_response!r}\n'
                        "Return only 0 or 1."
                    ),
                },
            ]

    return 0.0, last_raw_response, "invalid_binary_score"


async def _call_vertex_binary_judge(
    judge_client: Any,
    *,
    judge_model: str,
    judge_prompt: str,
    thinking_level: str | None,
) -> tuple[float, str, str | None]:
    _, types = _load_google_genai()
    last_raw_response = ""
    for attempt in range(2):
        user_prompt = judge_prompt
        if attempt == 1:
            user_prompt = (
                f"{judge_prompt}\n\n"
                f"Your previous response was invalid: {last_raw_response!r}\n"
                "Return only 0 or 1."
            )
        config_kwargs: dict[str, Any] = {
            "system_instruction": JUDGE_SYSTEM_PROMPT,
            "temperature": 0,
            "max_output_tokens": _VERTEX_JUDGE_MAX_OUTPUT_TOKENS,
        }
        if thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
        response = await _call_judge_with_retries(
            lambda: _call_vertex_generate_content(
                judge_client,
                model=_normalise_vertex_model_name(judge_model),
                contents=user_prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        )
        raw_response = str(getattr(response, "text", "") or "").strip()
        last_raw_response = raw_response
        try:
            return _parse_binary_judge_score(raw_response), raw_response, None
        except ValueError:
            if attempt == 1:
                break

    return 0.0, last_raw_response, "invalid_binary_score"


async def _score_correctness(
    state: vf.State,
    completion,
    answer: str,
    info: dict[str, Any] | None,
    *,
    judge_provider: str,
    judge_client: Any,
    judge_model: str,
    judge_thinking_level: str | None,
    judge_candidate_max_chars: int,
    judge_question_max_chars: int,
    judge_expected_max_chars: int,
) -> CorrectnessResult:
    predicted_answer = _get_predicted_answer(state, completion).strip()
    expected_answers = _get_expected_answers(answer, info)
    question = str((info or {}).get("question", "")).strip()

    if not predicted_answer:
        result = CorrectnessResult(
            predicted_answer="",
            expected_answers=expected_answers,
            score=0.0,
            raw_response="0",
            parse_error=None,
        )
        _record_judge_payload(
            state,
            predicted_answer=result.predicted_answer,
            expected_answers=result.expected_answers,
            score=result.score,
            raw_response=result.raw_response,
            parse_error=result.parse_error,
        )
        return result

    if len(predicted_answer) > judge_candidate_max_chars:
        result = CorrectnessResult(
            predicted_answer=predicted_answer,
            expected_answers=expected_answers,
            score=0.0,
            raw_response="[candidate_too_large]",
            parse_error="candidate_too_large",
        )
        _record_judge_payload(
            state,
            predicted_answer=result.predicted_answer,
            expected_answers=result.expected_answers,
            score=result.score,
            raw_response=result.raw_response,
            parse_error=result.parse_error,
        )
        return result

    if _is_oolong_pairs_task(info):
        score, raw_response, stats = _score_oolong_pairs(predicted_answer, expected_answers)
        _record_oolong_pairs_metrics(state, stats)
        result = CorrectnessResult(
            predicted_answer=predicted_answer,
            expected_answers=expected_answers,
            score=score,
            raw_response=raw_response,
            parse_error=None,
        )
        _record_judge_payload(
            state,
            predicted_answer=result.predicted_answer,
            expected_answers=result.expected_answers,
            score=result.score,
            raw_response=result.raw_response,
            parse_error=result.parse_error,
        )
        return result

    if _is_oolong_semantic_aggregation_task(info):
        score, raw_response, parse_error = _score_oolong_semantic_aggregation(
            predicted_answer,
            expected_answers,
            answer_type=str((info or {}).get("answer_type") or ""),
        )
        _record_oolong_semantic_metrics(state, score)
        result = CorrectnessResult(
            predicted_answer=predicted_answer,
            expected_answers=expected_answers,
            score=score,
            raw_response=raw_response,
            parse_error=parse_error,
        )
        _record_judge_payload(
            state,
            predicted_answer=result.predicted_answer,
            expected_answers=result.expected_answers,
            score=result.score,
            raw_response=result.raw_response,
            parse_error=result.parse_error,
        )
        return result

    if _is_exact_match(predicted_answer, expected_answers):
        result = CorrectnessResult(
            predicted_answer=predicted_answer,
            expected_answers=expected_answers,
            score=1.0,
            raw_response="[exact_match]",
            parse_error=None,
        )
        _record_judge_payload(
            state,
            predicted_answer=result.predicted_answer,
            expected_answers=result.expected_answers,
            score=result.score,
            raw_response=result.raw_response,
            parse_error=result.parse_error,
        )
        return result

    judge_expected_answers = [
        _truncate_for_judge(str(expected_answer), max_chars=judge_expected_max_chars) for expected_answer in expected_answers
    ]
    judge_prompt = JUDGE_PROMPT.format(
        question=_truncate_for_judge(question or "(not provided)", max_chars=judge_question_max_chars),
        expected_answers=_format_expected_answers(judge_expected_answers),
        predicted_answer=predicted_answer,
    )
    if judge_provider == "vertex":
        try:
            score, raw_response, parse_error = await _call_vertex_binary_judge(
                judge_client,
                judge_model=judge_model,
                judge_prompt=judge_prompt,
                thinking_level=judge_thinking_level,
            )
        except Exception as exc:
            if not _is_oversized_judge_exception(exc):
                raise
            score, raw_response, parse_error = 0.0, f"[vertex_oversized_input: {type(exc).__name__}]", "vertex_input_too_large"
    else:
        score, raw_response, parse_error = await _call_binary_judge(
            judge_client,
            judge_model=judge_model,
            judge_prompt=judge_prompt,
        )

    result = CorrectnessResult(
        predicted_answer=predicted_answer,
        expected_answers=expected_answers,
        score=score,
        raw_response=raw_response,
        parse_error=parse_error,
    )
    _record_judge_payload(
        state,
        predicted_answer=result.predicted_answer,
        expected_answers=result.expected_answers,
        score=result.score,
        raw_response=result.raw_response,
        parse_error=result.parse_error,
    )
    return result


async def _score_correctness_with_protocol(
    state: vf.State,
    completion,
    answer: str,
    info: dict[str, Any] | None,
    *,
    judge_provider: str,
    judge_client: Any,
    judge_model: str,
    judge_thinking_level: str | None,
    judge_candidate_max_chars: int,
    judge_question_max_chars: int,
    judge_expected_max_chars: int,
    missing_final_at_max_turn_zero_reward: bool,
) -> CorrectnessResult:
    if missing_final_at_max_turn_zero_reward and _missing_formal_final_at_max_turn(state):
        predicted_answer = _get_predicted_answer(state, completion).strip()
        result = CorrectnessResult(
            predicted_answer=predicted_answer,
            expected_answers=_get_expected_answers(answer, info),
            score=0.0,
            raw_response="[missing_final_at_max_turn]",
            parse_error=None,
        )
        _record_judge_payload(
            state,
            predicted_answer=result.predicted_answer,
            expected_answers=result.expected_answers,
            score=result.score,
            raw_response=result.raw_response,
            parse_error=result.parse_error,
        )
        return result

    return await _score_correctness(
        state,
        completion,
        answer,
        info,
        judge_provider=judge_provider,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_thinking_level=judge_thinking_level,
        judge_candidate_max_chars=judge_candidate_max_chars,
        judge_question_max_chars=judge_question_max_chars,
        judge_expected_max_chars=judge_expected_max_chars,
    )


def _adaptive_beta_for_solve_rate(*, solve_rate: float, beta_max: float, gamma: float, solve_rate_floor: float) -> float:
    if solve_rate <= solve_rate_floor:
        return 0.0
    if solve_rate_floor >= 1.0:
        return beta_max
    ramp = (solve_rate - solve_rate_floor) / (1.0 - solve_rate_floor)
    return beta_max * (ramp**gamma)


def _weighted_turn_token_cost(
    breakdown: TokenBreakdown,
    *,
    root_token_multiplier: float,
    plain_subcall_token_multiplier: float,
) -> float:
    return (
        float(root_token_multiplier) * float(breakdown.rlm_turn_tokens)
        + float(plain_subcall_token_multiplier) * float(breakdown.plain_subcall_tokens)
    )


def _adaptive_cost_value(
    state: vf.State,
    *,
    cost_basis: str,
    root_token_multiplier: float,
    plain_subcall_token_multiplier: float,
) -> float:
    breakdown = _segment_rollout_token_breakdown(state)
    if cost_basis == "total_tokens":
        return float(breakdown.total_tokens)
    if cost_basis == "weighted_turn_tokens":
        return _weighted_turn_token_cost(
            breakdown,
            root_token_multiplier=root_token_multiplier,
            plain_subcall_token_multiplier=plain_subcall_token_multiplier,
        )
    else:
        raise ValueError(f"adaptive_efficiency_cost_basis must be one of {sorted(_VALID_ADAPTIVE_COST_BASES)}")


def _min_max_normalized(value: float, *, min_value: float, max_value: float) -> float:
    span = max_value - min_value
    if span <= 0:
        return 0.0
    return (float(value) - float(min_value)) / float(span)


def _clip_reward(value: float, *, reward_clip_min: float, reward_clip_max: float) -> float:
    return min(reward_clip_max, max(reward_clip_min, value))


def build_rubric(
    *,
    judge_model: str,
    judge_base_url: str,
    judge_api_key: str | None,
    judge_provider: str = "openai_compatible",
    judge_default_headers: dict[str, str] | None = None,
    judge_vertex_project: str | None = None,
    judge_vertex_location: str = "global",
    judge_thinking_level: str | None = "medium",
    efficiency_penalty_mode: str = "static_per_1k",
    adaptive_efficiency_beta_max: float = 0.05,
    adaptive_efficiency_gamma: float = 2.0,
    adaptive_efficiency_solve_rate_floor: float = 0.25,
    adaptive_efficiency_cost_basis: str = "total_tokens",
    efficiency_root_token_multiplier: float = 1.0,
    efficiency_plain_subcall_token_multiplier: float = 1.0,
    efficiency_penalty_applies_to: str = "correct_only",
    reward_clip_min: float = 0.0,
    reward_clip_max: float = 1.0,
    max_turn_penalty_enabled: bool = False,
    max_turn_penalty: float = 0.25,
    missing_final_at_max_turn_zero_reward: bool = True,
    judge_candidate_max_chars: int = 8192,
    judge_question_max_chars: int = 32768,
    judge_expected_max_chars: int = 8192,
) -> vf.Rubric:
    if efficiency_penalty_mode not in _VALID_EFFICIENCY_PENALTY_MODES:
        raise ValueError(f"efficiency_penalty_mode must be one of {sorted(_VALID_EFFICIENCY_PENALTY_MODES)}")
    if adaptive_efficiency_cost_basis not in _VALID_ADAPTIVE_COST_BASES:
        raise ValueError(f"adaptive_efficiency_cost_basis must be one of {sorted(_VALID_ADAPTIVE_COST_BASES)}")
    if efficiency_penalty_applies_to not in _VALID_EFFICIENCY_PENALTY_SCOPES:
        raise ValueError(f"efficiency_penalty_applies_to must be one of {sorted(_VALID_EFFICIENCY_PENALTY_SCOPES)}")
    if reward_clip_min > reward_clip_max:
        raise ValueError("reward_clip_min must be <= reward_clip_max")
    if adaptive_efficiency_beta_max < 0.0:
        raise ValueError("adaptive_efficiency_beta_max must be >= 0.0")
    if adaptive_efficiency_gamma <= 0.0:
        raise ValueError("adaptive_efficiency_gamma must be > 0.0")
    if not 0.0 <= adaptive_efficiency_solve_rate_floor < 1.0:
        raise ValueError("adaptive_efficiency_solve_rate_floor must be >= 0.0 and < 1.0")
    if efficiency_root_token_multiplier < 0.0:
        raise ValueError("efficiency_root_token_multiplier must be >= 0.0")
    if efficiency_plain_subcall_token_multiplier < 0.0:
        raise ValueError("efficiency_plain_subcall_token_multiplier must be >= 0.0")
    if max_turn_penalty < 0.0:
        raise ValueError("max_turn_penalty must be >= 0.0")
    if judge_candidate_max_chars < 1:
        raise ValueError("judge_candidate_max_chars must be >= 1")
    if judge_question_max_chars < 1:
        raise ValueError("judge_question_max_chars must be >= 1")
    if judge_expected_max_chars < 1:
        raise ValueError("judge_expected_max_chars must be >= 1")

    if judge_provider == "vertex":
        if not judge_vertex_project:
            raise ValueError("judge_vertex_project is required when judge_provider='vertex'")
        _load_google_genai()
        judge_client: Any = _VertexJudgeClientFactory(project=judge_vertex_project, location=judge_vertex_location)
    elif judge_provider == "openai_compatible":
        judge_client = AsyncOpenAI(
            base_url=judge_base_url,
            api_key=judge_api_key or "EMPTY",
            default_headers=judge_default_headers,
        )
    else:
        raise ValueError("judge_provider must be one of ['openai_compatible', 'vertex']")

    async def reward_fn(state: vf.State, completion, answer: str, info: dict[str, Any] | None) -> float:
        correctness_result = await _score_correctness_with_protocol(
            state,
            completion,
            answer,
            info,
            judge_provider=judge_provider,
            judge_client=judge_client,
            judge_model=judge_model,
            judge_thinking_level=judge_thinking_level,
            judge_candidate_max_chars=judge_candidate_max_chars,
            judge_question_max_chars=judge_question_max_chars,
            judge_expected_max_chars=judge_expected_max_chars,
            missing_final_at_max_turn_zero_reward=missing_final_at_max_turn_zero_reward,
        )
        efficiency_penalty, prompt_tokens, completion_tokens, total_tokens = _efficiency_penalty_from_state(state)
        breakdown = _segment_rollout_token_breakdown(state)
        weighted_tokens = _weighted_turn_token_cost(
            breakdown,
            root_token_multiplier=efficiency_root_token_multiplier,
            plain_subcall_token_multiplier=efficiency_plain_subcall_token_multiplier,
        )
        terminal_penalty = _max_turn_penalty_from_state(
            state,
            correctness=correctness_result.score,
            max_turn_penalty_enabled=max_turn_penalty_enabled,
            max_turn_penalty=max_turn_penalty,
        )

        scoped_efficiency_penalty = efficiency_penalty if (
            correctness_result.score > 0.0 or efficiency_penalty_applies_to == "all_rollouts"
        ) else 0.0
        total_reward = _clip_reward(
            correctness_result.score - terminal_penalty - scoped_efficiency_penalty,
            reward_clip_min=reward_clip_min,
            reward_clip_max=reward_clip_max,
        )
        _record_reward_breakdown(
            state,
            correctness=correctness_result.score,
            efficiency_penalty=scoped_efficiency_penalty,
            total_reward=total_reward,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            trainable_tokens=breakdown.trainable_tokens,
            rlm_turn_tokens=breakdown.rlm_turn_tokens,
            plain_subcall_tokens=breakdown.plain_subcall_tokens,
            weighted_tokens=weighted_tokens,
            incorrect_cost_penalty=scoped_efficiency_penalty if correctness_result.score <= 0.0 else 0.0,
            max_turn_penalty=terminal_penalty,
        )
        return total_reward

    async def adaptive_group_reward_fn(states: list[vf.State]) -> list[float]:
        correctness_results = await asyncio.gather(
            *(
                _score_correctness_with_protocol(
                    state,
                    state.get("completion", []),
                    str(state.get("answer", "")),
                    state.get("info", {}),
                    judge_provider=judge_provider,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    judge_thinking_level=judge_thinking_level,
                    judge_candidate_max_chars=judge_candidate_max_chars,
                    judge_question_max_chars=judge_question_max_chars,
                    judge_expected_max_chars=judge_expected_max_chars,
                    missing_final_at_max_turn_zero_reward=missing_final_at_max_turn_zero_reward,
                )
                for state in states
            )
        )
        correctness_scores = [1.0 if result.score > 0.0 else 0.0 for result in correctness_results]
        solve_rate = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0
        beta = _adaptive_beta_for_solve_rate(
            solve_rate=solve_rate,
            beta_max=adaptive_efficiency_beta_max,
            gamma=adaptive_efficiency_gamma,
            solve_rate_floor=adaptive_efficiency_solve_rate_floor,
        )

        costs = [
            _adaptive_cost_value(
                state,
                cost_basis=adaptive_efficiency_cost_basis,
                root_token_multiplier=efficiency_root_token_multiplier,
                plain_subcall_token_multiplier=efficiency_plain_subcall_token_multiplier,
            )
            for state in states
        ]
        correct_costs = [cost for cost, correctness in zip(costs, correctness_scores, strict=True) if correctness > 0.0]
        if efficiency_penalty_applies_to == "all_rollouts":
            normalization_costs = costs
        else:
            normalization_costs = correct_costs
        min_cost = min(normalization_costs) if len(normalization_costs) >= 2 else 0
        max_cost = max(normalization_costs) if len(normalization_costs) >= 2 else 0

        rewards: list[float] = []
        for state, correctness, cost in zip(states, correctness_scores, costs, strict=True):
            breakdown = _segment_rollout_token_breakdown(state)
            weighted_tokens = _weighted_turn_token_cost(
                breakdown,
                root_token_multiplier=efficiency_root_token_multiplier,
                plain_subcall_token_multiplier=efficiency_plain_subcall_token_multiplier,
            )
            normalized_cost = 0.0
            applies_to_rollout = correctness > 0.0 or efficiency_penalty_applies_to == "all_rollouts"
            if applies_to_rollout:
                normalized_cost = _min_max_normalized(cost, min_value=min_cost, max_value=max_cost)
            adaptive_cost_penalty = beta * normalized_cost if applies_to_rollout else 0.0
            terminal_penalty = _max_turn_penalty_from_state(
                state,
                correctness=correctness,
                max_turn_penalty_enabled=max_turn_penalty_enabled,
                max_turn_penalty=max_turn_penalty,
            )
            total_reward = _clip_reward(
                correctness - terminal_penalty - adaptive_cost_penalty,
                reward_clip_min=reward_clip_min,
                reward_clip_max=reward_clip_max,
            )
            _record_reward_breakdown(
                state,
                correctness=correctness,
                efficiency_penalty=adaptive_cost_penalty,
                total_reward=total_reward,
                prompt_tokens=breakdown.prompt_tokens,
                completion_tokens=breakdown.completion_tokens,
                total_tokens=breakdown.total_tokens,
                trainable_tokens=breakdown.trainable_tokens,
                rlm_turn_tokens=breakdown.rlm_turn_tokens,
                plain_subcall_tokens=breakdown.plain_subcall_tokens,
                weighted_tokens=weighted_tokens,
                group_solve_rate=solve_rate,
                adaptive_beta=beta,
                adaptive_normalized_cost=normalized_cost,
                adaptive_cost_penalty=adaptive_cost_penalty,
                incorrect_cost_penalty=adaptive_cost_penalty if correctness <= 0.0 else 0.0,
                max_turn_penalty=terminal_penalty,
            )
            rewards.append(total_reward)
        return rewards

    if efficiency_penalty_mode == "adaptive_group":
        return vf.Rubric(funcs=[adaptive_group_reward_fn])
    return vf.Rubric(funcs=[reward_fn])


async def correctness_metric(state: vf.State) -> float:
    return float(state.get("reward_correctness", 0.0))


async def judge_score_metric(state: vf.State) -> float:
    return float(state.get("judge_score", 0.0))


async def oolong_pairs_precision_metric(state: vf.State) -> float:
    return float(state.get("oolong_pairs_precision", 0.0))


async def oolong_pairs_recall_metric(state: vf.State) -> float:
    return float(state.get("oolong_pairs_recall", 0.0))


async def oolong_pairs_f1_metric(state: vf.State) -> float:
    return float(state.get("oolong_pairs_f1", 0.0))


async def oolong_semantic_score_metric(state: vf.State) -> float:
    return float(state.get("oolong_semantic_score", 0.0))


async def oolong_semantic_exact_metric(state: vf.State) -> float:
    return float(state.get("oolong_semantic_exact", 0.0))


async def efficiency_penalty_metric(state: vf.State) -> float:
    return float(state.get("reward_efficiency_penalty", 0.0))


async def incorrect_cost_penalty_metric(state: vf.State) -> float:
    return float(state.get("reward_incorrect_cost_penalty", 0.0))


async def max_turn_penalty_metric(state: vf.State) -> float:
    return float(state.get("reward_max_turn_penalty", 0.0))


async def cost_prompt_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_prompt_tokens", 0.0))


async def cost_completion_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_completion_tokens", 0.0))


async def cost_total_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_total_tokens", 0.0))


async def cost_trainable_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_trainable_tokens", 0.0))


async def cost_rlm_turn_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_rlm_turn_tokens", 0.0))


async def cost_plain_subcall_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_plain_subcall_tokens", 0.0))


async def cost_weighted_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_weighted_tokens", 0.0))


async def adaptive_group_solve_rate_metric(state: vf.State) -> float:
    return float(state.get("reward_group_solve_rate", 0.0))


async def adaptive_beta_metric(state: vf.State) -> float:
    return float(state.get("reward_adaptive_beta", 0.0))


async def adaptive_normalized_cost_metric(state: vf.State) -> float:
    return float(state.get("reward_adaptive_normalized_cost", 0.0))


async def adaptive_cost_penalty_metric(state: vf.State) -> float:
    return float(state.get("reward_adaptive_cost_penalty", 0.0))


async def used_repl_metric(state: vf.State) -> float:
    return 1.0 if state.get("used_repl") else 0.0


async def used_recursion_metric(state: vf.State) -> float:
    return 1.0 if state.get("used_recursion") else 0.0


async def used_llm_subcalls_metric(state: vf.State) -> float:
    return 1.0 if state.get("used_llm_subcalls") else 0.0


async def used_rlm_subcalls_metric(state: vf.State) -> float:
    return 1.0 if state.get("used_rlm_subcalls") else 0.0


async def num_subcalls_metric(state: vf.State) -> float:
    return float(state.get("num_subcalls", 0))


async def num_llm_subcalls_metric(state: vf.State) -> float:
    return float(state.get("num_llm_subcalls", 0))


async def num_rlm_subcalls_metric(state: vf.State) -> float:
    return float(state.get("num_rlm_subcalls", 0))


async def max_depth_metric(state: vf.State) -> float:
    return float(state.get("max_depth_reached", 0))


async def used_forced_finalize_prompt_metric(state: vf.State) -> float:
    return 1.0 if state.get("used_forced_finalize_prompt") else 0.0


async def hit_max_turn_without_final_metric(state: vf.State) -> float:
    return 1.0 if state.get("hit_max_turn_without_final") else 0.0


async def missing_final_metric(state: vf.State) -> float:
    return 1.0 if state.get("missing_final") else 0.0


async def finalized_before_forced_prompt_metric(state: vf.State) -> float:
    return 1.0 if state.get("finalized_before_forced_prompt") else 0.0


async def finalized_on_forced_prompt_metric(state: vf.State) -> float:
    return 1.0 if state.get("finalized_on_forced_prompt") else 0.0


def add_metrics(rubric: vf.Rubric) -> vf.Rubric:
    rubric.add_metric(correctness_metric)
    rubric.add_metric(judge_score_metric)
    rubric.add_metric(oolong_pairs_precision_metric)
    rubric.add_metric(oolong_pairs_recall_metric)
    rubric.add_metric(oolong_pairs_f1_metric)
    rubric.add_metric(oolong_semantic_score_metric)
    rubric.add_metric(oolong_semantic_exact_metric)
    rubric.add_metric(efficiency_penalty_metric)
    rubric.add_metric(incorrect_cost_penalty_metric)
    rubric.add_metric(max_turn_penalty_metric)
    rubric.add_metric(cost_prompt_tokens_metric)
    rubric.add_metric(cost_completion_tokens_metric)
    rubric.add_metric(cost_total_tokens_metric)
    rubric.add_metric(cost_trainable_tokens_metric)
    rubric.add_metric(cost_rlm_turn_tokens_metric)
    rubric.add_metric(cost_plain_subcall_tokens_metric)
    rubric.add_metric(cost_weighted_tokens_metric)
    rubric.add_metric(adaptive_group_solve_rate_metric)
    rubric.add_metric(adaptive_beta_metric)
    rubric.add_metric(adaptive_normalized_cost_metric)
    rubric.add_metric(adaptive_cost_penalty_metric)
    rubric.add_metric(used_repl_metric)
    rubric.add_metric(used_recursion_metric)
    rubric.add_metric(used_llm_subcalls_metric)
    rubric.add_metric(used_rlm_subcalls_metric)
    rubric.add_metric(num_subcalls_metric)
    rubric.add_metric(num_llm_subcalls_metric)
    rubric.add_metric(num_rlm_subcalls_metric)
    rubric.add_metric(max_depth_metric)
    rubric.add_metric(used_forced_finalize_prompt_metric)
    rubric.add_metric(hit_max_turn_without_final_metric)
    rubric.add_metric(missing_final_metric)
    rubric.add_metric(finalized_before_forced_prompt_metric)
    rubric.add_metric(finalized_on_forced_prompt_metric)
    return rubric
