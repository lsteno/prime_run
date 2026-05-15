from __future__ import annotations

import asyncio
import json
import random
import re
import unicodedata
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


def _get_predicted_answer(state: vf.State, completion) -> str:
    final_answer = state.get("final_answer")
    if final_answer is not None:
        return str(final_answer)
    if completion:
        return str(completion[-1].get("content", ""))
    return ""


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
    segments = state.get("rlm_segments")
    if isinstance(segments, list):
        prompt_tokens = 0
        completion_tokens = 0
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            prompt_tokens += _segment_token_length(segment, ids_key="prompt_ids", count_key="prompt_token_count")
            completion_tokens += _segment_token_length(
                segment,
                ids_key="completion_ids",
                count_key="completion_token_count",
            )
        return prompt_tokens, completion_tokens

    prompt_tokens = int(float(state.get("total_prompt_tokens", 0.0) or 0.0))
    completion_tokens = int(float(state.get("total_completion_tokens", 0.0) or 0.0))
    return prompt_tokens, completion_tokens


def _efficiency_penalty_from_state(state: vf.State) -> tuple[float, int, int, int]:
    prompt_tokens, completion_tokens = _segment_rollout_token_totals(state)
    total_tokens = prompt_tokens + completion_tokens
    penalty_coef = float(state.get("efficiency_penalty_coef", 0.0) or 0.0)
    if penalty_coef <= 0.0 or total_tokens <= 0:
        return 0.0, prompt_tokens, completion_tokens, total_tokens
    penalty = penalty_coef * (float(total_tokens) / _EFFICIENCY_PENALTY_PER_1K_TOKENS)
    return penalty, prompt_tokens, completion_tokens, total_tokens


def _record_reward_breakdown(
    state: vf.State,
    *,
    correctness: float,
    efficiency_penalty: float,
    total_reward: float,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> None:
    state["reward_correctness"] = correctness
    state["reward_efficiency_penalty"] = efficiency_penalty
    state["reward_total"] = total_reward
    state["cost_prompt_tokens"] = float(prompt_tokens)
    state["cost_completion_tokens"] = float(completion_tokens)
    state["cost_total_tokens"] = float(total_tokens)

    trajectory = state.get("trajectory") or []
    if not trajectory:
        return

    extras = trajectory[-1].setdefault("extras", {})
    rlm_debug = extras.setdefault("rlm_debug", {})
    rlm_debug.update(
        {
            "reward_correctness": correctness,
            "reward_efficiency_penalty": efficiency_penalty,
            "reward_total": total_reward,
            "cost_prompt_tokens": prompt_tokens,
            "cost_completion_tokens": completion_tokens,
            "cost_total_tokens": total_tokens,
        }
    )


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
) -> vf.Rubric:
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
        predicted_answer = _get_predicted_answer(state, completion).strip()
        expected_answers = _get_expected_answers(answer, info)
        question = str((info or {}).get("question", "")).strip()
        efficiency_penalty, prompt_tokens, completion_tokens, total_tokens = _efficiency_penalty_from_state(state)

        if not predicted_answer:
            total_reward = 0.0
            _record_reward_breakdown(
                state,
                correctness=0.0,
                efficiency_penalty=efficiency_penalty,
                total_reward=total_reward,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            _record_judge_payload(
                state,
                predicted_answer="",
                expected_answers=expected_answers,
                score=0.0,
                raw_response="0",
                parse_error=None,
            )
            return total_reward

        if _is_exact_match(predicted_answer, expected_answers):
            total_reward = max(0.0, 1.0 - efficiency_penalty)
            _record_reward_breakdown(
                state,
                correctness=1.0,
                efficiency_penalty=efficiency_penalty,
                total_reward=total_reward,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            _record_judge_payload(
                state,
                predicted_answer=predicted_answer,
                expected_answers=expected_answers,
                score=1.0,
                raw_response="[exact_match]",
                parse_error=None,
            )
            return total_reward

        judge_prompt = JUDGE_PROMPT.format(
            question=question or "(not provided)",
            expected_answers=_format_expected_answers(expected_answers),
            predicted_answer=predicted_answer,
        )
        if judge_provider == "vertex":
            score, raw_response, parse_error = await _call_vertex_binary_judge(
                judge_client,
                judge_model=judge_model,
                judge_prompt=judge_prompt,
                thinking_level=judge_thinking_level,
            )
        else:
            score, raw_response, parse_error = await _call_binary_judge(
                judge_client,
                judge_model=judge_model,
                judge_prompt=judge_prompt,
            )

        total_reward = max(0.0, score - efficiency_penalty) if score > 0.0 else 0.0
        _record_reward_breakdown(
            state,
            correctness=score,
            efficiency_penalty=efficiency_penalty,
            total_reward=total_reward,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        _record_judge_payload(
            state,
            predicted_answer=predicted_answer,
            expected_answers=expected_answers,
            score=score,
            raw_response=raw_response,
            parse_error=parse_error,
        )
        return total_reward

    return vf.Rubric(funcs=[reward_fn])


async def correctness_metric(state: vf.State) -> float:
    return float(state.get("reward_correctness", 0.0))


async def judge_score_metric(state: vf.State) -> float:
    return float(state.get("judge_score", 0.0))


async def efficiency_penalty_metric(state: vf.State) -> float:
    return float(state.get("reward_efficiency_penalty", 0.0))


async def cost_prompt_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_prompt_tokens", 0.0))


async def cost_completion_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_completion_tokens", 0.0))


async def cost_total_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_total_tokens", 0.0))


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


def add_metrics(rubric: vf.Rubric) -> vf.Rubric:
    rubric.add_metric(correctness_metric)
    rubric.add_metric(judge_score_metric)
    rubric.add_metric(efficiency_penalty_metric)
    rubric.add_metric(cost_prompt_tokens_metric)
    rubric.add_metric(cost_completion_tokens_metric)
    rubric.add_metric(cost_total_tokens_metric)
    rubric.add_metric(used_repl_metric)
    rubric.add_metric(used_recursion_metric)
    rubric.add_metric(used_llm_subcalls_metric)
    rubric.add_metric(used_rlm_subcalls_metric)
    rubric.add_metric(num_subcalls_metric)
    rubric.add_metric(num_llm_subcalls_metric)
    rubric.add_metric(num_rlm_subcalls_metric)
    rubric.add_metric(max_depth_metric)
    return rubric
