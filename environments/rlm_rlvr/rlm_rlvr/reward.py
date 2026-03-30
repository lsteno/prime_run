from __future__ import annotations

import json
import re
from typing import Any

from openai import AsyncOpenAI
import verifiers as vf

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


async def _call_binary_judge(
    judge_client: AsyncOpenAI,
    *,
    judge_model: str,
    judge_prompt: str,
) -> tuple[float, str]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_prompt},
    ]

    last_raw_response = ""
    for attempt in range(2):
        judge_response = await judge_client.chat.completions.create(
            model=judge_model,
            messages=messages,
            temperature=0,
            max_tokens=4,
        )
        raw_response = (judge_response.choices[0].message.content or "").strip()
        last_raw_response = raw_response
        try:
            return _parse_binary_judge_score(raw_response), raw_response
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

    return 0.0, last_raw_response


def build_rubric(
    *,
    judge_model: str,
    judge_base_url: str,
    judge_api_key: str,
    judge_default_headers: dict[str, str] | None = None,
) -> vf.Rubric:
    judge_client = AsyncOpenAI(
        base_url=judge_base_url,
        api_key=judge_api_key,
        default_headers=judge_default_headers,
    )

    async def reward_fn(state: vf.State, completion, answer: str, info: dict[str, Any] | None) -> float:
        predicted_answer = _get_predicted_answer(state, completion).strip()
        expected_answers = _get_expected_answers(answer, info)
        question = str((info or {}).get("question", "")).strip()

        if not predicted_answer:
            state["reward_correctness"] = 0.0
            _record_judge_payload(
                state,
                predicted_answer="",
                expected_answers=expected_answers,
                score=0.0,
                raw_response="0",
                parse_error=None,
            )
            return 0.0

        judge_prompt = JUDGE_PROMPT.format(
            question=question or "(not provided)",
            expected_answers=_format_expected_answers(expected_answers),
            predicted_answer=predicted_answer,
        )
        score, raw_response = await _call_binary_judge(
            judge_client,
            judge_model=judge_model,
            judge_prompt=judge_prompt,
        )

        state["reward_correctness"] = score
        parse_error = None if raw_response.strip() in {"0", "1"} else ("invalid_binary_score" if score == 0.0 else None)
        _record_judge_payload(
            state,
            predicted_answer=predicted_answer,
            expected_answers=expected_answers,
            score=score,
            raw_response=raw_response,
            parse_error=parse_error,
        )
        return score

    return vf.Rubric(funcs=[reward_fn])


async def correctness_metric(state: vf.State) -> float:
    return float(state.get("reward_correctness", 0.0))


async def judge_score_metric(state: vf.State) -> float:
    return float(state.get("judge_score", 0.0))


async def used_repl_metric(state: vf.State) -> float:
    return 1.0 if state.get("used_repl") else 0.0


async def used_recursion_metric(state: vf.State) -> float:
    return 1.0 if state.get("used_recursion") else 0.0


async def num_subcalls_metric(state: vf.State) -> float:
    return float(state.get("num_subcalls", 0))


async def max_depth_metric(state: vf.State) -> float:
    return float(state.get("max_depth_reached", 0))


def add_metrics(rubric: vf.Rubric) -> vf.Rubric:
    rubric.add_metric(correctness_metric)
    rubric.add_metric(judge_score_metric)
    rubric.add_metric(used_repl_metric)
    rubric.add_metric(used_recursion_metric)
    rubric.add_metric(num_subcalls_metric)
    rubric.add_metric(max_depth_metric)
    return rubric
