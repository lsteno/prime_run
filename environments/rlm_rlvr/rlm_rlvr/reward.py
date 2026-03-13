from __future__ import annotations

from typing import Any

import verifiers as vf

from .parsing import normalize_text


ANSWER_PRESENCE_BONUS = 0.01


def _get_expected_answers(answer: str, info: dict[str, Any] | None) -> list[str]:
    if info and info.get("acceptable_answers"):
        return [str(item) for item in info["acceptable_answers"]]
    return [answer]


def _get_predicted_answer(state: vf.State, completion) -> str:
    final_answer = state.get("final_answer")
    if final_answer is not None:
        return str(final_answer)
    if completion:
        return str(completion[-1].get("content", ""))
    return ""


async def reward_fn(state: vf.State, completion, answer: str, info: dict[str, Any] | None) -> float:
    predicted = normalize_text(_get_predicted_answer(state, completion))
    expected = {normalize_text(item) for item in _get_expected_answers(answer, info)}
    correctness = 1.0 if predicted in expected else 0.0
    answer_presence = ANSWER_PRESENCE_BONUS if predicted else 0.0
    penalty = float(state.get("num_subcalls", 0)) * float(state.get("efficiency_penalty_coef", 0.02))

    state["reward_correctness"] = correctness
    state["reward_answer_presence"] = answer_presence
    state["reward_efficiency_penalty"] = penalty
    return max(-1.0, min(1.0, correctness + answer_presence - penalty))


async def correctness_metric(state: vf.State) -> float:
    return float(state.get("reward_correctness", 0.0))


async def answer_presence_metric(state: vf.State) -> float:
    return float(state.get("reward_answer_presence", 0.0))


async def efficiency_penalty_metric(state: vf.State) -> float:
    return float(state.get("reward_efficiency_penalty", 0.0))


async def used_repl_metric(state: vf.State) -> float:
    return 1.0 if state.get("used_repl") else 0.0


async def used_recursion_metric(state: vf.State) -> float:
    return 1.0 if state.get("used_recursion") else 0.0


async def num_subcalls_metric(state: vf.State) -> float:
    return float(state.get("num_subcalls", 0))


async def max_depth_metric(state: vf.State) -> float:
    return float(state.get("max_depth_reached", 0))


def build_rubric() -> vf.Rubric:
    rubric = vf.Rubric(funcs=[reward_fn])
    rubric.add_metric(correctness_metric)
    rubric.add_metric(answer_presence_metric)
    rubric.add_metric(efficiency_penalty_metric)
    rubric.add_metric(used_repl_metric)
    rubric.add_metric(used_recursion_metric)
    rubric.add_metric(num_subcalls_metric)
    rubric.add_metric(max_depth_metric)
    return rubric