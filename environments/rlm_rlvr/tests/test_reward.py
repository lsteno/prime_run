from __future__ import annotations

import asyncio
from types import SimpleNamespace

import rlm_rlvr.reward as reward_module
from rlm_rlvr.reward import (
    _adaptive_beta_for_solve_rate,
    _efficiency_penalty_from_state,
    _extract_message_text,
    _is_exact_match,
    _parse_binary_judge_score,
    _segment_rollout_token_breakdown,
    _segment_rollout_token_totals,
    build_rubric,
)


def test_is_exact_match_handles_numeric_normalization() -> None:
    assert _is_exact_match("1,208", ["1208"])
    assert _is_exact_match('"1208"', ["1208"])


def test_is_exact_match_handles_json_normalization() -> None:
    assert _is_exact_match('{"b": 2, "a": 1}', ['{"a":1,"b":2}'])


def test_extract_message_text_falls_back_to_reasoning() -> None:
    message = SimpleNamespace(content=None, reasoning="1")
    assert _extract_message_text(message) == "1"


def test_parse_binary_judge_score_accepts_embedded_binary() -> None:
    assert _parse_binary_judge_score("score: 1") == 1.0


def test_segment_rollout_token_totals_include_non_trainable_subcalls() -> None:
    prompt_tokens, completion_tokens = _segment_rollout_token_totals(
        {
            "rlm_segments": [
                {
                    "kind": "root_turn",
                    "is_trainable_rlm_turn": True,
                    "prompt_ids": [1, 2, 3],
                    "completion_ids": [4, 5],
                },
                {
                    "kind": "plain_query",
                    "is_trainable_rlm_turn": False,
                    "prompt_ids": [10, 11],
                    "completion_ids": [12, 13, 14, 15],
                },
            ]
        }
    )

    assert prompt_tokens == 5
    assert completion_tokens == 6


def test_efficiency_penalty_uses_exact_segment_token_totals() -> None:
    penalty, prompt_tokens, completion_tokens, total_tokens = _efficiency_penalty_from_state(
        {
            "efficiency_penalty_coef": 0.02,
            "rlm_segments": [
                {
                    "prompt_ids": list(range(600)),
                    "completion_ids": list(range(400)),
                }
            ],
        }
    )

    assert prompt_tokens == 600
    assert completion_tokens == 400
    assert total_tokens == 1000
    assert penalty == 0.02


def test_token_breakdown_separates_trainable_and_plain_subcall_tokens() -> None:
    breakdown = _segment_rollout_token_breakdown(
        {
            "rlm_segments": [
                {
                    "kind": "root_turn",
                    "is_trainable_rlm_turn": True,
                    "prompt_ids": list(range(30)),
                    "completion_ids": list(range(20)),
                },
                {
                    "kind": "plain_query",
                    "is_trainable_rlm_turn": False,
                    "prompt_token_count": 100,
                    "completion_token_count": 50,
                    "prompt_ids": [],
                    "completion_ids": [],
                },
            ],
        }
    )

    assert breakdown.total_tokens == 200
    assert breakdown.trainable_tokens == 50
    assert breakdown.plain_subcall_tokens == 150


def test_adaptive_beta_ramps_after_solve_rate_floor() -> None:
    assert _adaptive_beta_for_solve_rate(solve_rate=0.25, beta_max=0.05, gamma=2.0, solve_rate_floor=0.25) == 0.0
    assert _adaptive_beta_for_solve_rate(solve_rate=0.5, beta_max=0.05, gamma=2.0, solve_rate_floor=0.25) == (
        0.05 * ((0.5 - 0.25) / 0.75) ** 2
    )
    assert _adaptive_beta_for_solve_rate(solve_rate=1.0, beta_max=0.05, gamma=2.0, solve_rate_floor=0.25) == 0.05


def test_build_rubric_applies_cost_penalty_on_exact_match(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
    )
    reward_fn = rubric.funcs[0]
    state = {
        "final_answer": "42",
        "efficiency_penalty_coef": 0.02,
        "rlm_segments": [
            {
                "kind": "root_turn",
                "is_trainable_rlm_turn": True,
                "prompt_ids": list(range(300)),
                "completion_ids": list(range(200)),
            },
            {
                "kind": "plain_query",
                "is_trainable_rlm_turn": False,
                "prompt_ids": list(range(100)),
                "completion_ids": list(range(50)),
            },
        ],
        "trajectory": [],
    }

    score = asyncio.run(reward_fn(state, [], "42", {"question": "What is the answer?"}))

    assert score == 1.0 - 0.02 * 0.65
    assert state["reward_correctness"] == 1.0
    assert state["reward_efficiency_penalty"] == 0.02 * 0.65
    assert state["cost_prompt_tokens"] == 400.0
    assert state["cost_completion_tokens"] == 250.0
    assert state["cost_total_tokens"] == 650.0


def test_build_rubric_clips_exact_match_reward_at_zero(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
    )
    reward_fn = rubric.funcs[0]
    state = {
        "final_answer": "42",
        "efficiency_penalty_coef": 2.0,
        "rlm_segments": [
            {
                "prompt_ids": list(range(600)),
                "completion_ids": list(range(400)),
            }
        ],
        "trajectory": [],
    }

    score = asyncio.run(reward_fn(state, [], "42", {"question": "What is the answer?"}))

    assert score == 0.0
    assert state["reward_correctness"] == 1.0
    assert state["reward_efficiency_penalty"] == 2.0
    assert state["reward_total"] == 0.0


def test_build_rubric_no_answer_reward_is_zero_with_cost_penalty(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
    )
    reward_fn = rubric.funcs[0]
    state = {
        "final_answer": "",
        "efficiency_penalty_coef": 0.02,
        "rlm_segments": [
            {
                "prompt_ids": list(range(600)),
                "completion_ids": list(range(400)),
            }
        ],
        "trajectory": [],
    }

    score = asyncio.run(reward_fn(state, [], "42", {"question": "What is the answer?"}))

    assert score == 0.0
    assert state["reward_correctness"] == 0.0
    assert state["reward_efficiency_penalty"] == 0.0
    assert state["reward_total"] == 0.0


def test_build_rubric_incorrect_judge_reward_is_zero_with_cost_penalty(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    async def _judge_zero(*args, **kwargs):
        del args, kwargs
        return 0.0, "0", None

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    monkeypatch.setattr(reward_module, "_call_binary_judge", _judge_zero)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
    )
    reward_fn = rubric.funcs[0]
    state = {
        "final_answer": "wrong",
        "efficiency_penalty_coef": 0.02,
        "rlm_segments": [
            {
                "prompt_ids": list(range(600)),
                "completion_ids": list(range(400)),
            }
        ],
        "trajectory": [],
    }

    score = asyncio.run(reward_fn(state, [], "42", {"question": "What is the answer?"}))

    assert score == 0.0
    assert state["reward_correctness"] == 0.0
    assert state["reward_efficiency_penalty"] == 0.0
    assert state["reward_total"] == 0.0


def test_static_cost_penalty_can_make_incorrect_rollout_negative_when_enabled(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    async def _judge_zero(*args, **kwargs):
        del args, kwargs
        return 0.0, "0", None

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    monkeypatch.setattr(reward_module, "_call_binary_judge", _judge_zero)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
        efficiency_penalty_applies_to="all_rollouts",
        reward_clip_min=-0.5,
    )
    reward_fn = rubric.funcs[0]
    state = {
        "final_answer": "wrong",
        "efficiency_penalty_coef": 2.0,
        "rlm_segments": [
            {
                "prompt_ids": list(range(600)),
                "completion_ids": list(range(400)),
            }
        ],
        "trajectory": [],
    }

    score = asyncio.run(reward_fn(state, [], "42", {"question": "What is the answer?"}))

    assert score == -0.5
    assert state["reward_correctness"] == 0.0
    assert state["reward_efficiency_penalty"] == 2.0
    assert state["reward_incorrect_cost_penalty"] == 2.0
    assert state["reward_total"] == -0.5


def test_build_rubric_zeroes_missing_formal_final_at_max_turn(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
        max_turn_penalty_enabled=True,
        missing_final_at_max_turn_zero_reward=True,
    )
    reward_fn = rubric.funcs[0]
    state = {
        "final_answer": None,
        "hit_max_turn_without_final": True,
        "efficiency_penalty_coef": 0.0,
        "rlm_segments": [],
        "trajectory": [],
    }

    score = asyncio.run(reward_fn(state, [{"content": "42"}], "42", {"question": "What is the answer?"}))

    assert score == 0.0
    assert state["reward_correctness"] == 0.0
    assert state["judge_raw_response"] == "[missing_final_at_max_turn]"
    assert state["reward_max_turn_penalty"] == 0.0


def test_build_rubric_zeroes_missing_final_from_stop_condition_without_flags(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
        max_turn_penalty_enabled=True,
        missing_final_at_max_turn_zero_reward=True,
    )
    reward_fn = rubric.funcs[0]
    state = {
        "final_answer": None,
        "stop_condition": "max_turns_reached",
        "efficiency_penalty_coef": 0.0,
        "rlm_segments": [],
        "trajectory": [],
    }

    score = asyncio.run(reward_fn(state, [{"content": "42"}], "42", {"question": "What is the answer?"}))

    assert score == 0.0
    assert state["reward_correctness"] == 0.0
    assert state["judge_predicted_answer"] == "42"
    assert state["judge_raw_response"] == "[missing_final_at_max_turn]"


def test_build_rubric_zeroes_missing_final_from_trajectory_debug(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
        max_turn_penalty_enabled=True,
        missing_final_at_max_turn_zero_reward=True,
    )
    reward_fn = rubric.funcs[0]
    state = {
        "final_answer": None,
        "efficiency_penalty_coef": 0.0,
        "rlm_segments": [],
        "trajectory": [
            {
                "extras": {
                    "rlm_debug": {
                        "missing_final": True,
                    }
                }
            }
        ],
    }

    score = asyncio.run(reward_fn(state, [{"content": "42"}], "42", {"question": "What is the answer?"}))

    assert score == 0.0
    assert state["reward_correctness"] == 0.0
    assert state["judge_raw_response"] == "[missing_final_at_max_turn]"


def test_build_rubric_penalizes_correct_forced_finalize_turn(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
        max_turn_penalty_enabled=True,
        max_turn_penalty=0.25,
    )
    reward_fn = rubric.funcs[0]
    state = {
        "final_answer": "42",
        "finalized_on_forced_prompt": True,
        "efficiency_penalty_coef": 0.0,
        "rlm_segments": [],
        "trajectory": [],
    }

    score = asyncio.run(reward_fn(state, [], "42", {"question": "What is the answer?"}))

    assert score == 0.75
    assert state["reward_correctness"] == 1.0
    assert state["reward_max_turn_penalty"] == 0.25


def _adaptive_test_state(final_answer: str, total_tokens: int, *, answer: str = "42") -> dict:
    prompt_tokens = max(0, total_tokens - 10)
    completion_tokens = min(10, total_tokens)
    return {
        "final_answer": final_answer,
        "answer": answer,
        "completion": [],
        "info": {"question": "What is the answer?"},
        "prompt": [],
        "task": "rlm_rlvr",
        "rlm_segments": [
            {
                "kind": "root_turn",
                "is_trainable_rlm_turn": True,
                "prompt_ids": list(range(prompt_tokens)),
                "completion_ids": list(range(completion_tokens)),
            }
        ],
        "trajectory": [],
    }


def _adaptive_rubric(monkeypatch):
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    async def _judge_zero(*args, **kwargs):
        del args, kwargs
        return 0.0, "0", None

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    monkeypatch.setattr(reward_module, "_call_binary_judge", _judge_zero)
    return build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
        efficiency_penalty_mode="adaptive_group",
    )


def test_adaptive_group_rubric_uses_group_reward_function(monkeypatch) -> None:
    rubric = _adaptive_rubric(monkeypatch)

    assert len(rubric.funcs) == 1
    assert rubric._is_group_func(rubric.funcs[0])


def test_adaptive_group_all_wrong_returns_zero_and_beta_zero(monkeypatch) -> None:
    rubric = _adaptive_rubric(monkeypatch)
    reward_fn = rubric.funcs[0]
    states = [_adaptive_test_state("wrong", total_tokens) for total_tokens in (20, 40, 80, 100)]

    scores = asyncio.run(reward_fn(states))

    assert scores == [0.0, 0.0, 0.0, 0.0]
    assert all(state["reward_correctness"] == 0.0 for state in states)
    assert all(state["reward_adaptive_beta"] == 0.0 for state in states)
    assert all(state["reward_efficiency_penalty"] == 0.0 for state in states)


def test_adaptive_group_single_correct_hard_group_has_no_cost_penalty(monkeypatch) -> None:
    rubric = _adaptive_rubric(monkeypatch)
    reward_fn = rubric.funcs[0]
    states = [
        _adaptive_test_state("42", 100),
        _adaptive_test_state("wrong", 20),
        _adaptive_test_state("wrong", 40),
        _adaptive_test_state("wrong", 80),
    ]

    scores = asyncio.run(reward_fn(states))

    assert scores == [1.0, 0.0, 0.0, 0.0]
    assert states[0]["reward_group_solve_rate"] == 0.25
    assert states[0]["reward_adaptive_beta"] == 0.0
    assert states[0]["reward_adaptive_normalized_cost"] == 0.0


def test_adaptive_group_penalizes_only_correct_rollouts_by_relative_cost_by_default(monkeypatch) -> None:
    rubric = _adaptive_rubric(monkeypatch)
    reward_fn = rubric.funcs[0]
    states = [
        _adaptive_test_state("42", 20),
        _adaptive_test_state("42", 40),
        _adaptive_test_state("42", 80),
        _adaptive_test_state("wrong", 10),
    ]

    scores = asyncio.run(reward_fn(states))

    expected_beta = 0.05 * ((0.75 - 0.25) / 0.75) ** 2
    assert scores[0] == 1.0
    assert scores[1] == 1.0 - expected_beta * ((40 - 20) / (80 - 20))
    assert scores[2] == 1.0 - expected_beta
    assert scores[3] == 0.0
    assert all(score > scores[3] for score in scores[:3])
    assert states[3]["reward_adaptive_normalized_cost"] == 0.0
    assert states[3]["reward_efficiency_penalty"] == 0.0


def test_adaptive_group_can_penalize_incorrect_rollouts_by_relative_group_cost(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    async def _judge_zero(*args, **kwargs):
        del args, kwargs
        return 0.0, "0", None

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    monkeypatch.setattr(reward_module, "_call_binary_judge", _judge_zero)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
        efficiency_penalty_mode="adaptive_group",
        adaptive_efficiency_beta_max=0.15,
        adaptive_efficiency_gamma=1.0,
        efficiency_penalty_applies_to="all_rollouts",
        reward_clip_min=-0.5,
        reward_clip_max=1.0,
    )
    reward_fn = rubric.funcs[0]
    states = [
        _adaptive_test_state("42", 20),
        _adaptive_test_state("wrong", 40),
        _adaptive_test_state("wrong", 80),
        _adaptive_test_state("42", 100),
    ]

    scores = asyncio.run(reward_fn(states))

    expected_beta = 0.15 * ((0.5 - 0.25) / 0.75)
    assert scores[0] == 1.0
    assert scores[1] == -(expected_beta * ((40 - 20) / (100 - 20)))
    assert scores[2] == -(expected_beta * ((80 - 20) / (100 - 20)))
    assert scores[3] == 1.0 - expected_beta
    assert states[1]["reward_correctness"] == 0.0
    assert states[1]["reward_efficiency_penalty"] > 0.0
    assert states[1]["reward_incorrect_cost_penalty"] == states[1]["reward_efficiency_penalty"]


def test_adaptive_group_all_wrong_keeps_beta_zero_even_with_all_rollouts_penalty(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    async def _judge_zero(*args, **kwargs):
        del args, kwargs
        return 0.0, "0", None

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    monkeypatch.setattr(reward_module, "_call_binary_judge", _judge_zero)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
        efficiency_penalty_mode="adaptive_group",
        adaptive_efficiency_beta_max=0.15,
        adaptive_efficiency_gamma=1.0,
        efficiency_penalty_applies_to="all_rollouts",
        reward_clip_min=-0.5,
        reward_clip_max=1.0,
    )
    reward_fn = rubric.funcs[0]
    states = [_adaptive_test_state("wrong", total_tokens) for total_tokens in (20, 40, 80, 100)]

    scores = asyncio.run(reward_fn(states))

    assert scores == [0.0, 0.0, 0.0, 0.0]
    assert all(state["reward_adaptive_beta"] == 0.0 for state in states)
    assert all(state["reward_incorrect_cost_penalty"] == 0.0 for state in states)


def test_adaptive_group_all_correct_compresses_cost(monkeypatch) -> None:
    rubric = _adaptive_rubric(monkeypatch)
    reward_fn = rubric.funcs[0]
    states = [
        _adaptive_test_state("42", 20),
        _adaptive_test_state("42", 40),
        _adaptive_test_state("42", 80),
        _adaptive_test_state("42", 100),
    ]

    scores = asyncio.run(reward_fn(states))

    assert states[0]["reward_group_solve_rate"] == 1.0
    assert states[0]["reward_adaptive_beta"] == 0.05
    assert scores == [1.0, 0.9875, 0.9625, 0.95]


def test_adaptive_group_penalizes_correct_forced_finalize_turn(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    async def _judge_zero(*args, **kwargs):
        del args, kwargs
        return 0.0, "0", None

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    monkeypatch.setattr(reward_module, "_call_binary_judge", _judge_zero)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
        efficiency_penalty_mode="adaptive_group",
        max_turn_penalty_enabled=True,
        max_turn_penalty=0.25,
    )
    reward_fn = rubric.funcs[0]
    states = [_adaptive_test_state("42", 20) for _ in range(4)]
    states[0]["finalized_on_forced_prompt"] = True

    scores = asyncio.run(reward_fn(states))

    assert scores == [0.75, 1.0, 1.0, 1.0]
    assert states[0]["reward_correctness"] == 1.0
    assert states[0]["reward_group_solve_rate"] == 1.0
    assert states[0]["reward_max_turn_penalty"] == 0.25


def test_adaptive_group_zeroes_missing_final_and_penalizes_valid_correct_rollouts(monkeypatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    async def _judge_zero(*args, **kwargs):
        del args, kwargs
        return 0.0, "0", None

    monkeypatch.setattr(reward_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    monkeypatch.setattr(reward_module, "_call_binary_judge", _judge_zero)
    rubric = build_rubric(
        judge_model="judge-model",
        judge_base_url="http://judge.local/v1",
        judge_api_key="EMPTY",
        efficiency_penalty_mode="adaptive_group",
        adaptive_efficiency_beta_max=0.15,
        adaptive_efficiency_gamma=1.0,
        max_turn_penalty_enabled=True,
        max_turn_penalty=0.25,
        missing_final_at_max_turn_zero_reward=True,
    )
    reward_fn = rubric.funcs[0]
    states = [
        _adaptive_test_state("42", 20),
        _adaptive_test_state("42", 80),
        _adaptive_test_state(None, 100),
        _adaptive_test_state("wrong", 10),
    ]
    states[0]["finalized_on_forced_prompt"] = True
    states[2]["stop_condition"] = "max_turns_reached"
    states[2]["completion"] = [{"content": "42"}]

    scores = asyncio.run(reward_fn(states))

    # Two of four rollouts are valid-correct, so beta is 0.15 * ((0.5 - 0.25) / 0.75).
    expected_beta = 0.15 * ((0.5 - 0.25) / 0.75)
    assert scores == [0.75, 1.0 - expected_beta, 0.0, 0.0]
    assert states[2]["reward_correctness"] == 0.0
    assert states[2]["judge_raw_response"] == "[missing_final_at_max_turn]"
    assert states[0]["reward_max_turn_penalty"] == 0.25


def test_vertex_judge_calls_generate_content_and_parses_binary(monkeypatch) -> None:
    class _FakeThinkingConfig:
        def __init__(self, *, thinking_level: str) -> None:
            self.thinking_level = thinking_level

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeAioModels:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(text="1")

    class _FakeClient:
        last_client = None

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.aio = SimpleNamespace(models=_FakeAioModels())
            _FakeClient.last_client = self

    fake_types = SimpleNamespace(ThinkingConfig=_FakeThinkingConfig, GenerateContentConfig=_FakeGenerateContentConfig)
    monkeypatch.setattr(reward_module, "_load_google_genai", lambda: (SimpleNamespace(Client=_FakeClient), fake_types))

    rubric = build_rubric(
        judge_provider="vertex",
        judge_model="gemini-3-flash-preview",
        judge_base_url="",
        judge_api_key=None,
        judge_vertex_project="test-project",
        judge_vertex_location="global",
        judge_thinking_level="medium",
    )
    reward_fn = rubric.funcs[0]
    state = {
        "final_answer": "forty two",
        "efficiency_penalty_coef": 0.0,
        "rlm_segments": [],
        "trajectory": [],
    }

    score = asyncio.run(reward_fn(state, [], "42", {"question": "What is the answer?"}))

    assert score == 1.0
    client = _FakeClient.last_client
    assert client.kwargs == {"vertexai": True, "project": "test-project", "location": "global"}
    call = client.aio.models.calls[0]
    assert call["model"] == "gemini-3-flash-preview"
    assert call["config"].kwargs["thinking_config"].thinking_level == "medium"
    assert call["config"].kwargs["max_output_tokens"] == 1024
    assert state["judge_raw_response"] == "1"


def test_openai_compatible_judge_retries_rate_limits(monkeypatch) -> None:
    async def _no_sleep(attempt: int) -> None:
        del attempt

    class _FakeRateLimitError(Exception):
        status_code = 429

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **kwargs):
            del kwargs
            self.calls += 1
            if self.calls < 3:
                raise _FakeRateLimitError("rate limited")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="1"))])

    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(reward_module, "_sleep_before_judge_retry", _no_sleep)

    score, raw_response, parse_error = asyncio.run(
        reward_module._call_binary_judge(
            fake_client,
            judge_model="judge-model",
            judge_prompt="prompt",
        )
    )

    assert score == 1.0
    assert raw_response == "1"
    assert parse_error is None
    assert completions.calls == 3


def test_vertex_judge_retries_resource_exhausted(monkeypatch) -> None:
    async def _no_sleep(attempt: int) -> None:
        del attempt

    class _FakeThinkingConfig:
        def __init__(self, *, thinking_level: str) -> None:
            self.thinking_level = thinking_level

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeAioModels:
        def __init__(self) -> None:
            self.calls = 0

        async def generate_content(self, **kwargs):
            del kwargs
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return SimpleNamespace(text="1")

    models = _FakeAioModels()
    fake_client = SimpleNamespace(aio=SimpleNamespace(models=models))
    fake_types = SimpleNamespace(ThinkingConfig=_FakeThinkingConfig, GenerateContentConfig=_FakeGenerateContentConfig)
    monkeypatch.setattr(reward_module, "_load_google_genai", lambda: (SimpleNamespace(), fake_types))
    monkeypatch.setattr(reward_module, "_sleep_before_judge_retry", _no_sleep)

    score, raw_response, parse_error = asyncio.run(
        reward_module._call_vertex_binary_judge(
            fake_client,
            judge_model="gemini-3-flash-preview",
            judge_prompt="prompt",
            thinking_level="medium",
        )
    )

    assert score == 1.0
    assert raw_response == "1"
    assert parse_error is None
    assert models.calls == 3


def test_vertex_judge_retries_access_token_type_unsupported(monkeypatch) -> None:
    async def _no_sleep(attempt: int) -> None:
        del attempt

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _FakeVertexJudgeClient:
        def __init__(self) -> None:
            self.calls = 0

        async def generate_content(self, **kwargs):
            del kwargs
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("401 UNAUTHENTICATED ACCESS_TOKEN_TYPE_UNSUPPORTED")
            return SimpleNamespace(text="1")

    fake_client = _FakeVertexJudgeClient()
    fake_types = SimpleNamespace(GenerateContentConfig=_FakeGenerateContentConfig)
    monkeypatch.setattr(reward_module, "_load_google_genai", lambda: (SimpleNamespace(), fake_types))
    monkeypatch.setattr(reward_module, "_sleep_before_judge_retry", _no_sleep)

    score, raw_response, parse_error = asyncio.run(
        reward_module._call_vertex_binary_judge(
            fake_client,
            judge_model="gemini-3-flash-preview",
            judge_prompt="prompt",
            thinking_level=None,
        )
    )

    assert score == 1.0
    assert raw_response == "1"
    assert parse_error is None
    assert fake_client.calls == 2
