import pytest

from prime_rl.orchestrator.rlm_trajectories import (
    blend_segment_advantage,
    rollout_to_training_sample_result,
    rollout_to_training_samples,
)


def _trainable_subcall(order: int) -> dict:
    return {
        "order": order,
        "kind": "plain_query",
        "train_scope": "llm_subcall",
        "is_trainable_rlm_turn": True,
        "prompt_ids": [100 + order],
        "completion_ids": [200 + order],
        "completion_logprobs": [-0.1 * order],
        "completion_mask": [True],
        "temperature": 0.8,
    }


def test_rollout_to_training_samples_prefers_recursive_segments() -> None:
    rollout = {
        "example_id": 1,
        "reward": 1.0,
        "error": None,
        "sampling_args": {"temperature": 0.8},
        "rlm_segments": [
            {
                "order": 1,
                "is_trainable_rlm_turn": False,
                "prompt_ids": [10, 11],
                "completion_ids": [12],
                "completion_logprobs": [-0.3],
                "completion_mask": [True],
                "temperature": 0.8,
            },
            {
                "order": 0,
                "is_trainable_rlm_turn": True,
                "prompt_ids": [1, 2],
                "completion_ids": [3, 4],
                "completion_logprobs": [-0.1, -0.2],
                "completion_mask": [True, True],
                "temperature": 0.8,
            },
        ],
    }

    samples = rollout_to_training_samples(rollout)

    assert samples is not None
    assert len(samples) == 1
    assert samples[0].prompt_ids == [1, 2]
    assert samples[0].completion_ids == [3, 4]


def test_rollout_to_training_samples_masks_error_segments() -> None:
    rollout = {
        "example_id": 1,
        "reward": 0.0,
        "error": {"error": "boom"},
        "sampling_args": {"temperature": 1.0},
        "rlm_segments": [
            {
                "order": 0,
                "is_trainable_rlm_turn": True,
                "prompt_ids": [1],
                "completion_ids": [2, 3],
                "completion_logprobs": [-0.1, -0.2],
                "completion_mask": [True, True],
            }
        ],
    }

    samples = rollout_to_training_samples(rollout)

    assert samples is not None
    assert samples[0].completion_mask == [False, False]


def test_rollout_to_training_samples_skips_non_trainable_segments() -> None:
    rollout = {
        "example_id": 1,
        "reward": 1.0,
        "error": None,
        "sampling_args": {"temperature": 0.8},
        "rlm_segments": [
            {
                "order": 0,
                "is_trainable_rlm_turn": False,
                "prompt_ids": [1],
                "completion_ids": [2],
                "completion_logprobs": [-0.1],
                "completion_mask": [True],
                "temperature": 0.8,
            }
        ],
    }

    samples = rollout_to_training_samples(rollout)

    assert samples == []


def test_rollout_to_training_samples_keeps_trainable_subcall_context_separate() -> None:
    rollout = {
        "example_id": 1,
        "reward": 1.0,
        "error": None,
        "sampling_args": {"temperature": 0.8},
        "rlm_segments": [
            {
                "order": 0,
                "kind": "root_turn",
                "train_scope": "root_turn",
                "is_trainable_rlm_turn": True,
                "prompt_ids": [1, 2],
                "completion_ids": [3],
                "completion_logprobs": [-0.1],
                "completion_mask": [True],
                "temperature": 0.8,
            },
            {
                "order": 1,
                "kind": "plain_query",
                "train_scope": "llm_subcall",
                "is_trainable_rlm_turn": True,
                "prompt_ids": [10, 11],
                "completion_ids": [12, 13],
                "completion_logprobs": [-0.2, -0.3],
                "completion_mask": [True, True],
                "temperature": 0.8,
            },
        ],
    }

    samples = rollout_to_training_samples(rollout)

    assert samples is not None
    assert len(samples) == 2
    assert samples[0].prompt_ids == [1, 2]
    assert samples[0].completion_ids == [3]
    assert samples[1].prompt_ids == [10, 11]
    assert samples[1].completion_ids == [12, 13]


def test_rollout_to_training_samples_keeps_legacy_segments_trainable() -> None:
    rollout = {
        "example_id": 1,
        "reward": 1.0,
        "error": None,
        "sampling_args": {"temperature": 0.8},
        "rlm_segments": [
            {
                "order": 0,
                "prompt_ids": [1, 2],
                "completion_ids": [3],
                "completion_logprobs": [-0.1],
                "completion_mask": [True],
                "temperature": 0.8,
            }
        ],
    }

    samples = rollout_to_training_samples(rollout)

    assert samples is not None
    assert len(samples) == 1
    assert samples[0].completion_ids == [3]


def test_rollout_to_training_sample_result_caps_trainable_subcalls_and_keeps_root() -> None:
    rollout = {
        "example_id": 7,
        "reward": 1.0,
        "error": None,
        "sampling_args": {"temperature": 0.8},
        "rlm_segments": [
            {
                "order": 0,
                "kind": "root_turn",
                "train_scope": "root_turn",
                "is_trainable_rlm_turn": True,
                "prompt_ids": [1],
                "completion_ids": [2],
                "completion_logprobs": [-0.1],
                "completion_mask": [True],
                "temperature": 0.8,
            },
            _trainable_subcall(1),
            _trainable_subcall(2),
            _trainable_subcall(3),
            _trainable_subcall(4),
        ],
    }

    result = rollout_to_training_sample_result(
        rollout,
        cache_key=9,
        max_trainable_llm_subcalls_per_rollout=2,
        selection_seed=42,
        selection_step=3,
    )

    assert result.samples is not None
    assert result.eligible_llm_subcalls == 4
    assert result.selected_llm_subcalls == 2
    assert result.dropped_llm_subcalls == 2
    assert len(result.samples) == 3
    assert result.samples[0].prompt_ids == [1]
    selected_subcall_prompt_ids = [sample.prompt_ids[0] for sample in result.samples[1:]]
    assert selected_subcall_prompt_ids == sorted(selected_subcall_prompt_ids)


def test_rollout_to_training_sample_result_selection_is_deterministic() -> None:
    rollout = {
        "example_id": 7,
        "reward": 1.0,
        "error": None,
        "sampling_args": {"temperature": 0.8},
        "rlm_segments": [_trainable_subcall(order) for order in range(5)],
    }

    first = rollout_to_training_sample_result(
        rollout,
        cache_key=4,
        max_trainable_llm_subcalls_per_rollout=3,
        selection_seed=123,
        selection_step=2,
    )
    second = rollout_to_training_sample_result(
        rollout,
        cache_key=4,
        max_trainable_llm_subcalls_per_rollout=3,
        selection_seed=123,
        selection_step=2,
    )

    assert first.samples is not None
    assert second.samples is not None
    assert [sample.prompt_ids for sample in first.samples] == [sample.prompt_ids for sample in second.samples]
    assert first.selected_llm_subcalls == 3
    assert first.dropped_llm_subcalls == 2


def test_rollout_to_training_sample_result_zero_cap_drops_only_llm_subcalls() -> None:
    rollout = {
        "example_id": 1,
        "reward": 1.0,
        "error": None,
        "sampling_args": {"temperature": 0.8},
        "rlm_segments": [
            {
                "order": 0,
                "kind": "root_turn",
                "train_scope": "root_turn",
                "is_trainable_rlm_turn": True,
                "prompt_ids": [1],
                "completion_ids": [2],
                "completion_logprobs": [-0.1],
                "completion_mask": [True],
                "temperature": 0.8,
            },
            _trainable_subcall(1),
        ],
    }

    result = rollout_to_training_sample_result(rollout, max_trainable_llm_subcalls_per_rollout=0)

    assert result.samples is not None
    assert len(result.samples) == 1
    assert result.samples[0].prompt_ids == [1]
    assert result.eligible_llm_subcalls == 1
    assert result.selected_llm_subcalls == 0
    assert result.dropped_llm_subcalls == 1


def test_semantic_subcall_cap_samples_distinct_chunks_and_exposes_local_advantages() -> None:
    subcalls = []
    for order, chunk_id, local_advantage in (
        (1, "chunk_001", 1.0),
        (2, "chunk_001", 0.5),
        (3, "chunk_002", 0.0),
        (4, "chunk_003", -1.0),
    ):
        segment = _trainable_subcall(order)
        segment.update(
            {
                "semantic_child_segment": True,
                "semantic_local_signal": True,
                "semantic_primary_chunk_id": chunk_id,
                "semantic_local_advantage": local_advantage,
            }
        )
        subcalls.append(segment)
    rollout = {
        "example_id": 7,
        "reward": 0.5,
        "error": None,
        "sampling_args": {"temperature": 0.8},
        "rlm_segments": subcalls,
    }

    result = rollout_to_training_sample_result(
        rollout,
        cache_key=9,
        max_trainable_llm_subcalls_per_rollout=3,
        selection_seed=42,
        selection_step=3,
    )

    assert result.samples is not None
    assert result.sample_local_advantages is not None
    assert len(result.samples) == 3
    assert all(sample.loss_branch == "semantic_child" for sample in result.samples)
    selected_orders = {sample.prompt_ids[0] - 100 for sample in result.samples}
    selected_chunks = {
        next(segment["semantic_primary_chunk_id"] for segment in subcalls if segment["order"] == order)
        for order in selected_orders
    }
    assert selected_chunks == {"chunk_001", "chunk_002", "chunk_003"}
    assert all(value is not None for value in result.sample_local_advantages)


def test_semantic_child_only_rollout_emits_no_root_sample() -> None:
    child = _trainable_subcall(1)
    child.update(
        {
            "semantic_child_segment": True,
            "semantic_local_signal": True,
            "semantic_primary_chunk_id": "chunk_001",
            "semantic_local_advantage": -0.5,
        }
    )
    rollout = {
        "example_id": 1,
        "reward": 0.0,
        "error": None,
        "sampling_args": {"temperature": 0.8},
        "rlm_child_only_training": True,
        "rlm_segments": [
            {
                "order": 0,
                "kind": "root_turn",
                "train_scope": "root_turn",
                "is_trainable_rlm_turn": True,
                "prompt_ids": [1],
                "completion_ids": [2],
                "completion_logprobs": [-0.1],
                "completion_mask": [True],
            },
            child,
        ],
    }

    result = rollout_to_training_sample_result(rollout)

    assert result.samples is not None
    assert [sample.prompt_ids for sample in result.samples] == [[101]]
    assert result.samples[0].loss_branch == "semantic_child"
    assert result.sample_local_advantages == [-0.5]


def test_semantic_child_without_recognized_local_signal_is_not_trained() -> None:
    child = _trainable_subcall(1)
    child.update({"semantic_child_segment": True, "semantic_local_signal": False})
    rollout = {
        "example_id": 1,
        "reward": 1.0,
        "error": None,
        "sampling_args": {"temperature": 0.8},
        "rlm_segments": [child],
    }

    result = rollout_to_training_sample_result(rollout)

    assert result.samples == []
    assert result.eligible_llm_subcalls == 0


def test_semantic_advantage_blend_is_80_percent_local() -> None:
    assert blend_segment_advantage(
        terminal_advantage=0.5, local_advantage=-1.0, local_weight=0.8
    ) == pytest.approx(-0.7)
    assert blend_segment_advantage(terminal_advantage=0.5, local_advantage=None, local_weight=0.8) == 0.5
