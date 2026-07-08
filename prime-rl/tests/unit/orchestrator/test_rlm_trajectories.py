from prime_rl.orchestrator.rlm_trajectories import rollout_to_training_sample_result, rollout_to_training_samples


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
