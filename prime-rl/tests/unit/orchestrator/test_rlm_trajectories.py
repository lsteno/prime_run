from prime_rl.orchestrator.rlm_trajectories import rollout_to_training_samples


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
