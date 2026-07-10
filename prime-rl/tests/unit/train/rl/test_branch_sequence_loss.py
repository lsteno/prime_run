from __future__ import annotations

import torch

from prime_rl.trainer.rl.loss import LossInputs, LossOutputs, compute_loss


def _linear_policy_loss(inputs: LossInputs) -> LossOutputs:
    policy_loss = inputs.advantages[inputs.loss_mask].sum()
    zero = policy_loss * 0.0
    return LossOutputs(
        loss=policy_loss,
        metrics={},
        policy_loss=policy_loss,
        regularization_loss=zero,
    )


def _sequence(length: int, advantage: float, branch: int):
    return (
        torch.zeros(length),
        torch.zeros(length),
        torch.full((length,), advantage),
        torch.ones(length, dtype=torch.bool),
        torch.full((length,), branch, dtype=torch.int64),
    )


def test_branch_sequence_loss_is_length_neutral_and_balanced() -> None:
    root = _sequence(100, 2.0, 0)
    child = _sequence(1, 4.0, 1)

    loss, metrics = compute_loss(
        trainer_logprobs=[root[0], child[0]],
        inference_logprobs=[root[1], child[1]],
        teacher_logprobs=None,
        advantages=[root[2], child[2]],
        loss_mask=[root[3], child[3]],
        loss_fn=_linear_policy_loss,
        loss_scale=101,
        loss_branches=[root[4], child[4]],
        normalization="branch_sequence",
        semantic_child_fraction=0.5,
        global_branch_counts=(1, 1),
        global_trainable_tokens=101,
    )

    assert torch.isclose(loss, torch.tensor(3.0))
    assert metrics["branch_root_sequence_count"].item() == 1
    assert metrics["branch_child_sequence_count"].item() == 1


def test_branch_sequence_loss_renormalizes_when_only_children_exist() -> None:
    child = _sequence(7, -1.0, 1)

    loss, _ = compute_loss(
        trainer_logprobs=[child[0]],
        inference_logprobs=[child[1]],
        teacher_logprobs=None,
        advantages=[child[2]],
        loss_mask=[child[3]],
        loss_fn=_linear_policy_loss,
        loss_scale=7,
        loss_branches=[child[4]],
        normalization="branch_sequence",
        semantic_child_fraction=0.5,
        global_branch_counts=(0, 1),
        global_trainable_tokens=7,
    )

    assert torch.isclose(loss, torch.tensor(-1.0))


def test_branch_sequence_loss_weights_repeated_root_turns_per_rollout() -> None:
    repeated_rollout = [_sequence(3, 2.0, 0) for _ in range(4)]
    concise_rollout = _sequence(3, 4.0, 0)
    sequences = [*repeated_rollout, concise_rollout]
    sequence_weights = [
        torch.full((3,), weight)
        for weight in (0.25, 0.25, 0.25, 0.25, 1.0)
    ]

    loss, metrics = compute_loss(
        trainer_logprobs=[sequence[0] for sequence in sequences],
        inference_logprobs=[sequence[1] for sequence in sequences],
        teacher_logprobs=None,
        advantages=[sequence[2] for sequence in sequences],
        loss_mask=[sequence[3] for sequence in sequences],
        loss_fn=_linear_policy_loss,
        loss_scale=15,
        loss_branches=[sequence[4] for sequence in sequences],
        sequence_loss_weights=sequence_weights,
        normalization="branch_sequence",
        semantic_child_fraction=0.5,
        global_branch_counts=(5, 0),
        global_branch_weight_sums=(2.0, 0.0),
        global_trainable_tokens=15,
    )

    assert torch.isclose(loss, torch.tensor(3.0))
    assert metrics["branch_root_sequence_count"].item() == 5
    assert metrics["branch_root_sequence_weight"].item() == 2.0
    assert torch.isclose(metrics["branch_root_policy_contribution"], torch.tensor([3.0])).all()


def test_token_normalization_preserves_length_weighting() -> None:
    root = _sequence(100, 2.0, 0)
    child = _sequence(1, 4.0, 1)

    loss, _ = compute_loss(
        trainer_logprobs=[root[0], child[0]],
        inference_logprobs=[root[1], child[1]],
        teacher_logprobs=None,
        advantages=[root[2], child[2]],
        loss_mask=[root[3], child[3]],
        loss_fn=_linear_policy_loss,
        loss_scale=101,
    )

    assert torch.isclose(loss, torch.tensor(204.0 / 101.0))
