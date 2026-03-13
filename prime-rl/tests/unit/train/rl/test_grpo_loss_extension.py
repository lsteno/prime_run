import torch

from prime_rl.trainer.rl.grpo_loss import clipped_grpo_loss
from prime_rl.trainer.rl.loss import LossInputs


def test_clipped_grpo_loss_matches_manual_objective() -> None:
    inputs = LossInputs(
        trainer_logprobs=torch.tensor([0.0, 0.4]),
        inference_logprobs=torch.tensor([0.0, 0.0]),
        teacher_logprobs=None,
        advantages=torch.tensor([1.0, -1.0]),
        loss_mask=torch.tensor([True, True]),
    )

    result = clipped_grpo_loss(inputs, clip_eps=0.2)

    ratio = torch.exp(inputs.trainer_logprobs - inputs.inference_logprobs)
    clipped = torch.clamp(ratio, 0.8, 1.2)
    expected = -torch.min(ratio * inputs.advantages, clipped * inputs.advantages).sum()

    assert torch.isclose(result.loss, expected)
    assert "clip_frac" in result.metrics