from __future__ import annotations

import torch

from prime_rl.trainer.rl.loss import LossInputs, LossOutputs


def clipped_grpo_loss(
    inputs: LossInputs,
    *,
    clip_eps: float = 0.2,
    epsilon_low: float | None = None,
    epsilon_high: float | None = None,
) -> LossOutputs:
    lower = 1.0 - (clip_eps if epsilon_low is None else epsilon_low)
    upper = 1.0 + (clip_eps if epsilon_high is None else epsilon_high)

    ratio = torch.exp(inputs.trainer_logprobs - inputs.inference_logprobs)
    clipped_ratio = torch.clamp(ratio, min=lower, max=upper)
    surr1 = ratio * inputs.advantages
    surr2 = clipped_ratio * inputs.advantages
    masked_loss = -torch.min(surr1, surr2)[inputs.loss_mask]
    loss = masked_loss.sum()

    clip_frac = (ratio != clipped_ratio)[inputs.loss_mask].float().mean() if inputs.loss_mask.any() else torch.tensor(0.0)
    return LossOutputs(loss=loss, metrics={"clip_frac": clip_frac})