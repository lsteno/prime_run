from dataclasses import dataclass
from typing import Any, Callable

import torch
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig, LossConfig
from prime_rl.utils.utils import import_object


@dataclass
class LossInputs:
    """Inputs for computing loss on a single sample."""

    trainer_logprobs: Float[Tensor, " seq"]
    inference_logprobs: Float[Tensor, " seq"]
    teacher_logprobs: Float[Tensor, " seq"] | None
    advantages: Float[Tensor, " seq"]
    loss_mask: Bool[Tensor, " seq"]


@dataclass
class LossOutputs:
    """Outputs from computing loss on a single sample."""

    loss: Float[Tensor, ""]
    metrics: dict[str, Tensor]
    policy_loss: Float[Tensor, ""] | None = None
    regularization_loss: Float[Tensor, ""] | None = None


LossFn = Callable[..., LossOutputs]
"""Type for a per-sample loss function.

Expected signature:
    def my_loss(inputs: LossInputs, **kwargs) -> LossOutputs:
        ...
"""


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def selective_log_softmax(
    logits: Float[Tensor, "batch seq vocab"], index: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq"]:
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def compute_entropy(shifted_logits: Float[Tensor, "batch seq vocab"]) -> Float[Tensor, "batch seq"]:
    with torch.no_grad():
        pd = torch.nn.functional.softmax(shifted_logits, dim=-1)
        entropy = torch.logsumexp(shifted_logits, dim=-1) - torch.sum(pd * shifted_logits, dim=-1)
    return entropy


@jaxtyped(typechecker=typechecker)
def shift_logits(
    logits: Float[Tensor, "batch seq vocab"], left_pad_logit: Float[Tensor, "batch 1 vocab"] | None = None
) -> Float[Tensor, "batch seq vocab"]:
    """Removes final token logits and adds a left pad logit for the first token."""
    # We drop the last logit because it corresponds to the next token that will be sampled but is not here yet
    batch, seq, vocab = logits.shape
    logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
    if left_pad_logit is None:
        left_pad_logit = torch.zeros(batch, 1, vocab, device=logits.device, dtype=logits.dtype)  # (batch, 1, vocab)
    logits = torch.cat([left_pad_logit, logits], dim=1)  # (batch, seq, vocab)
    return logits


def shift_tensor_left(t: Float[Tensor, "batch seq"]) -> Float[Tensor, "batch seq"]:
    """Shifts the tensor one token to the left.

    Used to create labels from input_ids: labels[i] = input_ids[i+1].
    The last position is padded with 0 (a valid token index) since this value
    will be shifted off by shift_tensor_right and never used.
    """
    return torch.cat([t[:, 1:], torch.full((t.shape[0], 1), 0, device=t.device, dtype=t.dtype)], dim=1)


def shift_tensor_right(t: Float[Tensor, "batch seq"], pad_value: float | None = None) -> Float[Tensor, "batch seq"]:
    """Shifts the tensor one token to the right, prepending a padding value.

    Used to realign logprobs/entropy after computing with shifted labels.
    After shift: result[i] = t[i-1], result[0] = pad_value.
    This converts from "predict next token" convention to "probability of current token" convention.

    Args:
        t: Tensor to shift right
        pad_value: Value to use for position 0. If None, uses 0.0 for backward compatibility.
                   For logprobs, should be log(1/vocab_size) to represent uniform distribution.
                   For entropy, should be log(vocab_size) to represent maximum entropy.
    """
    if pad_value is None:
        pad_value = 0.0
    return torch.cat([torch.full((t.shape[0], 1), pad_value, device=t.device, dtype=t.dtype), t[:, :-1]], dim=1)


def _safe_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of values over a boolean mask; returns 0 when mask is empty."""
    denom = torch.clamp_min(mask.sum(), 1)
    return values[mask].sum() / denom


def default_loss_fn(inputs: LossInputs, loss_config: DefaultLossConfig) -> LossOutputs:
    """
    We implement IPO (INTELLECT Policy Optimization) loss, which combines:
    - DPPO-Binary TV Loss (https://arxiv.org/pdf/2602.04879)
    - Kimi-K2.5 KL Loss (https://arxiv.org/pdf/2602.02276)

    Unlike the DPPO-Bin TV mask, we mask independently of the advantage sign.
    This is, because in Async RL, we do not take multiple steps on the same
    data, and so policy updates are not well-predicted by the advantage sign.
    This shift is similar to the shift from GRPO -> CISPO, but with the trust
    region being approximated by the probability difference instead of ratio.
    """
    trainer_logprobs = inputs.trainer_logprobs
    inference_logprobs = inputs.inference_logprobs
    teacher_logprobs = inputs.teacher_logprobs
    advantages = inputs.advantages
    loss_mask = inputs.loss_mask

    trainer_probs = torch.exp(trainer_logprobs)
    inference_probs = torch.exp(inference_logprobs)
    probs_diff = trainer_probs - inference_probs
    ipo_invalid_mask_high = probs_diff > loss_config.ipo_mask_high
    ipo_invalid_mask_low = probs_diff < -loss_config.ipo_mask_low
    ipo_invalid_mask = ipo_invalid_mask_high | ipo_invalid_mask_low

    is_masked = ipo_invalid_mask
    is_masked_low = ipo_invalid_mask_low
    is_masked_high = ipo_invalid_mask_high
    keep_mask = loss_mask & ~is_masked

    log_importance_ratio = trainer_logprobs - inference_logprobs
    importance_ratio = torch.exp(log_importance_ratio)
    mismatch_kl = importance_ratio - log_importance_ratio - 1

    advantages = loss_config.adv_tau * advantages
    if teacher_logprobs is not None:
        teacher_kl = teacher_logprobs - trainer_logprobs
        advantages = advantages + loss_config.teacher_tau * teacher_kl.detach()
    else:
        teacher_kl = None

    pg_loss = keep_mask * advantages * importance_ratio
    kl_loss = loss_mask * log_importance_ratio**2
    policy_loss = -pg_loss.sum()
    regularization_loss = loss_config.kl_tau * kl_loss.sum()
    loss = policy_loss + regularization_loss

    metrics = {
        "mismatch_kl": _safe_mean(mismatch_kl, loss_mask),  # all trainable tokens
        "masked_mismatch_kl": _safe_mean(mismatch_kl, loss_mask & is_masked),  # all trainable, masked tokens
        "unmasked_mismatch_kl": _safe_mean(mismatch_kl, keep_mask),  # all trainable, unmasked tokens
        "is_masked": _safe_mean(is_masked, loss_mask),
        "is_masked_low": _safe_mean(is_masked_low, loss_mask),
        "is_masked_high": _safe_mean(is_masked_high, loss_mask),
    }
    if teacher_kl is not None:
        metrics["teacher_kl"] = _safe_mean(teacher_kl, loss_mask)

    return LossOutputs(
        loss=loss,
        metrics=metrics,
        policy_loss=policy_loss,
        regularization_loss=regularization_loss,
    )


def setup_loss_fn(loss_config: LossConfig) -> LossFn:
    """Setup the loss function based on config."""
    if isinstance(loss_config, CustomLossConfig):
        custom_fn = import_object(loss_config.import_path)
        kwargs = loss_config.kwargs

        def loss_fn(inputs: LossInputs) -> LossOutputs:
            return custom_fn(inputs, **kwargs)

        return loss_fn

    def loss_fn(inputs: LossInputs) -> LossOutputs:
        return default_loss_fn(inputs, loss_config)

    return loss_fn


def compute_loss(
    trainer_logprobs: list[Float[Tensor, " seq_i"]],
    inference_logprobs: list[Float[Tensor, " seq_i"]],
    teacher_logprobs: list[Float[Tensor, " seq_i"]] | None,
    advantages: list[Float[Tensor, " seq_i"]],
    loss_mask: list[Bool[Tensor, " seq_i"]],
    loss_fn: LossFn,
    loss_scale: int,
    loss_branches: list[Int[Tensor, " seq_i"]] | None = None,
    sequence_loss_weights: list[Float[Tensor, " seq_i"]] | None = None,
    normalization: str = "token",
    semantic_child_fraction: float = 0.5,
    global_branch_counts: tuple[int, int] | None = None,
    global_branch_weight_sums: tuple[float, float] | None = None,
    global_trainable_tokens: int | None = None,
    dp_scale: int = 1,
) -> tuple[Float[Tensor, ""], dict[str, Any]]:
    """
    Compute loss for packed sequences (batch size = 1, multiple sequences packed along sequence dimension).

    Args:
        trainer_logprobs: Log probabilities for each sequence
        inference_logprobs: Reference log probabilities for each sequence
        teacher_logprobs: Teacher log probabilities for each sequence, or None
        advantages: Advantages for each sequence
        loss_mask: Loss mask for each sequence
        loss_fn: Per-sequence loss function
        loss_scale: Scale factor to normalize the loss

    Returns:
        Tuple of (scaled_loss, aggregated_metrics)
    """
    zero_loss = trainer_logprobs[0].sum() * 0.0
    total_loss: Tensor = zero_loss
    branch_policy_losses: list[Tensor] = [zero_loss, zero_loss]
    regularization_loss: Tensor = zero_loss
    all_metrics: dict[str, list[Tensor]] = {}

    if teacher_logprobs is None:
        teacher_logprobs = [None] * len(trainer_logprobs)
    if loss_branches is None:
        loss_branches = [torch.zeros_like(mask, dtype=torch.int64) for mask in loss_mask]
    if sequence_loss_weights is None:
        sequence_loss_weights = [torch.ones_like(mask, dtype=torch.float32) for mask in loss_mask]

    for t_logp, i_logp, teach_logp, adv, mask, branches, sequence_weights in zip(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask,
        loss_branches,
        sequence_loss_weights,
        strict=True,
    ):
        inputs = LossInputs(
            trainer_logprobs=t_logp,
            inference_logprobs=i_logp,
            teacher_logprobs=teach_logp,
            advantages=adv,
            loss_mask=mask,
        )

        result = loss_fn(inputs)

        if normalization == "branch_sequence":
            if result.policy_loss is None or result.regularization_loss is None:
                raise ValueError("branch_sequence normalization requires the default structured loss outputs")
            trainable_tokens = int(mask.sum().item())
            if trainable_tokens:
                trainable_branches = torch.unique(branches[mask])
                if trainable_branches.numel() != 1 or int(trainable_branches.item()) not in {0, 1}:
                    raise ValueError("Each training sequence must belong to exactly one supported loss branch")
                trainable_weights = torch.unique(sequence_weights[mask])
                if trainable_weights.numel() != 1 or float(trainable_weights.item()) < 0.0:
                    raise ValueError("Each training sequence must have one non-negative policy weight")
                branch = int(trainable_branches.item())
                sequence_weight = trainable_weights.item()
                branch_policy_losses[branch] = (
                    branch_policy_losses[branch] + sequence_weight * result.policy_loss / trainable_tokens
                )
                regularization_loss = regularization_loss + result.regularization_loss
        else:
            total_loss = total_loss + result.loss

        for k, v in result.metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)

    if normalization == "branch_sequence":
        if global_branch_weight_sums is None and global_branch_counts is not None:
            global_branch_weight_sums = tuple(float(value) for value in global_branch_counts)
        if global_branch_counts is None or global_branch_weight_sums is None or global_trainable_tokens is None:
            raise ValueError("branch_sequence normalization requires global branch counts, weights, and token counts")
        root_count, child_count = global_branch_counts
        root_weight, child_weight = global_branch_weight_sums
        if root_count and child_count:
            root_fraction = 1.0 - semantic_child_fraction
            child_fraction = semantic_child_fraction
        elif root_count:
            root_fraction, child_fraction = 1.0, 0.0
        elif child_count:
            root_fraction, child_fraction = 0.0, 1.0
        else:
            root_fraction = child_fraction = 0.0
        policy_loss: Tensor = zero_loss
        if root_count and root_weight > 0.0:
            policy_loss = policy_loss + root_fraction * branch_policy_losses[0] / root_weight
        if child_count and child_weight > 0.0:
            policy_loss = policy_loss + child_fraction * branch_policy_losses[1] / child_weight
        scaled_loss = dp_scale * (
            policy_loss + regularization_loss / max(global_trainable_tokens, 1)
        )
    else:
        scaled_loss = total_loss / loss_scale

    aggregated: dict[str, Any] = {}
    for k, v in all_metrics.items():
        if v[0].dim() == 0:
            aggregated[k] = torch.stack(v)
        else:
            aggregated[k] = torch.cat(v)

    if normalization == "branch_sequence":
        root_count, child_count = global_branch_counts or (0, 0)
        root_weight, child_weight = global_branch_weight_sums or (0.0, 0.0)
        device = trainer_logprobs[0].device
        root_contribution = (
            dp_scale * root_fraction * branch_policy_losses[0] / root_weight
            if root_count and root_weight > 0.0
            else zero_loss
        )
        child_contribution = (
            dp_scale * child_fraction * branch_policy_losses[1] / child_weight
            if child_count and child_weight > 0.0
            else zero_loss
        )
        aggregated.update(
            {
                "branch_root_policy_loss": torch.as_tensor(branch_policy_losses[0], device=device).reshape(1),
                "branch_child_policy_loss": torch.as_tensor(branch_policy_losses[1], device=device).reshape(1),
                "branch_root_sequence_count": torch.tensor([root_count], device=device, dtype=torch.float32),
                "branch_child_sequence_count": torch.tensor([child_count], device=device, dtype=torch.float32),
                "branch_root_sequence_weight": torch.tensor([root_weight], device=device, dtype=torch.float32),
                "branch_child_sequence_weight": torch.tensor([child_weight], device=device, dtype=torch.float32),
                "branch_root_policy_contribution": root_contribution.reshape(1),
                "branch_child_policy_contribution": child_contribution.reshape(1),
                "branch_semantic_child_fraction": torch.tensor(
                    [semantic_child_fraction if root_count and child_count else float(bool(child_count))],
                    device=device,
                ),
            }
        )

    return scaled_loss, aggregated
