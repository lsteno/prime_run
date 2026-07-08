from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import verifiers as vf

from prime_rl.orchestrator.trajectories import interleave_rollout
from prime_rl.transport import TrainingSample


@dataclass(frozen=True)
class RlmTrainingSampleResult:
    samples: list[TrainingSample] | None
    eligible_llm_subcalls: int = 0
    selected_llm_subcalls: int = 0
    dropped_llm_subcalls: int = 0


def _segment_is_trainable(segment: dict) -> bool:
    if "is_trainable_rlm_turn" in segment:
        return bool(segment.get("is_trainable_rlm_turn", False))
    return True


def _segment_is_plain_llm_subcall(segment: dict) -> bool:
    return segment.get("train_scope") == "llm_subcall" or segment.get("kind") == "plain_query"


def _selection_rng(
    *,
    selection_seed: int | None,
    selection_step: int | None,
    cache_key: int | None,
    example_id: object,
) -> random.Random:
    if selection_seed is None:
        return random.Random()
    payload = f"{selection_seed}:{selection_step}:{cache_key}:{example_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def rollout_to_training_sample_result(
    output: vf.RolloutOutput,
    vlm_cache=None,
    cache_key: int | None = None,
    max_trainable_llm_subcalls_per_rollout: int | None = None,
    selection_seed: int | None = None,
    selection_step: int | None = None,
) -> RlmTrainingSampleResult:
    segments = output.get("rlm_segments")
    if not segments:
        return RlmTrainingSampleResult(samples=interleave_rollout(output, vlm_cache=vlm_cache, cache_key=cache_key))

    has_error = output.get("error") is not None
    default_temperature = float((output.get("sampling_args") or {}).get("temperature", 1.0))
    ordered_segments = sorted(segments, key=lambda item: int(item.get("order", 0)))
    trainable_llm_subcall_positions = [
        idx
        for idx, segment in enumerate(ordered_segments)
        if _segment_is_trainable(segment) and _segment_is_plain_llm_subcall(segment)
    ]
    eligible_llm_subcalls = len(trainable_llm_subcall_positions)
    selected_llm_subcall_positions = set(trainable_llm_subcall_positions)
    if (
        max_trainable_llm_subcalls_per_rollout is not None
        and eligible_llm_subcalls > max_trainable_llm_subcalls_per_rollout
    ):
        rng = _selection_rng(
            selection_seed=selection_seed,
            selection_step=selection_step,
            cache_key=cache_key,
            example_id=output.get("example_id"),
        )
        selected_llm_subcall_positions = set(
            rng.sample(trainable_llm_subcall_positions, k=max_trainable_llm_subcalls_per_rollout)
        )

    samples: list[TrainingSample] = []
    selected_llm_subcalls = 0
    for idx, segment in enumerate(ordered_segments):
        if not _segment_is_trainable(segment):
            continue
        if _segment_is_plain_llm_subcall(segment):
            if idx not in selected_llm_subcall_positions:
                continue
            selected_llm_subcalls += 1
        completion_ids = list(segment["completion_ids"])
        completion_mask = [bool(value) for value in segment.get("completion_mask", [True] * len(completion_ids))]
        if has_error:
            completion_mask = [False] * len(completion_mask)
        sample = TrainingSample(
            prompt_ids=list(segment["prompt_ids"]),
            prompt_mask=[False] * len(segment["prompt_ids"]),
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=[float(value) for value in segment["completion_logprobs"]],
            completion_temperatures=[float(segment.get("temperature", default_temperature))] * len(completion_ids),
            teacher_logprobs=None,
            advantage=None,
        )
        samples.append(sample)

    return RlmTrainingSampleResult(
        samples=samples,
        eligible_llm_subcalls=eligible_llm_subcalls,
        selected_llm_subcalls=selected_llm_subcalls,
        dropped_llm_subcalls=eligible_llm_subcalls - selected_llm_subcalls,
    )


def rollout_to_training_samples(
    output: vf.RolloutOutput,
    vlm_cache=None,
    cache_key: int | None = None,
    max_trainable_llm_subcalls_per_rollout: int | None = None,
    selection_seed: int | None = None,
    selection_step: int | None = None,
) -> list[TrainingSample] | None:
    return rollout_to_training_sample_result(
        output,
        vlm_cache=vlm_cache,
        cache_key=cache_key,
        max_trainable_llm_subcalls_per_rollout=max_trainable_llm_subcalls_per_rollout,
        selection_seed=selection_seed,
        selection_step=selection_step,
    ).samples
