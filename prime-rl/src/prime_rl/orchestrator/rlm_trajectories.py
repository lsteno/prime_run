from __future__ import annotations

import verifiers as vf

from prime_rl.orchestrator.trajectories import interleave_rollout
from prime_rl.transport import TrainingSample


def rollout_to_training_samples(
    output: vf.RolloutOutput,
    vlm_cache=None,
    cache_key: int | None = None,
) -> list[TrainingSample] | None:
    segments = output.get("rlm_segments")
    if not segments:
        return interleave_rollout(output, vlm_cache=vlm_cache, cache_key=cache_key)

    has_error = output.get("error") is not None
    default_temperature = float((output.get("sampling_args") or {}).get("temperature", 1.0))
    ordered_segments = sorted(segments, key=lambda item: int(item.get("order", 0)))

    samples: list[TrainingSample] = []
    for segment in ordered_segments:
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

    return samples