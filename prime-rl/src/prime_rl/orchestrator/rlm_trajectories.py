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
    sample_local_advantages: list[float | None] | None = None
    eligible_llm_subcalls: int = 0
    selected_llm_subcalls: int = 0
    dropped_llm_subcalls: int = 0


def blend_segment_advantage(
    *, terminal_advantage: float, local_advantage: float | None, local_weight: float
) -> float:
    if local_advantage is None:
        return terminal_advantage
    return local_weight * local_advantage + (1.0 - local_weight) * terminal_advantage


def _segment_is_trainable(segment: dict) -> bool:
    if "is_trainable_rlm_turn" in segment:
        return bool(segment.get("is_trainable_rlm_turn", False))
    return True


def _segment_is_plain_llm_subcall(segment: dict) -> bool:
    return segment.get("train_scope") == "llm_subcall" or segment.get("kind") == "plain_query"


def _select_semantic_subcalls(
    *,
    positions: list[int],
    ordered_segments: list[dict],
    limit: int,
    rng: random.Random,
) -> set[int]:
    if limit <= 0:
        return set()
    selected: list[int] = []
    selected_chunks: set[str] = set()
    for require_local_signal in (True, False):
        candidates = [
            position
            for position in positions
            if position not in selected
            and bool(ordered_segments[position].get("semantic_local_signal")) is require_local_signal
        ]
        by_chunk: dict[str, list[int]] = {}
        without_chunk: list[int] = []
        for position in candidates:
            chunk_id = ordered_segments[position].get("semantic_primary_chunk_id")
            if chunk_id:
                by_chunk.setdefault(str(chunk_id), []).append(position)
            else:
                without_chunk.append(position)
        chunk_ids = list(by_chunk)
        rng.shuffle(chunk_ids)
        for chunk_id in chunk_ids:
            if len(selected) >= limit:
                break
            if chunk_id in selected_chunks:
                continue
            selected.append(rng.choice(by_chunk[chunk_id]))
            selected_chunks.add(chunk_id)
        if len(selected) < limit and without_chunk:
            selected.extend(rng.sample(without_chunk, k=min(limit - len(selected), len(without_chunk))))
        if len(selected) >= limit:
            break
    remaining = [position for position in positions if position not in selected]
    if len(selected) < limit and remaining:
        selected.extend(rng.sample(remaining, k=min(limit - len(selected), len(remaining))))
    return set(selected)


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
    semantic_rollout = any(
        bool(segment.get("semantic_child_segment"))
        for segment in ordered_segments
        if _segment_is_plain_llm_subcall(segment)
    )
    trainable_llm_subcall_positions = [
        idx
        for idx, segment in enumerate(ordered_segments)
        if _segment_is_trainable(segment)
        and _segment_is_plain_llm_subcall(segment)
        and (
            not semantic_rollout
            or bool(segment.get("semantic_local_signal"))
            or bool(segment.get("semantic_terminal_fallback_eligible"))
        )
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
        if semantic_rollout:
            selected_llm_subcall_positions = _select_semantic_subcalls(
                positions=trainable_llm_subcall_positions,
                ordered_segments=ordered_segments,
                limit=max_trainable_llm_subcalls_per_rollout,
                rng=rng,
            )
        else:
            selected_llm_subcall_positions = set(
                rng.sample(trainable_llm_subcall_positions, k=max_trainable_llm_subcalls_per_rollout)
            )

    samples: list[TrainingSample] = []
    sample_local_advantages: list[float | None] = []
    selected_llm_subcalls = 0
    child_only_training = bool(output.get("rlm_child_only_training", False))
    trainable_root_positions = [
        idx
        for idx, segment in enumerate(ordered_segments)
        if _segment_is_trainable(segment)
        and not _segment_is_plain_llm_subcall(segment)
        and not child_only_training
    ]
    root_sequence_weight = 1.0 / len(trainable_root_positions) if trainable_root_positions else 1.0
    for idx, segment in enumerate(ordered_segments):
        if not _segment_is_trainable(segment):
            continue
        if child_only_training and not (
            _segment_is_plain_llm_subcall(segment) and bool(segment.get("semantic_local_signal"))
        ):
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
            loss_branch=(
                "semantic_child"
                if bool(segment.get("semantic_child_segment"))
                else "root"
            ),
            sequence_loss_weight=(1.0 if _segment_is_plain_llm_subcall(segment) else root_sequence_weight),
        )
        samples.append(sample)
        local_advantage = segment.get("semantic_local_advantage")
        sample_local_advantages.append(float(local_advantage) if local_advantage is not None else None)

    return RlmTrainingSampleResult(
        samples=samples,
        sample_local_advantages=sample_local_advantages,
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
