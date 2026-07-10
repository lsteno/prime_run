from __future__ import annotations

import ast
import asyncio
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI
import verifiers as vf

from .live_trace import write_live_trace
from .semantic_evidence import parse_semantic_prompt_evidence

JUDGE_PROMPT = """You are a binary grader.

You will be given three things:
1. The dataset task.
2. One or more gold reference answers for that task.
3. The candidate answer produced by the model you are evaluating.

Your job is to judge whether the candidate answer is semantically correct with respect to the dataset task and the gold reference answer(s).

Dataset task:
{question}

Gold reference answer(s):
{expected_answers}

Candidate model answer:
{predicted_answer}

How to score:
- Return 1 if the candidate answer clearly conveys the same final answer as any gold reference answer.
- Semantic correctness matters more than exact wording, formatting, quoting style, or JSON formatting.
- Return 1 if the candidate answer contains the correct fact, entity, string, or number, even if extra harmless text is present.
- Return 0 if the candidate answer is missing the core answer, gives the wrong answer, contradicts the correct answer, is only scratchpad/reasoning/code without a clear final answer, or is empty.
- If the candidate answer is truncated, malformed, or noisy, you should return 1 only if the correct final answer is clearly present.
- The only valid outputs are 0 or 1.

Return exactly one character: 0 or 1.
"""

JUDGE_SYSTEM_PROMPT = (
    "You are a strict binary grader. Return exactly one character: 0 or 1. "
    "Do not return JSON, explanations, tool calls, or any other text."
)

_EFFICIENCY_PENALTY_PER_1K_TOKENS = 1000.0
_JUDGE_RETRY_MAX_ATTEMPTS = 6
_JUDGE_RETRY_BASE_SECONDS = 1.0
_JUDGE_RETRY_MAX_SECONDS = 30.0
_VERTEX_JUDGE_MAX_OUTPUT_TOKENS = 1024
_VALID_EFFICIENCY_PENALTY_MODES = {
    "static_per_1k",
    "adaptive_group",
    "accuracy_stratified_group",
}
_VALID_ADAPTIVE_COST_BASES = {"total_tokens", "weighted_turn_tokens"}
_VALID_EFFICIENCY_PENALTY_SCOPES = {"correct_only", "all_rollouts"}
_JUDGE_TRUNCATION_MARKER = "\n\n[truncated before semantic judging]"


@dataclass(frozen=True)
class CorrectnessResult:
    predicted_answer: str
    expected_answers: list[str]
    score: float
    raw_response: str
    parse_error: str | None


@dataclass(frozen=True)
class TokenBreakdown:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    trainable_tokens: int
    rlm_turn_tokens: int
    plain_subcall_tokens: int


@dataclass(frozen=True)
class SemanticDelegationScore:
    task_score: float
    chunk_accuracy: float
    progress: float
    exact: float
    schema_valid: float
    extra_keys: int = 0


def _truncate_for_judge(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep_chars = max(0, max_chars - len(_JUDGE_TRUNCATION_MARKER))
    return text[:keep_chars].rstrip() + _JUDGE_TRUNCATION_MARKER


def _is_oversized_judge_exception(exc: BaseException) -> bool:
    error_text = f"{type(exc).__name__}: {exc}".casefold()
    return any(
        marker in error_text
        for marker in (
            "text fields that are too large",
            "input token count",
            "exceeds",
            "too many tokens",
            "request payload size exceeds",
            "payload too large",
        )
    )


def _get_predicted_answer(state: vf.State, completion) -> str:
    final_answer = state.get("final_answer")
    if final_answer is not None:
        return str(final_answer)
    if completion:
        return str(completion[-1].get("content", ""))
    return ""


def _last_rlm_debug(state: vf.State) -> dict[str, Any]:
    trajectory = state.get("trajectory") or []
    if not trajectory:
        return {}
    last_step = trajectory[-1]
    if not isinstance(last_step, dict):
        return {}
    extras = last_step.get("extras") or {}
    debug = extras.get("rlm_debug") or {}
    return debug if isinstance(debug, dict) else {}


def _state_bool_with_debug_fallback(state: vf.State, key: str) -> bool:
    if key in state:
        return bool(state.get(key))
    return bool(_last_rlm_debug(state).get(key))


def _get_expected_answers(answer: str, info: dict[str, Any] | None) -> list[str]:
    if info and info.get("acceptable_answers"):
        return [str(item) for item in info["acceptable_answers"]]
    return [answer]


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
        text = text[1:-1].strip()
    return text


def _canonicalize_json(value: str) -> str | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonicalize_number(value: str) -> str | None:
    candidate = value.replace(",", "").strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", candidate):
        return None

    try:
        normalized = format(Decimal(candidate).normalize(), "f")
    except InvalidOperation:
        return None

    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _equivalent_forms(value: Any) -> set[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return {""}

    forms = {normalized, normalized.casefold()}
    json_form = _canonicalize_json(normalized)
    if json_form is not None:
        forms.add(json_form)

    number_form = _canonicalize_number(normalized)
    if number_form is not None:
        forms.add(number_form)

    return forms


def _is_exact_match(predicted_answer: str, expected_answers: list[str]) -> bool:
    predicted_forms = _equivalent_forms(predicted_answer)
    for expected_answer in expected_answers:
        if predicted_forms & _equivalent_forms(expected_answer):
            return True
    return False


def _parse_binary_judge_score(raw_text: str) -> float:
    text = raw_text.strip()
    if text in {"0", "1"}:
        return float(text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        score = parsed.get("score")
        if score in {0, 1, 0.0, 1.0, "0", "1"}:
            return float(score)
    elif parsed in {0, 1, 0.0, 1.0, "0", "1"}:
        return float(parsed)

    match = re.search(r"\b([01])\b", text)
    if match is not None:
        return float(match.group(1))
    raise ValueError(
        f"Judge response did not contain a valid binary score: {raw_text!r}"
    )


def _format_expected_answers(answers: list[str]) -> str:
    return "\n".join(f"- {answer}" for answer in answers)


_PAIR_PATTERN = re.compile(r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)")


def _extract_pair_set(value: Any) -> set[tuple[str, str]]:
    if value is None:
        return set()
    if isinstance(value, list):
        texts = [_normalize_text(item) for item in value]
    else:
        text = _normalize_text(value)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            texts = [_normalize_text(item) for item in parsed]
        else:
            texts = [text]

    pairs: set[tuple[str, str]] = set()
    for text in texts:
        for left, right in _PAIR_PATTERN.findall(text):
            first, second = sorted((left, right), key=lambda item: int(item))
            pairs.add((first, second))
    return pairs


def _is_oolong_pairs_task(info: dict[str, Any] | None) -> bool:
    if not info:
        return False
    dataset_name = str(info.get("dataset_name") or "")
    answer_type = str(info.get("answer_type") or "")
    metadata = info.get("metadata") if isinstance(info.get("metadata"), dict) else {}
    origin = str(
        metadata.get("benchmark_origin") or metadata.get("original_benchmark") or ""
    )
    return (
        dataset_name == "oolong_pairs"
        or origin == "oolong_pairs"
        or "oolong_pairs" in str(info.get("source_task") or "")
        or answer_type == "list_of_answers"
    )


def _score_oolong_pairs(
    predicted_answer: str, expected_answers: list[str]
) -> tuple[float, str, dict[str, float]]:
    predicted_pairs = _extract_pair_set(predicted_answer)
    expected_pairs: set[tuple[str, str]] = set()
    for answer in expected_answers:
        expected_pairs |= _extract_pair_set(answer)
    if not expected_pairs:
        score = 1.0 if not predicted_pairs else 0.0
        stats = {
            "precision": score,
            "recall": score,
            "f1": score,
            "predicted_pairs": float(len(predicted_pairs)),
            "expected_pairs": 0.0,
        }
        return score, "[oolong_pairs_empty_gold]", stats

    true_positive = len(predicted_pairs & expected_pairs)
    precision = true_positive / len(predicted_pairs) if predicted_pairs else 0.0
    recall = true_positive / len(expected_pairs)
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if precision + recall > 0.0
        else 0.0
    )
    stats = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_pairs": float(len(predicted_pairs)),
        "expected_pairs": float(len(expected_pairs)),
    }
    return f1, "[oolong_pairs_f1]", stats


def _record_oolong_pairs_metrics(state: vf.State, stats: dict[str, float]) -> None:
    state["oolong_pairs_precision"] = stats["precision"]
    state["oolong_pairs_recall"] = stats["recall"]
    state["oolong_pairs_f1"] = stats["f1"]
    state["oolong_pairs_predicted_count"] = stats["predicted_pairs"]
    state["oolong_pairs_expected_count"] = stats["expected_pairs"]


def _is_oolong_semantic_aggregation_task(info: dict[str, Any] | None) -> bool:
    if not info:
        return False
    source_task = str(info.get("source_task") or "")
    metadata = info.get("metadata") if isinstance(info.get("metadata"), dict) else {}
    derived_dataset = str(metadata.get("derived_dataset") or "")
    return derived_dataset == "oolong_semantic_agg_v2" or (
        source_task.startswith("TASK_TYPE.SEMANTIC_")
        and derived_dataset
        not in {
            "oolong_semantic_delegation_v3",
            "oolong_semantic_delegation_v4",
            "oolong_semantic_delegation_v5",
            "oolong_semantic_delegation_v6",
        }
    )


def _semantic_delegation_metadata(info: dict[str, Any] | None) -> dict[str, Any] | None:
    if not info:
        return None
    metadata = info.get("metadata")
    if not isinstance(metadata, dict):
        return None
    if metadata.get("derived_dataset") not in {
        "oolong_semantic_delegation_v3",
        "oolong_semantic_delegation_v4",
        "oolong_semantic_delegation_v5",
        "oolong_semantic_delegation_v6",
    }:
        return None
    return metadata


def _normalize_semantic_label(value: Any, label_space: list[str]) -> str | None:
    candidate = _strip_answer_prefix(str(value)).strip()
    final_match = re.fullmatch(
        r"FINAL\((.*)\)", candidate, flags=re.IGNORECASE | re.DOTALL
    )
    if final_match:
        candidate = _normalize_text(final_match.group(1))
    by_normalized = {_normalize_text(label).casefold(): label for label in label_space}
    return by_normalized.get(_normalize_text(candidate).casefold())


def _score_semantic_delegation_v3(
    predicted_answer: str,
    info: dict[str, Any] | None,
) -> tuple[SemanticDelegationScore, str, str | None]:
    metadata = _semantic_delegation_metadata(info)
    if metadata is None:
        raise ValueError("Semantic delegation v3 scoring requires verifier metadata.")
    label_space = [str(label) for label in metadata.get("label_space") or []]
    task_type = str(metadata.get("semantic_task_type") or "")

    if task_type == "global_comparison":
        expected = str(metadata.get("global_outcome") or "")
        candidate = _strip_answer_prefix(predicted_answer)
        normalized = _normalize_semantic_label(candidate, [*label_space, "same"])
        exact = 1.0 if normalized == expected else 0.0
        return (
            SemanticDelegationScore(
                task_score=exact,
                chunk_accuracy=exact,
                progress=exact,
                exact=exact,
                schema_valid=1.0 if normalized is not None else 0.0,
            ),
            "[semantic_delegation_global_exact]"
            if exact
            else "[semantic_delegation_global_mismatch]",
            None if normalized is not None else "invalid_global_label",
        )

    expected_map = {
        str(chunk_id): str(label)
        for chunk_id, label in (metadata.get("chunk_labels") or {}).items()
    }
    if not expected_map:
        return (
            SemanticDelegationScore(0.0, 0.0, 0.0, 0.0, 0.0),
            "[semantic_delegation_missing_gold]",
            "missing_gold",
        )
    parsed = _extract_json_object(predicted_answer)
    if parsed is None:
        return (
            SemanticDelegationScore(0.0, 0.0, 0.0, 0.0, 0.0),
            "[semantic_delegation_invalid_json]",
            "invalid_json",
        )

    normalized_keys = {str(key).casefold(): str(key) for key in expected_map}
    predicted_map: dict[str, str] = {}
    schema_valid = True
    extra_keys = 0
    for raw_key, raw_label in parsed.items():
        expected_key = normalized_keys.get(str(raw_key).casefold())
        if expected_key is None:
            extra_keys += 1
            continue
        normalized_label = _normalize_semantic_label(raw_label, label_space)
        if normalized_label is None:
            schema_valid = False
            continue
        predicted_map[expected_key] = normalized_label

    correct = sum(
        predicted_map.get(chunk_id) == expected_label
        for chunk_id, expected_label in expected_map.items()
    )
    chunk_accuracy = correct / len(expected_map)
    is_v6 = metadata.get("derived_dataset") == "oolong_semantic_delegation_v6"
    progress = chunk_accuracy if is_v6 else max(0.0, 2.0 * chunk_accuracy - 1.0)
    exact = float(correct == len(expected_map) and extra_keys == 0 and schema_valid)
    task_score = chunk_accuracy if is_v6 else 0.5 * exact + 0.5 * progress
    return (
        SemanticDelegationScore(
            task_score=task_score,
            chunk_accuracy=chunk_accuracy,
            progress=progress,
            exact=exact,
            schema_valid=float(schema_valid),
            extra_keys=extra_keys,
        ),
        "[semantic_delegation_exact]" if exact else "[semantic_delegation_partial]",
        None if schema_valid else "invalid_chunk_label",
    )


def _annotate_semantic_child_segments_v3(
    state: vf.State, info: dict[str, Any] | None
) -> None:
    metadata = _semantic_delegation_metadata(info)
    if metadata is None:
        return
    gold_labels = {
        str(record_id): str(label)
        for record_id, label in (metadata.get("record_labels") or {}).items()
    }
    record_chunks = {
        str(record_id): str(chunk_id)
        for record_id, chunk_id in (metadata.get("record_chunks") or {}).items()
    }
    label_space = [str(label) for label in metadata.get("label_space") or []]
    segments = state.get("rlm_segments")
    if not isinstance(segments, list):
        return

    total_expected = 0
    total_correct = 0
    total_predictions = 0
    queried_ids: list[str] = []
    locally_scored_segments = 0
    exact_segments = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        if (
            segment.get("kind") != "plain_query"
            and segment.get("train_scope") != "llm_subcall"
        ):
            continue
        segment["semantic_child_segment"] = True
        prompt_ids = [
            str(record_id)
            for record_id in segment.get("semantic_prompt_record_ids") or []
            if str(record_id) in gold_labels
        ]
        prompt_ids = list(dict.fromkeys(prompt_ids))
        chunks = sorted(
            {
                record_chunks[record_id]
                for record_id in prompt_ids
                if record_id in record_chunks
            }
        )
        segment["semantic_recognized_record_count"] = len(prompt_ids)
        segment["semantic_recognized_chunk_ids"] = chunks
        segment["semantic_primary_chunk_id"] = chunks[0] if len(chunks) == 1 else None
        queried_ids.extend(prompt_ids)
        if not prompt_ids:
            segment["semantic_local_signal"] = False
            segment["semantic_local_schema_valid"] = False
            continue

        response_text = str(segment.get("response_text") or "")
        parsed = _extract_json_object(response_text)
        predictions: dict[str, str] = {}
        schema_valid = parsed is not None
        if parsed is not None:
            prompt_lookup = {
                record_id.casefold(): record_id for record_id in prompt_ids
            }
            for raw_id, raw_label in parsed.items():
                record_id = prompt_lookup.get(str(raw_id).casefold())
                if record_id is None:
                    continue
                label = _normalize_semantic_label(raw_label, label_space)
                if label is not None:
                    predictions[record_id] = label
        elif len(prompt_ids) == 1:
            label = _normalize_semantic_label(response_text, label_space)
            if label is not None:
                predictions[prompt_ids[0]] = label
                schema_valid = True

        correct = sum(
            predictions.get(record_id) == gold_labels[record_id]
            for record_id in prompt_ids
        )
        local_accuracy = correct / len(prompt_ids)
        coverage = len(predictions) / len(prompt_ids)
        local_advantage = 2.0 * local_accuracy - 1.0
        local_exact = float(
            correct == len(prompt_ids) and len(predictions) == len(prompt_ids)
        )
        segment.update(
            {
                "semantic_local_signal": True,
                "semantic_local_accuracy": local_accuracy,
                "semantic_local_coverage": coverage,
                "semantic_local_schema_valid": bool(schema_valid),
                "semantic_local_advantage": local_advantage,
                "semantic_local_exact": local_exact,
                "semantic_local_prediction_count": len(predictions),
            }
        )
        locally_scored_segments += 1
        exact_segments += int(local_exact)
        total_expected += len(prompt_ids)
        total_correct += correct
        total_predictions += len(predictions)

    unique_queried = set(queried_ids)
    state["semantic_child_scored_segments"] = float(locally_scored_segments)
    state["semantic_child_local_accuracy"] = (
        total_correct / total_expected if total_expected else 0.0
    )
    state["semantic_child_local_coverage"] = (
        total_predictions / total_expected if total_expected else 0.0
    )
    state["semantic_child_exact_rate"] = (
        exact_segments / locally_scored_segments if locally_scored_segments else 0.0
    )
    state["semantic_records_per_subcall"] = (
        total_expected / locally_scored_segments if locally_scored_segments else 0.0
    )
    state["semantic_record_coverage"] = (
        len(unique_queried) / len(gold_labels) if gold_labels else 0.0
    )
    state["semantic_duplicate_coverage"] = (
        (len(queried_ids) - len(unique_queried)) / len(queried_ids)
        if queried_ids
        else 0.0
    )


def _strict_semantic_input(
    segment: dict[str, Any],
    *,
    gold_labels: dict[str, str],
    record_chunks: dict[str, str],
    context_hashes: dict[str, str],
    chunk_records: dict[str, set[str]],
    min_record_map_records: int,
) -> tuple[str | None, list[str], str | None, bool]:
    prompt_hashes = {
        str(record_id).casefold(): str(text_hash)
        for record_id, text_hash in (
            segment.get("semantic_prompt_record_hashes") or {}
        ).items()
    }
    prompt_ids = list(prompt_hashes)
    if not prompt_ids:
        return None, [], "no_structured_records", False
    if segment.get("semantic_prompt_malformed_record_lines"):
        return None, [], "malformed_record_line", False
    if segment.get("semantic_prompt_duplicate_record_ids"):
        return None, [], "duplicate_record_id", False
    if any(
        record_id not in gold_labels or record_id not in context_hashes
        for record_id in prompt_ids
    ):
        return None, [], "unknown_record_id", False
    if any(
        prompt_hashes[record_id] != context_hashes[record_id]
        for record_id in prompt_ids
    ):
        return None, [], "record_text_mismatch", False

    chunks = {record_chunks[record_id] for record_id in prompt_ids}
    if len(chunks) != 1:
        return None, [], "mixed_chunks", False
    chunk_id = next(iter(chunks))
    full_chunk = set(prompt_ids) == chunk_records.get(chunk_id, set())
    if full_chunk:
        return chunk_id, prompt_ids, None, True
    if len(prompt_ids) < min_record_map_records:
        return None, prompt_ids, "too_few_records", False
    return chunk_id, prompt_ids, None, False


def _score_strict_semantic_child(
    response_text: str,
    *,
    chunk_id: str,
    prompt_ids: list[str],
    full_chunk: bool,
    gold_labels: dict[str, str],
    chunk_labels: dict[str, str],
    label_space: list[str],
) -> tuple[str, float, float, bool, int]:
    parsed = _extract_json_object(response_text)
    normalized_chunk_label: str | None = None
    if full_chunk:
        if parsed is not None and set(map(str, parsed)) == {chunk_id}:
            normalized_chunk_label = _normalize_semantic_label(
                parsed[chunk_id], label_space
            )
        elif parsed is None:
            normalized_chunk_label = _normalize_semantic_label(
                response_text, label_space
            )
    if normalized_chunk_label is not None:
        accuracy = float(normalized_chunk_label == chunk_labels[chunk_id])
        return "chunk_label", accuracy, 1.0, True, 1

    predictions: dict[str, str] = {}
    schema_valid = parsed is not None
    if parsed is not None:
        prompt_lookup = {record_id.casefold(): record_id for record_id in prompt_ids}
        for raw_id, raw_label in parsed.items():
            record_id = prompt_lookup.get(str(raw_id).casefold())
            label = _normalize_semantic_label(raw_label, label_space)
            if record_id is None or label is None:
                schema_valid = False
                continue
            predictions[record_id] = label

    correct = sum(
        predictions.get(record_id) == gold_labels[record_id] for record_id in prompt_ids
    )
    accuracy = correct / len(prompt_ids) if schema_valid else 0.0
    coverage = len(predictions) / len(prompt_ids)
    return "record_map", accuracy, coverage, schema_valid, len(predictions)


def _annotate_semantic_child_segments_v4(state: vf.State, info: dict[str, Any]) -> None:
    metadata = _semantic_delegation_metadata(info)
    assert metadata is not None
    gold_labels = {
        str(record_id).casefold(): str(label)
        for record_id, label in (metadata.get("record_labels") or {}).items()
    }
    record_chunks = {
        str(record_id).casefold(): str(chunk_id)
        for record_id, chunk_id in (metadata.get("record_chunks") or {}).items()
    }
    chunk_labels = {
        str(chunk_id): str(label)
        for chunk_id, label in (metadata.get("chunk_labels") or {}).items()
    }
    chunk_records: dict[str, set[str]] = {}
    for record_id, chunk_id in record_chunks.items():
        chunk_records.setdefault(chunk_id, set()).add(record_id)
    context_hashes = parse_semantic_prompt_evidence(
        str(info.get("context") or "")
    ).record_hashes
    label_space = [str(label) for label in metadata.get("label_space") or []]
    min_records = int(state.get("semantic_record_map_min_records", 8))
    segments = state.get("rlm_segments")
    if not isinstance(segments, list):
        return

    child_count = 0
    verified_count = 0
    valid_output_count = 0
    chunk_contract_count = 0
    record_contract_count = 0
    exact_count = 0
    local_accuracy_sum = 0.0
    local_coverage_sum = 0.0
    prediction_count = 0
    expected_count = 0
    queried_ids: list[str] = []
    verified_full_chunks: set[str] = set()
    rejection_counts: dict[str, int] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        if (
            segment.get("kind") != "plain_query"
            and segment.get("train_scope") != "llm_subcall"
        ):
            continue
        child_count += 1
        segment["semantic_child_segment"] = True
        chunk_id, prompt_ids, rejection_reason, full_chunk = _strict_semantic_input(
            segment,
            gold_labels=gold_labels,
            record_chunks=record_chunks,
            context_hashes=context_hashes,
            chunk_records=chunk_records,
            min_record_map_records=min_records,
        )
        segment["semantic_recognized_record_count"] = len(prompt_ids)
        segment["semantic_recognized_chunk_ids"] = [chunk_id] if chunk_id else []
        segment["semantic_primary_chunk_id"] = chunk_id
        segment["semantic_full_chunk"] = full_chunk
        segment["semantic_input_verified"] = rejection_reason is None
        segment["semantic_input_rejection_reason"] = rejection_reason
        if rejection_reason is not None or chunk_id is None:
            segment["semantic_local_signal"] = False
            segment["semantic_local_schema_valid"] = False
            rejection_counts[rejection_reason or "unknown"] = (
                rejection_counts.get(rejection_reason or "unknown", 0) + 1
            )
            continue

        verified_count += 1
        queried_ids.extend(prompt_ids)
        if full_chunk:
            verified_full_chunks.add(chunk_id)
        contract, accuracy, coverage, schema_valid, predictions = (
            _score_strict_semantic_child(
                str(segment.get("response_text") or ""),
                chunk_id=chunk_id,
                prompt_ids=prompt_ids,
                full_chunk=full_chunk,
                gold_labels=gold_labels,
                chunk_labels=chunk_labels,
                label_space=label_space,
            )
        )
        local_advantage = 2.0 * accuracy - 1.0
        segment.update(
            {
                "semantic_local_signal": True,
                "semantic_local_contract": contract,
                "semantic_local_accuracy": accuracy,
                "semantic_local_coverage": coverage,
                "semantic_local_schema_valid": schema_valid,
                "semantic_local_advantage": local_advantage,
                "semantic_local_exact": float(accuracy == 1.0 and schema_valid),
                "semantic_local_prediction_count": predictions,
            }
        )
        valid_output_count += int(schema_valid)
        chunk_contract_count += int(contract == "chunk_label")
        record_contract_count += int(contract == "record_map")
        exact_count += int(accuracy == 1.0 and schema_valid)
        local_accuracy_sum += accuracy
        local_coverage_sum += coverage
        prediction_count += predictions
        expected_count += 1 if contract == "chunk_label" else len(prompt_ids)

    unique_queried = set(queried_ids)
    denominator = verified_count or 1
    state["semantic_child_scored_segments"] = float(verified_count)
    state["semantic_child_local_accuracy"] = local_accuracy_sum / denominator
    state["semantic_child_local_coverage"] = local_coverage_sum / denominator
    state["semantic_child_exact_rate"] = exact_count / denominator
    state["semantic_child_verified_input_rate"] = (
        verified_count / child_count if child_count else 0.0
    )
    state["semantic_child_invalid_input_rate"] = (
        1.0 - state["semantic_child_verified_input_rate"] if child_count else 0.0
    )
    state["semantic_child_valid_output_rate"] = valid_output_count / denominator
    state["semantic_child_chunk_contract_rate"] = chunk_contract_count / denominator
    state["semantic_child_record_contract_rate"] = record_contract_count / denominator
    state["semantic_records_per_subcall"] = len(queried_ids) / denominator
    state["semantic_record_coverage"] = (
        len(unique_queried) / len(gold_labels) if gold_labels else 0.0
    )
    state["semantic_duplicate_coverage"] = (
        (len(queried_ids) - len(unique_queried)) / len(queried_ids)
        if queried_ids
        else 0.0
    )
    state["semantic_verified_full_chunk_coverage"] = (
        len(verified_full_chunks) / len(chunk_labels) if chunk_labels else 0.0
    )
    state["semantic_child_prediction_count"] = float(prediction_count)
    state["semantic_child_expected_output_count"] = float(expected_count)
    state["semantic_child_input_rejections"] = rejection_counts


def _extract_natural_semantic_label(
    response_text: str, label_space: list[str]
) -> str | None:
    candidate = _strip_answer_prefix(response_text).strip()
    normalized = _normalize_semantic_label(candidate, label_space)
    if normalized is not None:
        return normalized

    parsed = _extract_json_object(candidate)
    if parsed:
        parsed_labels = {
            label
            for value in parsed.values()
            if (label := _normalize_semantic_label(value, label_space)) is not None
        }
        if len(parsed_labels) == 1:
            return next(iter(parsed_labels))

    normalized_labels = {
        _normalize_text(label).casefold(): label for label in label_space
    }
    label_pattern = "|".join(
        re.escape(label) for label in sorted(normalized_labels, key=len, reverse=True)
    )
    if not label_pattern:
        return None
    conclusion_pattern = re.compile(
        rf"\b(?:overall|majority|mostly|predominantly|conclusion|answer|label)\b"
        rf"[^.\n:;]{{0,32}}\b(?P<label>{label_pattern})\b",
        flags=re.IGNORECASE,
    )
    conclusion_matches = list(conclusion_pattern.finditer(_normalize_text(candidate)))
    if conclusion_matches:
        return normalized_labels[
            _normalize_text(conclusion_matches[-1].group("label")).casefold()
        ]

    mentioned = {
        normalized_label: label
        for normalized_label, label in normalized_labels.items()
        if re.search(
            rf"\b{re.escape(normalized_label)}\b",
            _normalize_text(candidate),
            flags=re.IGNORECASE,
        )
    }
    return next(iter(mentioned.values())) if len(mentioned) == 1 else None


def _annotate_semantic_child_segments_v5(state: vf.State, info: dict[str, Any]) -> None:
    metadata = _semantic_delegation_metadata(info)
    assert metadata is not None
    labels_by_hash = {
        str(text_hash): str(label)
        for text_hash, label in (
            metadata.get("record_labels_by_text_hash") or {}
        ).items()
    }
    chunks_by_hash = {
        str(text_hash): str(chunk_id)
        for text_hash, chunk_id in (
            metadata.get("record_chunks_by_text_hash") or {}
        ).items()
    }
    label_space = [str(label) for label in metadata.get("label_space") or []]
    chunk_labels = metadata.get("chunk_labels") or {}
    segments = state.get("rlm_segments")
    if not isinstance(segments, list):
        return

    child_count = 0
    matched_call_count = 0
    parsed_answer_count = 0
    local_signal_count = 0
    local_correct_count = 0
    total_matched_records = 0
    all_matched_hashes: set[str] = set()
    covered_chunks: set[str] = set()
    fallback_reasons: dict[str, int] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        if (
            segment.get("kind") != "plain_query"
            and segment.get("train_scope") != "llm_subcall"
        ):
            continue

        child_count += 1
        segment["semantic_child_segment"] = True
        segment["semantic_terminal_fallback_eligible"] = True
        matched_hashes = list(
            dict.fromkeys(
                str(text_hash)
                for text_hash in (
                    segment.get("semantic_prompt_matched_text_hashes") or []
                )
                if str(text_hash) in labels_by_hash
            )
        )
        matched_chunks = sorted(
            {
                chunks_by_hash[text_hash]
                for text_hash in matched_hashes
                if text_hash in chunks_by_hash
            }
        )
        segment["semantic_recognized_record_count"] = len(matched_hashes)
        segment["semantic_recognized_chunk_ids"] = matched_chunks
        segment["semantic_primary_chunk_id"] = (
            matched_chunks[0] if len(matched_chunks) == 1 else None
        )
        segment["semantic_input_verified"] = bool(matched_hashes)
        total_matched_records += len(matched_hashes)
        all_matched_hashes.update(matched_hashes)
        covered_chunks.update(matched_chunks)
        if matched_hashes:
            matched_call_count += 1

        prediction = _extract_natural_semantic_label(
            str(segment.get("response_text") or ""), label_space
        )
        if prediction is not None:
            parsed_answer_count += 1

        label_counts = {label: 0 for label in label_space}
        for text_hash in matched_hashes:
            label = labels_by_hash[text_hash]
            if label in label_counts:
                label_counts[label] += 1
        ordered_counts = sorted(label_counts.values(), reverse=True)
        target = None
        if (
            matched_hashes
            and len(ordered_counts) >= 2
            and ordered_counts[0] > ordered_counts[1]
        ):
            target = max(label_counts, key=label_counts.get)

        if not matched_hashes:
            fallback_reason = "no_matched_records"
        elif target is None:
            fallback_reason = "visible_label_tie"
        elif prediction is None:
            fallback_reason = "ambiguous_answer"
        else:
            fallback_reason = None

        if fallback_reason is not None:
            segment["semantic_local_signal"] = False
            segment["semantic_local_schema_valid"] = prediction is not None
            segment["semantic_local_fallback_reason"] = fallback_reason
            fallback_reasons[fallback_reason] = (
                fallback_reasons.get(fallback_reason, 0) + 1
            )
            continue

        accuracy = float(prediction == target)
        local_signal_count += 1
        local_correct_count += int(accuracy)
        segment.update(
            {
                "semantic_local_signal": True,
                "semantic_local_contract": "natural_majority",
                "semantic_local_accuracy": accuracy,
                "semantic_local_coverage": 1.0,
                "semantic_local_schema_valid": True,
                "semantic_local_advantage": 2.0 * accuracy - 1.0,
                "semantic_local_exact": accuracy,
                "semantic_local_prediction_count": 1,
            }
        )

    denominator = child_count or 1
    signal_denominator = local_signal_count or 1
    state["semantic_child_scored_segments"] = float(local_signal_count)
    state["semantic_child_local_accuracy"] = local_correct_count / signal_denominator
    state["semantic_child_local_coverage"] = local_signal_count / denominator
    state["semantic_child_exact_rate"] = local_correct_count / signal_denominator
    state["semantic_child_local_signal_rate"] = local_signal_count / denominator
    state["semantic_child_terminal_fallback_rate"] = (
        child_count - local_signal_count
    ) / denominator
    state["semantic_child_input_match_rate"] = matched_call_count / denominator
    state["semantic_child_natural_answer_parse_rate"] = (
        parsed_answer_count / denominator
    )
    state["semantic_child_verified_input_rate"] = matched_call_count / denominator
    state["semantic_child_invalid_input_rate"] = (
        child_count - matched_call_count
    ) / denominator
    state["semantic_child_valid_output_rate"] = parsed_answer_count / denominator
    state["semantic_child_chunk_contract_rate"] = 0.0
    state["semantic_child_record_contract_rate"] = 0.0
    state["semantic_records_per_subcall"] = total_matched_records / denominator
    state["semantic_record_coverage"] = (
        len(all_matched_hashes) / len(labels_by_hash) if labels_by_hash else 0.0
    )
    state["semantic_duplicate_coverage"] = 0.0
    state["semantic_verified_full_chunk_coverage"] = 0.0
    state["semantic_section_diversity"] = (
        len(covered_chunks) / len(chunk_labels) if chunk_labels else 0.0
    )
    state["semantic_child_input_rejections"] = fallback_reasons


def _annotate_semantic_child_segments_v6(state: vf.State, info: dict[str, Any]) -> None:
    metadata = _semantic_delegation_metadata(info)
    assert metadata is not None
    labels_by_hash = {
        str(text_hash): str(label)
        for text_hash, label in (
            metadata.get("record_labels_by_text_hash") or {}
        ).items()
    }
    packets_by_hash = {
        str(text_hash): str(packet_id)
        for text_hash, packet_id in (
            metadata.get("record_packets_by_text_hash") or {}
        ).items()
    }
    sections_by_hash = {
        str(text_hash): str(section_id)
        for text_hash, section_id in (
            metadata.get("record_chunks_by_text_hash") or {}
        ).items()
    }
    packet_labels = {
        str(key): str(value)
        for key, value in (metadata.get("packet_labels") or {}).items()
    }
    packet_records: dict[str, set[str]] = {}
    for text_hash, packet_id in packets_by_hash.items():
        packet_records.setdefault(packet_id, set()).add(text_hash)
    label_space = [str(label) for label in metadata.get("label_space") or []]
    chunk_labels = metadata.get("chunk_labels") or {}
    segments = state.get("rlm_segments")
    if not isinstance(segments, list):
        return

    child_count = 0
    matched_call_count = 0
    complete_packet_count = 0
    parsed_answer_count = 0
    local_signal_count = 0
    local_correct_count = 0
    total_matched_records = 0
    all_matched_hashes: set[str] = set()
    complete_packets: set[str] = set()
    covered_sections: set[str] = set()
    fallback_reasons: dict[str, int] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        if (
            segment.get("kind") != "plain_query"
            and segment.get("train_scope") != "llm_subcall"
        ):
            continue

        child_count += 1
        segment["semantic_child_segment"] = True
        segment["semantic_terminal_fallback_eligible"] = True
        matched_hashes = list(
            dict.fromkeys(
                str(text_hash)
                for text_hash in (
                    segment.get("semantic_prompt_matched_text_hashes") or []
                )
                if str(text_hash) in labels_by_hash
            )
        )
        matched_packets = sorted(
            {packets_by_hash[text_hash] for text_hash in matched_hashes}
        )
        matched_sections = sorted(
            {sections_by_hash[text_hash] for text_hash in matched_hashes}
        )
        complete_packet_id = None
        if len(matched_packets) == 1 and set(matched_hashes) == packet_records.get(
            matched_packets[0], set()
        ):
            complete_packet_id = matched_packets[0]
            complete_packets.add(complete_packet_id)
            complete_packet_count += 1
        segment["semantic_recognized_record_count"] = len(matched_hashes)
        segment["semantic_recognized_packet_ids"] = matched_packets
        segment["semantic_recognized_chunk_ids"] = matched_sections
        segment["semantic_primary_packet_id"] = complete_packet_id
        segment["semantic_primary_chunk_id"] = (
            matched_sections[0]
            if complete_packet_id and len(matched_sections) == 1
            else None
        )
        segment["semantic_complete_packet"] = complete_packet_id is not None
        segment["semantic_input_verified"] = complete_packet_id is not None
        total_matched_records += len(matched_hashes)
        all_matched_hashes.update(matched_hashes)
        covered_sections.update(matched_sections)
        if matched_hashes:
            matched_call_count += 1

        prediction = _extract_natural_semantic_label(
            str(segment.get("response_text") or ""), label_space
        )
        if prediction is not None:
            parsed_answer_count += 1

        if not matched_hashes:
            fallback_reason = "no_matched_records"
        elif len(matched_packets) != 1:
            fallback_reason = "mixed_packets"
        elif complete_packet_id is None:
            fallback_reason = "partial_packet"
        elif prediction is None:
            fallback_reason = "ambiguous_answer"
        else:
            fallback_reason = None

        if fallback_reason is not None:
            segment["semantic_local_signal"] = False
            segment["semantic_local_schema_valid"] = prediction is not None
            segment["semantic_local_fallback_reason"] = fallback_reason
            fallback_reasons[fallback_reason] = (
                fallback_reasons.get(fallback_reason, 0) + 1
            )
            continue

        accuracy = float(prediction == packet_labels[complete_packet_id])
        local_signal_count += 1
        local_correct_count += int(accuracy)
        segment.update(
            {
                "semantic_local_signal": True,
                "semantic_local_contract": "natural_packet_majority",
                "semantic_local_accuracy": accuracy,
                "semantic_local_coverage": 1.0,
                "semantic_local_schema_valid": True,
                "semantic_local_advantage": 2.0 * accuracy - 1.0,
                "semantic_local_exact": accuracy,
                "semantic_local_prediction_count": 1,
            }
        )

    denominator = child_count or 1
    signal_denominator = local_signal_count or 1
    state["semantic_child_scored_segments"] = float(local_signal_count)
    state["semantic_child_local_accuracy"] = local_correct_count / signal_denominator
    state["semantic_child_local_coverage"] = local_signal_count / denominator
    state["semantic_child_exact_rate"] = local_correct_count / signal_denominator
    state["semantic_child_local_signal_rate"] = local_signal_count / denominator
    state["semantic_child_terminal_fallback_rate"] = (
        child_count - local_signal_count
    ) / denominator
    state["semantic_child_input_match_rate"] = matched_call_count / denominator
    state["semantic_child_natural_answer_parse_rate"] = (
        parsed_answer_count / denominator
    )
    state["semantic_child_verified_input_rate"] = complete_packet_count / denominator
    state["semantic_child_invalid_input_rate"] = (
        child_count - complete_packet_count
    ) / denominator
    state["semantic_child_valid_output_rate"] = parsed_answer_count / denominator
    state["semantic_complete_packet_call_rate"] = complete_packet_count / denominator
    state["semantic_records_per_subcall"] = total_matched_records / denominator
    state["semantic_record_coverage"] = (
        len(all_matched_hashes) / len(labels_by_hash) if labels_by_hash else 0.0
    )
    state["semantic_packet_coverage"] = (
        len(complete_packets) / len(packet_records) if packet_records else 0.0
    )
    state["semantic_section_diversity"] = (
        len(covered_sections) / len(chunk_labels) if chunk_labels else 0.0
    )
    state["semantic_child_chunk_contract_rate"] = 0.0
    state["semantic_child_record_contract_rate"] = 0.0
    state["semantic_verified_full_chunk_coverage"] = 0.0
    state["semantic_duplicate_coverage"] = 0.0
    state["semantic_child_input_rejections"] = fallback_reasons


def _annotate_semantic_child_segments(
    state: vf.State, info: dict[str, Any] | None
) -> None:
    metadata = _semantic_delegation_metadata(info)
    if metadata is None:
        return
    if metadata.get("derived_dataset") == "oolong_semantic_delegation_v6":
        _annotate_semantic_child_segments_v6(state, info or {})
    elif metadata.get("derived_dataset") == "oolong_semantic_delegation_v5":
        _annotate_semantic_child_segments_v5(state, info or {})
    elif metadata.get("derived_dataset") == "oolong_semantic_delegation_v4":
        _annotate_semantic_child_segments_v4(state, info or {})
    else:
        _annotate_semantic_child_segments_v3(state, info)
    write_live_trace(state, event="semantic_scored")


def _record_semantic_delegation_metrics(
    state: vf.State, score: SemanticDelegationScore
) -> None:
    state["semantic_task_score"] = score.task_score
    state["semantic_chunk_accuracy"] = score.chunk_accuracy
    state["semantic_progress"] = score.progress
    state["semantic_exact"] = score.exact
    state["semantic_schema_valid"] = score.schema_valid
    state["semantic_extra_keys"] = float(score.extra_keys)
    state["correctness_metric_value"] = score.exact


def _strip_answer_prefix(text: str) -> str:
    text = _normalize_text(text)
    match = re.match(
        r"^(?:count|label|user|month|answer)\s*:\s*(.+)$", text, flags=re.IGNORECASE
    )
    if match:
        return _normalize_text(match.group(1))
    return text


def _extract_numeric_answer(text: str) -> str | None:
    stripped = _strip_answer_prefix(text)
    direct = _canonicalize_number(stripped)
    if direct is not None:
        return direct
    matches = re.findall(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", stripped.replace(",", "")
    )
    if len(matches) == 1:
        return _canonicalize_number(matches[0])
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = _normalize_text(text)
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _canonical_histogram(value: dict[str, Any]) -> dict[str, int] | None:
    result: dict[str, int] = {}
    for key, raw_count in value.items():
        number = _canonicalize_number(str(raw_count))
        if number is None:
            return None
        try:
            decimal = Decimal(number)
        except InvalidOperation:
            return None
        if decimal != decimal.to_integral_value():
            return None
        result[_normalize_text(key)] = int(decimal)
    return result


def _score_oolong_semantic_aggregation(
    predicted_answer: str,
    expected_answers: list[str],
    *,
    answer_type: str,
) -> tuple[float, str, str | None]:
    if answer_type == "ANSWER_TYPE.JSON":
        predicted_json = _extract_json_object(predicted_answer)
        if predicted_json is None:
            return 0.0, "[oolong_semantic_json_mismatch]", "invalid_json"
        predicted_histogram = _canonical_histogram(predicted_json)
        if predicted_histogram is None:
            return 0.0, "[oolong_semantic_json_mismatch]", "invalid_histogram"
        for expected_answer in expected_answers:
            expected_json = _extract_json_object(expected_answer)
            if expected_json is None:
                continue
            expected_histogram = _canonical_histogram(expected_json)
            if (
                expected_histogram is not None
                and predicted_histogram == expected_histogram
            ):
                return 1.0, "[oolong_semantic_exact]", None
        return 0.0, "[oolong_semantic_json_mismatch]", None

    if answer_type == "ANSWER_TYPE.NUMERIC":
        predicted_number = _extract_numeric_answer(predicted_answer)
        if predicted_number is None:
            return 0.0, "[oolong_semantic_numeric_mismatch]", "invalid_numeric"
        for expected_answer in expected_answers:
            expected_number = _extract_numeric_answer(expected_answer)
            if expected_number is not None and predicted_number == expected_number:
                return 1.0, "[oolong_semantic_exact]", None
        return 0.0, "[oolong_semantic_numeric_mismatch]", None

    predicted_label = _strip_answer_prefix(predicted_answer).casefold()
    for expected_answer in expected_answers:
        if predicted_label == _strip_answer_prefix(expected_answer).casefold():
            return 1.0, "[oolong_semantic_exact]", None
    return 0.0, "[oolong_semantic_label_mismatch]", None


def _record_oolong_semantic_metrics(state: vf.State, score: float) -> None:
    state["oolong_semantic_score"] = float(score)
    state["oolong_semantic_exact"] = 1.0 if score >= 1.0 else 0.0


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        joined = "".join(parts).strip()
        if joined:
            return joined

    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ""


def _load_google_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Vertex Gemini judging requires the google-genai package. "
            "Install environments/rlm_rlvr with google-genai[aiohttp]>=1.51.0."
        ) from exc
    return genai, types


def _normalise_vertex_model_name(model_name: str) -> str:
    if model_name.startswith("google/"):
        model_name = model_name.removeprefix("google/")
    if model_name == "gemini-3-flash":
        return "gemini-3-flash-preview"
    return model_name


def _exception_status_code(exc: BaseException) -> int | None:
    for attr_name in ("status_code", "status", "code"):
        value = getattr(exc, attr_name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    response = getattr(exc, "response", None)
    if response is not None:
        for attr_name in ("status_code", "status"):
            value = getattr(response, attr_name, None)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
    return None


def _is_retryable_judge_exception(exc: BaseException) -> bool:
    status_code = _exception_status_code(exc)
    if status_code == 429 or status_code in {500, 502, 503, 504}:
        return True

    error_text = f"{type(exc).__name__}: {exc}".upper()
    return any(
        marker in error_text
        for marker in (
            "429",
            "RATE_LIMIT",
            "RESOURCE_EXHAUSTED",
            "TOO MANY REQUESTS",
            "UNAVAILABLE",
            "SERVICE UNAVAILABLE",
            "DEADLINE_EXCEEDED",
            "INTERNAL",
            "ACCESS_TOKEN_TYPE_UNSUPPORTED",
        )
    )


async def _sleep_before_judge_retry(attempt: int) -> None:
    delay = min(_JUDGE_RETRY_MAX_SECONDS, _JUDGE_RETRY_BASE_SECONDS * (2**attempt))
    jitter = random.uniform(0.0, min(1.0, delay * 0.25))
    await asyncio.sleep(delay + jitter)


async def _call_judge_with_retries(request: Callable[[], Awaitable[Any]]) -> Any:
    for attempt in range(_JUDGE_RETRY_MAX_ATTEMPTS):
        try:
            return await request()
        except Exception as exc:
            if (
                attempt == _JUDGE_RETRY_MAX_ATTEMPTS - 1
                or not _is_retryable_judge_exception(exc)
            ):
                raise
            await _sleep_before_judge_retry(attempt)

    raise RuntimeError("unreachable judge retry state")


def _vertex_client_kwargs(*, project: str, location: str, types: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "vertexai": True,
        "project": project,
        "location": location,
    }
    http_options_type = getattr(types, "HttpOptions", None)
    if http_options_type is not None:
        kwargs["http_options"] = http_options_type(api_version="v1")
    return kwargs


async def _close_vertex_client(client: Any) -> None:
    aio_client = getattr(client, "aio", None)
    aclose = getattr(aio_client, "aclose", None)
    if callable(aclose):
        await aclose()
        return

    close = getattr(client, "close", None)
    if callable(close):
        close()


class _VertexJudgeClientFactory:
    def __init__(self, *, project: str, location: str) -> None:
        self.project = project
        self.location = location

    async def generate_content(self, **kwargs: Any) -> Any:
        genai, types = _load_google_genai()
        client = genai.Client(
            **_vertex_client_kwargs(
                project=self.project, location=self.location, types=types
            )
        )
        try:
            return await client.aio.models.generate_content(**kwargs)
        finally:
            await _close_vertex_client(client)


async def _call_vertex_generate_content(judge_client: Any, **kwargs: Any) -> Any:
    generate_content = getattr(judge_client, "generate_content", None)
    if callable(generate_content):
        return await generate_content(**kwargs)
    return await judge_client.aio.models.generate_content(**kwargs)


def _record_judge_payload(
    state: vf.State,
    *,
    predicted_answer: str,
    expected_answers: list[str],
    score: float,
    raw_response: str,
    parse_error: str | None,
) -> None:
    state["judge_predicted_answer"] = predicted_answer
    state["judge_expected_answers"] = list(expected_answers)
    state["judge_score"] = score
    state["judge_raw_response"] = raw_response
    state["judge_parse_error"] = parse_error

    trajectory = state.get("trajectory") or []
    if not trajectory:
        return

    extras = trajectory[-1].setdefault("extras", {})
    rlm_debug = extras.setdefault("rlm_debug", {})
    rlm_debug.update(
        {
            "predicted_answer": predicted_answer,
            "expected_answers": list(expected_answers),
            "judge_score": score,
            "judge_raw_response": raw_response,
            "judge_parse_error": parse_error,
        }
    )


def _missing_formal_final_at_max_turn(state: vf.State) -> bool:
    if _state_bool_with_debug_fallback(state, "hit_max_turn_without_final"):
        return True
    if _state_bool_with_debug_fallback(state, "missing_final"):
        return True
    return (
        state.get("final_answer") is None
        and state.get("stop_condition") == "max_turns_reached"
    )


def _max_turn_penalty_from_state(
    state: vf.State,
    *,
    correctness: float,
    max_turn_penalty_enabled: bool,
    max_turn_penalty: float,
) -> float:
    if not max_turn_penalty_enabled or correctness <= 0.0:
        return 0.0
    if _state_bool_with_debug_fallback(state, "finalized_on_forced_prompt"):
        return max(0.0, max_turn_penalty)
    return 0.0


def _segment_token_length(
    segment: dict[str, Any], *, ids_key: str, count_key: str | None = None
) -> int:
    if count_key is not None:
        count_value = segment.get(count_key)
        if count_value not in (None, ""):
            return int(count_value)
    token_ids = segment.get(ids_key)
    if isinstance(token_ids, list):
        return len(token_ids)
    return 0


def _segment_rollout_token_totals(state: vf.State) -> tuple[int, int]:
    breakdown = _segment_rollout_token_breakdown(state)
    return breakdown.prompt_tokens, breakdown.completion_tokens


def _segment_rollout_token_breakdown(state: vf.State) -> TokenBreakdown:
    segments = state.get("rlm_segments")
    if isinstance(segments, list):
        prompt_tokens = 0
        completion_tokens = 0
        trainable_tokens = 0
        rlm_turn_tokens = 0
        plain_subcall_tokens = 0
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment_prompt_tokens = _segment_token_length(
                segment,
                ids_key="prompt_ids",
                count_key="prompt_token_count",
            )
            segment_completion_tokens = _segment_token_length(
                segment,
                ids_key="completion_ids",
                count_key="completion_token_count",
            )
            segment_total_tokens = segment_prompt_tokens + segment_completion_tokens
            prompt_tokens += segment_prompt_tokens
            completion_tokens += segment_completion_tokens
            if bool(segment.get("is_trainable_rlm_turn", False)):
                trainable_tokens += segment_total_tokens
            if (
                segment.get("kind") == "plain_query"
                or segment.get("train_scope") == "llm_subcall"
            ):
                plain_subcall_tokens += segment_total_tokens
            else:
                rlm_turn_tokens += segment_total_tokens
        return TokenBreakdown(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            trainable_tokens=trainable_tokens,
            rlm_turn_tokens=rlm_turn_tokens,
            plain_subcall_tokens=plain_subcall_tokens,
        )

    prompt_tokens = int(float(state.get("total_prompt_tokens", 0.0) or 0.0))
    completion_tokens = int(float(state.get("total_completion_tokens", 0.0) or 0.0))
    return TokenBreakdown(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        trainable_tokens=prompt_tokens + completion_tokens,
        rlm_turn_tokens=prompt_tokens + completion_tokens,
        plain_subcall_tokens=0,
    )


def _efficiency_penalty_from_state(state: vf.State) -> tuple[float, int, int, int]:
    breakdown = _segment_rollout_token_breakdown(state)
    penalty_coef = float(state.get("efficiency_penalty_coef", 0.0) or 0.0)
    if penalty_coef <= 0.0 or breakdown.total_tokens <= 0:
        return (
            0.0,
            breakdown.prompt_tokens,
            breakdown.completion_tokens,
            breakdown.total_tokens,
        )
    penalty = penalty_coef * (
        float(breakdown.total_tokens) / _EFFICIENCY_PENALTY_PER_1K_TOKENS
    )
    return (
        penalty,
        breakdown.prompt_tokens,
        breakdown.completion_tokens,
        breakdown.total_tokens,
    )


def _record_reward_breakdown(
    state: vf.State,
    *,
    correctness: float,
    efficiency_penalty: float,
    total_reward: float,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    trainable_tokens: int | None = None,
    rlm_turn_tokens: int | None = None,
    plain_subcall_tokens: int | None = None,
    weighted_tokens: float | None = None,
    group_solve_rate: float | None = None,
    adaptive_beta: float | None = None,
    adaptive_normalized_cost: float | None = None,
    adaptive_cost_penalty: float | None = None,
    incorrect_cost_penalty: float = 0.0,
    max_turn_penalty: float = 0.0,
) -> None:
    state["reward_correctness"] = correctness
    state["reward_efficiency_penalty"] = efficiency_penalty
    state["reward_incorrect_cost_penalty"] = incorrect_cost_penalty
    state["reward_max_turn_penalty"] = max_turn_penalty
    state["reward_total"] = total_reward
    state["cost_prompt_tokens"] = float(prompt_tokens)
    state["cost_completion_tokens"] = float(completion_tokens)
    state["cost_total_tokens"] = float(total_tokens)
    if trainable_tokens is not None:
        state["cost_trainable_tokens"] = float(trainable_tokens)
    if rlm_turn_tokens is not None:
        state["cost_rlm_turn_tokens"] = float(rlm_turn_tokens)
    if plain_subcall_tokens is not None:
        state["cost_plain_subcall_tokens"] = float(plain_subcall_tokens)
    if weighted_tokens is not None:
        state["cost_weighted_tokens"] = float(weighted_tokens)
    if group_solve_rate is not None:
        state["reward_group_solve_rate"] = group_solve_rate
    if adaptive_beta is not None:
        state["reward_adaptive_beta"] = adaptive_beta
    if adaptive_normalized_cost is not None:
        state["reward_adaptive_normalized_cost"] = adaptive_normalized_cost
    if adaptive_cost_penalty is not None:
        state["reward_adaptive_cost_penalty"] = adaptive_cost_penalty

    trajectory = state.get("trajectory") or []
    if not trajectory:
        return

    debug_payload = {
        "reward_correctness": correctness,
        "reward_efficiency_penalty": efficiency_penalty,
        "reward_incorrect_cost_penalty": incorrect_cost_penalty,
        "reward_max_turn_penalty": max_turn_penalty,
        "reward_total": total_reward,
        "cost_prompt_tokens": prompt_tokens,
        "cost_completion_tokens": completion_tokens,
        "cost_total_tokens": total_tokens,
    }
    if trainable_tokens is not None:
        debug_payload["cost_trainable_tokens"] = trainable_tokens
    if rlm_turn_tokens is not None:
        debug_payload["cost_rlm_turn_tokens"] = rlm_turn_tokens
    if plain_subcall_tokens is not None:
        debug_payload["cost_plain_subcall_tokens"] = plain_subcall_tokens
    if weighted_tokens is not None:
        debug_payload["cost_weighted_tokens"] = weighted_tokens
    if group_solve_rate is not None:
        debug_payload["reward_group_solve_rate"] = group_solve_rate
    if adaptive_beta is not None:
        debug_payload["reward_adaptive_beta"] = adaptive_beta
    if adaptive_normalized_cost is not None:
        debug_payload["reward_adaptive_normalized_cost"] = adaptive_normalized_cost
    if adaptive_cost_penalty is not None:
        debug_payload["reward_adaptive_cost_penalty"] = adaptive_cost_penalty

    extras = trajectory[-1].setdefault("extras", {})
    rlm_debug = extras.setdefault("rlm_debug", {})
    rlm_debug.update(debug_payload)


async def _call_binary_judge(
    judge_client: AsyncOpenAI,
    *,
    judge_model: str,
    judge_prompt: str,
) -> tuple[float, str, str | None]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_prompt},
    ]

    last_raw_response = ""
    for attempt in range(2):
        judge_response = await _call_judge_with_retries(
            lambda: judge_client.chat.completions.create(
                model=judge_model,
                messages=messages,
                temperature=0,
                max_tokens=8,
                extra_body={"reasoning": {"enabled": False}},
            )
        )
        raw_response = _extract_message_text(judge_response.choices[0].message)
        last_raw_response = raw_response
        try:
            return _parse_binary_judge_score(raw_response), raw_response, None
        except ValueError:
            if attempt == 1:
                break
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"{judge_prompt}\n\n"
                        f"Your previous response was invalid: {raw_response!r}\n"
                        "Return only 0 or 1."
                    ),
                },
            ]

    return 0.0, last_raw_response, "invalid_binary_score"


async def _call_vertex_binary_judge(
    judge_client: Any,
    *,
    judge_model: str,
    judge_prompt: str,
    thinking_level: str | None,
) -> tuple[float, str, str | None]:
    _, types = _load_google_genai()
    last_raw_response = ""
    for attempt in range(2):
        user_prompt = judge_prompt
        if attempt == 1:
            user_prompt = (
                f"{judge_prompt}\n\n"
                f"Your previous response was invalid: {last_raw_response!r}\n"
                "Return only 0 or 1."
            )
        config_kwargs: dict[str, Any] = {
            "system_instruction": JUDGE_SYSTEM_PROMPT,
            "temperature": 0,
            "max_output_tokens": _VERTEX_JUDGE_MAX_OUTPUT_TOKENS,
        }
        if thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level
            )
        response = await _call_judge_with_retries(
            lambda: _call_vertex_generate_content(
                judge_client,
                model=_normalise_vertex_model_name(judge_model),
                contents=user_prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        )
        raw_response = str(getattr(response, "text", "") or "").strip()
        last_raw_response = raw_response
        try:
            return _parse_binary_judge_score(raw_response), raw_response, None
        except ValueError:
            if attempt == 1:
                break

    return 0.0, last_raw_response, "invalid_binary_score"


async def _score_correctness(
    state: vf.State,
    completion,
    answer: str,
    info: dict[str, Any] | None,
    *,
    judge_provider: str,
    judge_client: Any,
    judge_model: str,
    judge_thinking_level: str | None,
    judge_candidate_max_chars: int,
    judge_question_max_chars: int,
    judge_expected_max_chars: int,
) -> CorrectnessResult:
    predicted_answer = _get_predicted_answer(state, completion).strip()
    expected_answers = _get_expected_answers(answer, info)
    question = str((info or {}).get("question", "")).strip()

    if not predicted_answer:
        result = CorrectnessResult(
            predicted_answer="",
            expected_answers=expected_answers,
            score=0.0,
            raw_response="0",
            parse_error=None,
        )
        _record_judge_payload(
            state,
            predicted_answer=result.predicted_answer,
            expected_answers=result.expected_answers,
            score=result.score,
            raw_response=result.raw_response,
            parse_error=result.parse_error,
        )
        return result

    if len(predicted_answer) > judge_candidate_max_chars:
        result = CorrectnessResult(
            predicted_answer=predicted_answer,
            expected_answers=expected_answers,
            score=0.0,
            raw_response="[candidate_too_large]",
            parse_error="candidate_too_large",
        )
        _record_judge_payload(
            state,
            predicted_answer=result.predicted_answer,
            expected_answers=result.expected_answers,
            score=result.score,
            raw_response=result.raw_response,
            parse_error=result.parse_error,
        )
        return result

    if _is_oolong_pairs_task(info):
        score, raw_response, stats = _score_oolong_pairs(
            predicted_answer, expected_answers
        )
        _record_oolong_pairs_metrics(state, stats)
        result = CorrectnessResult(
            predicted_answer=predicted_answer,
            expected_answers=expected_answers,
            score=score,
            raw_response=raw_response,
            parse_error=None,
        )
        _record_judge_payload(
            state,
            predicted_answer=result.predicted_answer,
            expected_answers=result.expected_answers,
            score=result.score,
            raw_response=result.raw_response,
            parse_error=result.parse_error,
        )
        return result

    if _semantic_delegation_metadata(info) is not None:
        _annotate_semantic_child_segments(state, info)
        semantic_score, raw_response, parse_error = _score_semantic_delegation_v3(
            predicted_answer, info
        )
        _record_semantic_delegation_metrics(state, semantic_score)
        result = CorrectnessResult(
            predicted_answer=predicted_answer,
            expected_answers=expected_answers,
            score=semantic_score.task_score,
            raw_response=raw_response,
            parse_error=parse_error,
        )
        _record_judge_payload(
            state,
            predicted_answer=result.predicted_answer,
            expected_answers=result.expected_answers,
            score=result.score,
            raw_response=result.raw_response,
            parse_error=result.parse_error,
        )
        return result

    if _is_oolong_semantic_aggregation_task(info):
        score, raw_response, parse_error = _score_oolong_semantic_aggregation(
            predicted_answer,
            expected_answers,
            answer_type=str((info or {}).get("answer_type") or ""),
        )
        _record_oolong_semantic_metrics(state, score)
        result = CorrectnessResult(
            predicted_answer=predicted_answer,
            expected_answers=expected_answers,
            score=score,
            raw_response=raw_response,
            parse_error=parse_error,
        )
        _record_judge_payload(
            state,
            predicted_answer=result.predicted_answer,
            expected_answers=result.expected_answers,
            score=result.score,
            raw_response=result.raw_response,
            parse_error=result.parse_error,
        )
        return result

    if _is_exact_match(predicted_answer, expected_answers):
        result = CorrectnessResult(
            predicted_answer=predicted_answer,
            expected_answers=expected_answers,
            score=1.0,
            raw_response="[exact_match]",
            parse_error=None,
        )
        _record_judge_payload(
            state,
            predicted_answer=result.predicted_answer,
            expected_answers=result.expected_answers,
            score=result.score,
            raw_response=result.raw_response,
            parse_error=result.parse_error,
        )
        return result

    judge_expected_answers = [
        _truncate_for_judge(str(expected_answer), max_chars=judge_expected_max_chars)
        for expected_answer in expected_answers
    ]
    judge_prompt = JUDGE_PROMPT.format(
        question=_truncate_for_judge(
            question or "(not provided)", max_chars=judge_question_max_chars
        ),
        expected_answers=_format_expected_answers(judge_expected_answers),
        predicted_answer=predicted_answer,
    )
    if judge_provider == "vertex":
        try:
            score, raw_response, parse_error = await _call_vertex_binary_judge(
                judge_client,
                judge_model=judge_model,
                judge_prompt=judge_prompt,
                thinking_level=judge_thinking_level,
            )
        except Exception as exc:
            if not _is_oversized_judge_exception(exc):
                raise
            score, raw_response, parse_error = (
                0.0,
                f"[vertex_oversized_input: {type(exc).__name__}]",
                "vertex_input_too_large",
            )
    else:
        score, raw_response, parse_error = await _call_binary_judge(
            judge_client,
            judge_model=judge_model,
            judge_prompt=judge_prompt,
        )

    result = CorrectnessResult(
        predicted_answer=predicted_answer,
        expected_answers=expected_answers,
        score=score,
        raw_response=raw_response,
        parse_error=parse_error,
    )
    _record_judge_payload(
        state,
        predicted_answer=result.predicted_answer,
        expected_answers=result.expected_answers,
        score=result.score,
        raw_response=result.raw_response,
        parse_error=result.parse_error,
    )
    return result


async def _score_correctness_with_protocol(
    state: vf.State,
    completion,
    answer: str,
    info: dict[str, Any] | None,
    *,
    judge_provider: str,
    judge_client: Any,
    judge_model: str,
    judge_thinking_level: str | None,
    judge_candidate_max_chars: int,
    judge_question_max_chars: int,
    judge_expected_max_chars: int,
    missing_final_at_max_turn_zero_reward: bool,
) -> CorrectnessResult:
    _annotate_semantic_child_segments(state, info)
    if missing_final_at_max_turn_zero_reward and _missing_formal_final_at_max_turn(
        state
    ):
        predicted_answer = _get_predicted_answer(state, completion).strip()
        result = CorrectnessResult(
            predicted_answer=predicted_answer,
            expected_answers=_get_expected_answers(answer, info),
            score=0.0,
            raw_response="[missing_final_at_max_turn]",
            parse_error=None,
        )
        _record_judge_payload(
            state,
            predicted_answer=result.predicted_answer,
            expected_answers=result.expected_answers,
            score=result.score,
            raw_response=result.raw_response,
            parse_error=result.parse_error,
        )
        return result

    return await _score_correctness(
        state,
        completion,
        answer,
        info,
        judge_provider=judge_provider,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_thinking_level=judge_thinking_level,
        judge_candidate_max_chars=judge_candidate_max_chars,
        judge_question_max_chars=judge_question_max_chars,
        judge_expected_max_chars=judge_expected_max_chars,
    )


def _adaptive_beta_for_solve_rate(
    *,
    solve_rate: float,
    beta_max: float,
    gamma: float,
    solve_rate_floor: float,
    beta_min: float = 0.0,
) -> float:
    if solve_rate <= solve_rate_floor:
        return beta_min
    if solve_rate_floor >= 1.0:
        return beta_max
    ramp = (solve_rate - solve_rate_floor) / (1.0 - solve_rate_floor)
    return beta_min + (beta_max - beta_min) * (ramp**gamma)


def _weighted_turn_token_cost(
    breakdown: TokenBreakdown,
    *,
    root_token_multiplier: float,
    plain_subcall_token_multiplier: float,
) -> float:
    return float(root_token_multiplier) * float(breakdown.rlm_turn_tokens) + float(
        plain_subcall_token_multiplier
    ) * float(breakdown.plain_subcall_tokens)


def _adaptive_cost_value(
    state: vf.State,
    *,
    cost_basis: str,
    root_token_multiplier: float,
    plain_subcall_token_multiplier: float,
) -> float:
    breakdown = _segment_rollout_token_breakdown(state)
    if cost_basis == "total_tokens":
        return float(breakdown.total_tokens)
    if cost_basis == "weighted_turn_tokens":
        return _weighted_turn_token_cost(
            breakdown,
            root_token_multiplier=root_token_multiplier,
            plain_subcall_token_multiplier=plain_subcall_token_multiplier,
        )
    else:
        raise ValueError(
            f"adaptive_efficiency_cost_basis must be one of {sorted(_VALID_ADAPTIVE_COST_BASES)}"
        )


def _min_max_normalized(value: float, *, min_value: float, max_value: float) -> float:
    span = max_value - min_value
    if span <= 0:
        return 0.0
    return (float(value) - float(min_value)) / float(span)


def _clip_reward(
    value: float, *, reward_clip_min: float, reward_clip_max: float
) -> float:
    return min(reward_clip_max, max(reward_clip_min, value))


def build_rubric(
    *,
    judge_model: str,
    judge_base_url: str,
    judge_api_key: str | None,
    judge_provider: str = "openai_compatible",
    judge_default_headers: dict[str, str] | None = None,
    judge_vertex_project: str | None = None,
    judge_vertex_location: str = "global",
    judge_thinking_level: str | None = "medium",
    efficiency_penalty_mode: str = "static_per_1k",
    adaptive_efficiency_beta_min: float = 0.0,
    adaptive_efficiency_beta_max: float = 0.05,
    adaptive_efficiency_gamma: float = 2.0,
    adaptive_efficiency_solve_rate_floor: float = 0.25,
    adaptive_efficiency_cost_basis: str = "total_tokens",
    efficiency_root_token_multiplier: float = 1.0,
    efficiency_plain_subcall_token_multiplier: float = 1.0,
    efficiency_penalty_applies_to: str = "correct_only",
    efficiency_tie_break_max: float = 0.05,
    reward_clip_min: float = 0.0,
    reward_clip_max: float = 1.0,
    max_turn_penalty_enabled: bool = False,
    max_turn_penalty: float = 0.25,
    missing_final_at_max_turn_zero_reward: bool = True,
    judge_candidate_max_chars: int = 8192,
    judge_question_max_chars: int = 32768,
    judge_expected_max_chars: int = 8192,
) -> vf.Rubric:
    if efficiency_penalty_mode not in _VALID_EFFICIENCY_PENALTY_MODES:
        raise ValueError(
            f"efficiency_penalty_mode must be one of {sorted(_VALID_EFFICIENCY_PENALTY_MODES)}"
        )
    if adaptive_efficiency_cost_basis not in _VALID_ADAPTIVE_COST_BASES:
        raise ValueError(
            f"adaptive_efficiency_cost_basis must be one of {sorted(_VALID_ADAPTIVE_COST_BASES)}"
        )
    if efficiency_penalty_applies_to not in _VALID_EFFICIENCY_PENALTY_SCOPES:
        raise ValueError(
            f"efficiency_penalty_applies_to must be one of {sorted(_VALID_EFFICIENCY_PENALTY_SCOPES)}"
        )
    if reward_clip_min > reward_clip_max:
        raise ValueError("reward_clip_min must be <= reward_clip_max")
    if adaptive_efficiency_beta_min < 0.0:
        raise ValueError("adaptive_efficiency_beta_min must be >= 0.0")
    if adaptive_efficiency_beta_max < adaptive_efficiency_beta_min:
        raise ValueError(
            "adaptive_efficiency_beta_max must be >= adaptive_efficiency_beta_min"
        )
    if adaptive_efficiency_gamma <= 0.0:
        raise ValueError("adaptive_efficiency_gamma must be > 0.0")
    if not 0.0 <= adaptive_efficiency_solve_rate_floor < 1.0:
        raise ValueError(
            "adaptive_efficiency_solve_rate_floor must be >= 0.0 and < 1.0"
        )
    if efficiency_root_token_multiplier < 0.0:
        raise ValueError("efficiency_root_token_multiplier must be >= 0.0")
    if efficiency_plain_subcall_token_multiplier < 0.0:
        raise ValueError("efficiency_plain_subcall_token_multiplier must be >= 0.0")
    if efficiency_tie_break_max < 0.0:
        raise ValueError("efficiency_tie_break_max must be >= 0.0")
    if max_turn_penalty < 0.0:
        raise ValueError("max_turn_penalty must be >= 0.0")
    if judge_candidate_max_chars < 1:
        raise ValueError("judge_candidate_max_chars must be >= 1")
    if judge_question_max_chars < 1:
        raise ValueError("judge_question_max_chars must be >= 1")
    if judge_expected_max_chars < 1:
        raise ValueError("judge_expected_max_chars must be >= 1")

    if judge_provider == "vertex":
        if not judge_vertex_project:
            raise ValueError(
                "judge_vertex_project is required when judge_provider='vertex'"
            )
        _load_google_genai()
        judge_client: Any = _VertexJudgeClientFactory(
            project=judge_vertex_project, location=judge_vertex_location
        )
    elif judge_provider == "openai_compatible":
        judge_client = AsyncOpenAI(
            base_url=judge_base_url,
            api_key=judge_api_key or "EMPTY",
            default_headers=judge_default_headers,
        )
    else:
        raise ValueError(
            "judge_provider must be one of ['openai_compatible', 'vertex']"
        )

    async def reward_fn(
        state: vf.State, completion, answer: str, info: dict[str, Any] | None
    ) -> float:
        correctness_result = await _score_correctness_with_protocol(
            state,
            completion,
            answer,
            info,
            judge_provider=judge_provider,
            judge_client=judge_client,
            judge_model=judge_model,
            judge_thinking_level=judge_thinking_level,
            judge_candidate_max_chars=judge_candidate_max_chars,
            judge_question_max_chars=judge_question_max_chars,
            judge_expected_max_chars=judge_expected_max_chars,
            missing_final_at_max_turn_zero_reward=missing_final_at_max_turn_zero_reward,
        )
        efficiency_penalty, prompt_tokens, completion_tokens, total_tokens = (
            _efficiency_penalty_from_state(state)
        )
        breakdown = _segment_rollout_token_breakdown(state)
        weighted_tokens = _weighted_turn_token_cost(
            breakdown,
            root_token_multiplier=efficiency_root_token_multiplier,
            plain_subcall_token_multiplier=efficiency_plain_subcall_token_multiplier,
        )
        terminal_penalty = _max_turn_penalty_from_state(
            state,
            correctness=correctness_result.score,
            max_turn_penalty_enabled=max_turn_penalty_enabled,
            max_turn_penalty=max_turn_penalty,
        )

        scoped_efficiency_penalty = (
            efficiency_penalty
            if (
                correctness_result.score > 0.0
                or efficiency_penalty_applies_to == "all_rollouts"
            )
            else 0.0
        )
        total_reward = _clip_reward(
            correctness_result.score - terminal_penalty - scoped_efficiency_penalty,
            reward_clip_min=reward_clip_min,
            reward_clip_max=reward_clip_max,
        )
        _record_reward_breakdown(
            state,
            correctness=correctness_result.score,
            efficiency_penalty=scoped_efficiency_penalty,
            total_reward=total_reward,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            trainable_tokens=breakdown.trainable_tokens,
            rlm_turn_tokens=breakdown.rlm_turn_tokens,
            plain_subcall_tokens=breakdown.plain_subcall_tokens,
            weighted_tokens=weighted_tokens,
            incorrect_cost_penalty=scoped_efficiency_penalty
            if correctness_result.score <= 0.0
            else 0.0,
            max_turn_penalty=terminal_penalty,
        )
        return total_reward

    async def adaptive_group_reward_fn(states: list[vf.State]) -> list[float]:
        correctness_results = await asyncio.gather(
            *(
                _score_correctness_with_protocol(
                    state,
                    state.get("completion", []),
                    str(state.get("answer", "")),
                    state.get("info", {}),
                    judge_provider=judge_provider,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    judge_thinking_level=judge_thinking_level,
                    judge_candidate_max_chars=judge_candidate_max_chars,
                    judge_question_max_chars=judge_question_max_chars,
                    judge_expected_max_chars=judge_expected_max_chars,
                    missing_final_at_max_turn_zero_reward=missing_final_at_max_turn_zero_reward,
                )
                for state in states
            )
        )
        task_scores = [float(result.score) for result in correctness_results]
        progress_scores = [
            float(state.get("semantic_progress", task_score))
            for state, task_score in zip(states, task_scores, strict=True)
        ]
        solve_rate = (
            sum(progress_scores) / len(progress_scores) if progress_scores else 0.0
        )
        beta = _adaptive_beta_for_solve_rate(
            solve_rate=solve_rate,
            beta_min=adaptive_efficiency_beta_min,
            beta_max=adaptive_efficiency_beta_max,
            gamma=adaptive_efficiency_gamma,
            solve_rate_floor=adaptive_efficiency_solve_rate_floor,
        )

        costs = [
            _adaptive_cost_value(
                state,
                cost_basis=adaptive_efficiency_cost_basis,
                root_token_multiplier=efficiency_root_token_multiplier,
                plain_subcall_token_multiplier=efficiency_plain_subcall_token_multiplier,
            )
            for state in states
        ]
        correct_costs = [
            cost
            for cost, progress in zip(costs, progress_scores, strict=True)
            if progress > 0.0
        ]
        if efficiency_penalty_applies_to == "all_rollouts":
            normalization_costs = costs
        else:
            normalization_costs = correct_costs
        min_cost = min(normalization_costs) if len(normalization_costs) >= 2 else 0
        max_cost = max(normalization_costs) if len(normalization_costs) >= 2 else 0

        rewards: list[float] = []
        for state, task_score, progress, cost in zip(
            states, task_scores, progress_scores, costs, strict=True
        ):
            breakdown = _segment_rollout_token_breakdown(state)
            weighted_tokens = _weighted_turn_token_cost(
                breakdown,
                root_token_multiplier=efficiency_root_token_multiplier,
                plain_subcall_token_multiplier=efficiency_plain_subcall_token_multiplier,
            )
            normalized_cost = 0.0
            applies_to_rollout = (
                progress > 0.0 or efficiency_penalty_applies_to == "all_rollouts"
            )
            if applies_to_rollout:
                normalized_cost = _min_max_normalized(
                    cost, min_value=min_cost, max_value=max_cost
                )
            adaptive_cost_penalty = (
                beta * normalized_cost if applies_to_rollout else 0.0
            )
            terminal_penalty = _max_turn_penalty_from_state(
                state,
                correctness=task_score,
                max_turn_penalty_enabled=max_turn_penalty_enabled,
                max_turn_penalty=max_turn_penalty,
            )
            total_reward = _clip_reward(
                task_score - terminal_penalty - adaptive_cost_penalty,
                reward_clip_min=reward_clip_min,
                reward_clip_max=reward_clip_max,
            )
            _record_reward_breakdown(
                state,
                correctness=task_score,
                efficiency_penalty=adaptive_cost_penalty,
                total_reward=total_reward,
                prompt_tokens=breakdown.prompt_tokens,
                completion_tokens=breakdown.completion_tokens,
                total_tokens=breakdown.total_tokens,
                trainable_tokens=breakdown.trainable_tokens,
                rlm_turn_tokens=breakdown.rlm_turn_tokens,
                plain_subcall_tokens=breakdown.plain_subcall_tokens,
                weighted_tokens=weighted_tokens,
                group_solve_rate=solve_rate,
                adaptive_beta=beta,
                adaptive_normalized_cost=normalized_cost,
                adaptive_cost_penalty=adaptive_cost_penalty,
                incorrect_cost_penalty=adaptive_cost_penalty
                if progress <= 0.0
                else 0.0,
                max_turn_penalty=terminal_penalty,
            )
            rewards.append(total_reward)
        return rewards

    async def accuracy_stratified_group_reward_fn(
        states: list[vf.State],
    ) -> list[float]:
        correctness_results = await asyncio.gather(
            *(
                _score_correctness_with_protocol(
                    state,
                    state.get("completion", []),
                    str(state.get("answer", "")),
                    state.get("info", {}),
                    judge_provider=judge_provider,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    judge_thinking_level=judge_thinking_level,
                    judge_candidate_max_chars=judge_candidate_max_chars,
                    judge_question_max_chars=judge_question_max_chars,
                    judge_expected_max_chars=judge_expected_max_chars,
                    missing_final_at_max_turn_zero_reward=missing_final_at_max_turn_zero_reward,
                )
                for state in states
            )
        )
        task_scores = [float(result.score) for result in correctness_results]
        costs = [
            _adaptive_cost_value(
                state,
                cost_basis=adaptive_efficiency_cost_basis,
                root_token_multiplier=efficiency_root_token_multiplier,
                plain_subcall_token_multiplier=efficiency_plain_subcall_token_multiplier,
            )
            for state in states
        ]
        strata: dict[float, list[int]] = {}
        for index, score in enumerate(task_scores):
            strata.setdefault(round(score, 8), []).append(index)
        unique_scores = sorted(strata)
        positive_gaps = [
            right - left
            for left, right in zip(unique_scores, unique_scores[1:])
            if right > left
        ]
        correctness_safe_max = (
            min(positive_gaps) / 2.0 if positive_gaps else efficiency_tie_break_max
        )
        penalty_max = min(efficiency_tie_break_max, correctness_safe_max)

        normalized_costs = [0.0] * len(states)
        stratum_sizes = [1] * len(states)
        for indices in strata.values():
            for index in indices:
                stratum_sizes[index] = len(indices)
            if len(indices) < 2:
                continue
            stratum_costs = [costs[index] for index in indices]
            min_cost = min(stratum_costs)
            max_cost = max(stratum_costs)
            for index in indices:
                normalized_costs[index] = _min_max_normalized(
                    costs[index], min_value=min_cost, max_value=max_cost
                )

        rewards: list[float] = []
        for index, (state, task_score, normalized_cost) in enumerate(
            zip(states, task_scores, normalized_costs, strict=True)
        ):
            breakdown = _segment_rollout_token_breakdown(state)
            weighted_tokens = _weighted_turn_token_cost(
                breakdown,
                root_token_multiplier=efficiency_root_token_multiplier,
                plain_subcall_token_multiplier=efficiency_plain_subcall_token_multiplier,
            )
            cost_penalty = penalty_max * normalized_cost
            terminal_penalty = _max_turn_penalty_from_state(
                state,
                correctness=task_score,
                max_turn_penalty_enabled=max_turn_penalty_enabled,
                max_turn_penalty=max_turn_penalty,
            )
            total_reward = _clip_reward(
                task_score - terminal_penalty - cost_penalty,
                reward_clip_min=reward_clip_min,
                reward_clip_max=reward_clip_max,
            )
            _record_reward_breakdown(
                state,
                correctness=task_score,
                efficiency_penalty=cost_penalty,
                total_reward=total_reward,
                prompt_tokens=breakdown.prompt_tokens,
                completion_tokens=breakdown.completion_tokens,
                total_tokens=breakdown.total_tokens,
                trainable_tokens=breakdown.trainable_tokens,
                rlm_turn_tokens=breakdown.rlm_turn_tokens,
                plain_subcall_tokens=breakdown.plain_subcall_tokens,
                weighted_tokens=weighted_tokens,
                adaptive_normalized_cost=normalized_cost,
                adaptive_cost_penalty=cost_penalty,
                incorrect_cost_penalty=cost_penalty if task_score <= 0.0 else 0.0,
                max_turn_penalty=terminal_penalty,
            )
            state["reward_accuracy_stratum_size"] = float(stratum_sizes[index])
            state["reward_accuracy_stratified_cost_penalty"] = cost_penalty
            state["reward_efficiency_tie_break_max"] = penalty_max
            rewards.append(total_reward)
        return rewards

    if efficiency_penalty_mode == "adaptive_group":
        return vf.Rubric(funcs=[adaptive_group_reward_fn])
    if efficiency_penalty_mode == "accuracy_stratified_group":
        return vf.Rubric(funcs=[accuracy_stratified_group_reward_fn])
    return vf.Rubric(funcs=[reward_fn])


async def correctness_metric(state: vf.State) -> float:
    return float(
        state.get("correctness_metric_value", state.get("reward_correctness", 0.0))
    )


async def semantic_progress_metric(state: vf.State) -> float:
    return float(state.get("semantic_progress", state.get("reward_correctness", 0.0)))


async def semantic_chunk_accuracy_metric(state: vf.State) -> float:
    return float(
        state.get("semantic_chunk_accuracy", state.get("reward_correctness", 0.0))
    )


async def semantic_exact_metric(state: vf.State) -> float:
    return float(state.get("semantic_exact", 0.0))


async def semantic_child_local_accuracy_metric(state: vf.State) -> float:
    return float(state.get("semantic_child_local_accuracy", 0.0))


async def semantic_child_local_coverage_metric(state: vf.State) -> float:
    return float(state.get("semantic_child_local_coverage", 0.0))


async def semantic_complete_packet_call_rate_metric(state: vf.State) -> float:
    return float(state.get("semantic_complete_packet_call_rate", 0.0))


async def semantic_packet_coverage_metric(state: vf.State) -> float:
    return float(state.get("semantic_packet_coverage", 0.0))


async def semantic_child_verified_input_rate_metric(state: vf.State) -> float:
    return float(state.get("semantic_child_verified_input_rate", 0.0))


async def semantic_child_invalid_input_rate_metric(state: vf.State) -> float:
    return float(state.get("semantic_child_invalid_input_rate", 0.0))


async def semantic_child_valid_output_rate_metric(state: vf.State) -> float:
    return float(state.get("semantic_child_valid_output_rate", 0.0))


async def semantic_child_local_signal_rate_metric(state: vf.State) -> float:
    return float(state.get("semantic_child_local_signal_rate", 0.0))


async def semantic_child_terminal_fallback_rate_metric(state: vf.State) -> float:
    return float(state.get("semantic_child_terminal_fallback_rate", 0.0))


async def semantic_child_input_match_rate_metric(state: vf.State) -> float:
    return float(state.get("semantic_child_input_match_rate", 0.0))


async def semantic_child_natural_answer_parse_rate_metric(state: vf.State) -> float:
    return float(state.get("semantic_child_natural_answer_parse_rate", 0.0))


async def semantic_section_diversity_metric(state: vf.State) -> float:
    return float(state.get("semantic_section_diversity", 0.0))


async def semantic_child_chunk_contract_rate_metric(state: vf.State) -> float:
    return float(state.get("semantic_child_chunk_contract_rate", 0.0))


async def semantic_child_record_contract_rate_metric(state: vf.State) -> float:
    return float(state.get("semantic_child_record_contract_rate", 0.0))


async def semantic_verified_full_chunk_coverage_metric(state: vf.State) -> float:
    return float(state.get("semantic_verified_full_chunk_coverage", 0.0))


async def semantic_records_per_subcall_metric(state: vf.State) -> float:
    return float(state.get("semantic_records_per_subcall", 0.0))


async def semantic_record_coverage_metric(state: vf.State) -> float:
    return float(state.get("semantic_record_coverage", 0.0))


async def semantic_duplicate_coverage_metric(state: vf.State) -> float:
    return float(state.get("semantic_duplicate_coverage", 0.0))


async def subcall_batch_rejection_rate_metric(state: vf.State) -> float:
    attempts = float(state.get("subcall_batch_attempts", 0.0))
    rejections = float(state.get("subcall_batch_rejections", 0.0))
    return rejections / attempts if attempts > 0.0 else 0.0


async def subcall_budget_exhausted_metric(state: vf.State) -> float:
    return 1.0 if state.get("subcall_budget_exhausted") else 0.0


async def judge_score_metric(state: vf.State) -> float:
    return float(state.get("judge_score", 0.0))


async def oolong_pairs_precision_metric(state: vf.State) -> float:
    return float(state.get("oolong_pairs_precision", 0.0))


async def oolong_pairs_recall_metric(state: vf.State) -> float:
    return float(state.get("oolong_pairs_recall", 0.0))


async def oolong_pairs_f1_metric(state: vf.State) -> float:
    return float(state.get("oolong_pairs_f1", 0.0))


async def oolong_semantic_score_metric(state: vf.State) -> float:
    return float(state.get("oolong_semantic_score", 0.0))


async def oolong_semantic_exact_metric(state: vf.State) -> float:
    return float(state.get("oolong_semantic_exact", 0.0))


async def efficiency_penalty_metric(state: vf.State) -> float:
    return float(state.get("reward_efficiency_penalty", 0.0))


async def incorrect_cost_penalty_metric(state: vf.State) -> float:
    return float(state.get("reward_incorrect_cost_penalty", 0.0))


async def max_turn_penalty_metric(state: vf.State) -> float:
    return float(state.get("reward_max_turn_penalty", 0.0))


async def cost_prompt_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_prompt_tokens", 0.0))


async def cost_completion_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_completion_tokens", 0.0))


async def cost_total_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_total_tokens", 0.0))


async def cost_trainable_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_trainable_tokens", 0.0))


async def cost_rlm_turn_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_rlm_turn_tokens", 0.0))


async def cost_plain_subcall_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_plain_subcall_tokens", 0.0))


async def cost_weighted_tokens_metric(state: vf.State) -> float:
    return float(state.get("cost_weighted_tokens", 0.0))


async def adaptive_group_solve_rate_metric(state: vf.State) -> float:
    return float(state.get("reward_group_solve_rate", 0.0))


async def adaptive_beta_metric(state: vf.State) -> float:
    return float(state.get("reward_adaptive_beta", 0.0))


async def adaptive_normalized_cost_metric(state: vf.State) -> float:
    return float(state.get("reward_adaptive_normalized_cost", 0.0))


async def adaptive_cost_penalty_metric(state: vf.State) -> float:
    return float(state.get("reward_adaptive_cost_penalty", 0.0))


async def accuracy_stratum_size_metric(state: vf.State) -> float:
    return float(state.get("reward_accuracy_stratum_size", 0.0))


async def accuracy_stratified_cost_penalty_metric(state: vf.State) -> float:
    return float(state.get("reward_accuracy_stratified_cost_penalty", 0.0))


async def used_repl_metric(state: vf.State) -> float:
    return 1.0 if state.get("used_repl") else 0.0


async def used_recursion_metric(state: vf.State) -> float:
    return 1.0 if state.get("used_recursion") else 0.0


async def used_llm_subcalls_metric(state: vf.State) -> float:
    return 1.0 if state.get("used_llm_subcalls") else 0.0


async def used_rlm_subcalls_metric(state: vf.State) -> float:
    return 1.0 if state.get("used_rlm_subcalls") else 0.0


async def num_subcalls_metric(state: vf.State) -> float:
    return float(state.get("num_subcalls", 0))


async def num_llm_subcalls_metric(state: vf.State) -> float:
    return float(state.get("num_llm_subcalls", 0))


async def num_rlm_subcalls_metric(state: vf.State) -> float:
    return float(state.get("num_rlm_subcalls", 0))


async def max_depth_metric(state: vf.State) -> float:
    return float(state.get("max_depth_reached", 0))


async def used_forced_finalize_prompt_metric(state: vf.State) -> float:
    return 1.0 if state.get("used_forced_finalize_prompt") else 0.0


async def hit_max_turn_without_final_metric(state: vf.State) -> float:
    return 1.0 if state.get("hit_max_turn_without_final") else 0.0


async def missing_final_metric(state: vf.State) -> float:
    return 1.0 if state.get("missing_final") else 0.0


async def finalized_before_forced_prompt_metric(state: vf.State) -> float:
    return 1.0 if state.get("finalized_before_forced_prompt") else 0.0


async def finalized_on_forced_prompt_metric(state: vf.State) -> float:
    return 1.0 if state.get("finalized_on_forced_prompt") else 0.0


def add_metrics(rubric: vf.Rubric) -> vf.Rubric:
    rubric.add_metric(correctness_metric)
    rubric.add_metric(semantic_progress_metric)
    rubric.add_metric(semantic_chunk_accuracy_metric)
    rubric.add_metric(semantic_exact_metric)
    rubric.add_metric(semantic_child_local_accuracy_metric)
    rubric.add_metric(semantic_child_local_coverage_metric)
    rubric.add_metric(semantic_complete_packet_call_rate_metric)
    rubric.add_metric(semantic_packet_coverage_metric)
    rubric.add_metric(semantic_child_verified_input_rate_metric)
    rubric.add_metric(semantic_child_invalid_input_rate_metric)
    rubric.add_metric(semantic_child_valid_output_rate_metric)
    rubric.add_metric(semantic_child_local_signal_rate_metric)
    rubric.add_metric(semantic_child_terminal_fallback_rate_metric)
    rubric.add_metric(semantic_child_input_match_rate_metric)
    rubric.add_metric(semantic_child_natural_answer_parse_rate_metric)
    rubric.add_metric(semantic_section_diversity_metric)
    rubric.add_metric(semantic_child_chunk_contract_rate_metric)
    rubric.add_metric(semantic_child_record_contract_rate_metric)
    rubric.add_metric(semantic_verified_full_chunk_coverage_metric)
    rubric.add_metric(semantic_records_per_subcall_metric)
    rubric.add_metric(semantic_record_coverage_metric)
    rubric.add_metric(semantic_duplicate_coverage_metric)
    rubric.add_metric(subcall_batch_rejection_rate_metric)
    rubric.add_metric(subcall_budget_exhausted_metric)
    rubric.add_metric(judge_score_metric)
    rubric.add_metric(oolong_pairs_precision_metric)
    rubric.add_metric(oolong_pairs_recall_metric)
    rubric.add_metric(oolong_pairs_f1_metric)
    rubric.add_metric(oolong_semantic_score_metric)
    rubric.add_metric(oolong_semantic_exact_metric)
    rubric.add_metric(efficiency_penalty_metric)
    rubric.add_metric(incorrect_cost_penalty_metric)
    rubric.add_metric(max_turn_penalty_metric)
    rubric.add_metric(cost_prompt_tokens_metric)
    rubric.add_metric(cost_completion_tokens_metric)
    rubric.add_metric(cost_total_tokens_metric)
    rubric.add_metric(cost_trainable_tokens_metric)
    rubric.add_metric(cost_rlm_turn_tokens_metric)
    rubric.add_metric(cost_plain_subcall_tokens_metric)
    rubric.add_metric(cost_weighted_tokens_metric)
    rubric.add_metric(adaptive_group_solve_rate_metric)
    rubric.add_metric(adaptive_beta_metric)
    rubric.add_metric(adaptive_normalized_cost_metric)
    rubric.add_metric(adaptive_cost_penalty_metric)
    rubric.add_metric(accuracy_stratum_size_metric)
    rubric.add_metric(accuracy_stratified_cost_penalty_metric)
    rubric.add_metric(used_repl_metric)
    rubric.add_metric(used_recursion_metric)
    rubric.add_metric(used_llm_subcalls_metric)
    rubric.add_metric(used_rlm_subcalls_metric)
    rubric.add_metric(num_subcalls_metric)
    rubric.add_metric(num_llm_subcalls_metric)
    rubric.add_metric(num_rlm_subcalls_metric)
    rubric.add_metric(max_depth_metric)
    rubric.add_metric(used_forced_finalize_prompt_metric)
    rubric.add_metric(hit_max_turn_without_final_metric)
    rubric.add_metric(missing_final_metric)
    rubric.add_metric(finalized_before_forced_prompt_metric)
    rubric.add_metric(finalized_on_forced_prompt_metric)
    return rubric
