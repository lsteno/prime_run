from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.rlm_traces import export_sft
from rlm_rlvr.prompt_variants import DEFAULT_PROMPT_VARIANT, get_system_prompt_template


DEFAULT_INPUT = Path("outputs/rlm_traces/combined-good-sft-traces-v1/records.jsonl")
DEFAULT_V2_INPUTS = [
    Path("outputs/rlm_traces/combined-good-sft-traces-v1/records.jsonl"),
    Path("outputs/rlm_traces/prime-gpt54-gpt55-sft-missing-auto-retry-v1/combined_candidate_records.jsonl"),
]
DEFAULT_OUTPUT_DIR = Path("outputs/rlm_traces/combined-good-sft-traces-v2/curated-v2")
CURATION_VERSION = "curated-v2"
HIGH_SUBCALL_THRESHOLD = 20

FENCE_RE = re.compile(r"```([A-Za-z0-9_-]*)\n(.*?)```", re.DOTALL)
FINAL_ANY_RE = re.compile(r"\bFINAL(?:_VAR)?\s*\(", re.DOTALL)
FINAL_LITERAL_RE = re.compile(r"\bFINAL\s*\(", re.DOTALL)
FINAL_VAR_RE = re.compile(r"\bFINAL_VAR\s*\(\s*([A-Za-z_][A-Za-z0-9_]*|['\"]([A-Za-z_][A-Za-z0-9_]*)['\"])\s*\)", re.DOTALL)
FINAL_IDENTIFIER_RE = re.compile(r"\bFINAL\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$", re.DOTALL)
ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", re.MULTILINE)
PREFERRED_FINAL_VARIABLES = (
    "answer",
    "final_answer",
    "result",
    "output",
    "final",
    "ans",
    "solution",
    "final_json",
)

PRESERVE_QUOTED_LITERAL_FINAL_IN_REPL_SOURCE_IDS = {
    "frames-0010": "The stored final answer includes quotes; preserving the original final REPL turn avoids teaching outside-REPL literal quotes as answer content.",
    "frames-0230": "The stored final answer includes quotes; preserving the original final REPL turn avoids teaching outside-REPL literal quotes as answer content.",
    "frames-0256": "The stored final answer includes quotes; preserving the original final REPL turn avoids teaching outside-REPL literal quotes as answer content.",
    "frames-0424": "The stored final answer includes quotes; preserving the original final REPL turn avoids teaching outside-REPL literal quotes as answer content.",
    "frames-0490": "The stored final answer includes quotes; preserving the original final REPL turn avoids teaching outside-REPL literal quotes as answer content.",
    "frames-0534": "The stored final answer includes quotes; preserving the original final REPL turn avoids teaching outside-REPL literal quotes as answer content.",
}

MANUAL_AUDIT_ONLY_SOURCE_DECISIONS = {
    "frames-0029": "Subcalls only returned NOT_FOUND; the root answered from its own context search rather than using subcall evidence or valid exclusion.",
    "frames-0052": "Subcalls only returned NOT_FOUND; the root recovered with direct context search rather than using subcall evidence or valid exclusion.",
    "frames-0087": "All subcalls were NOT_FOUND; the trajectory relies on root-led context search rather than a subcall-supported answer.",
    "frames-0529": "All Harry Potter page-count subcalls returned NOT_FOUND; the answer was not derived from useful delegation.",
    "frames-0693": "All NBA standings subcalls returned NOT_FOUND; the answer came from root REPL inspection.",
    "frames-0262": "Both Vivaldi subcalls returned NOT_FOUND; the trace does not show useful delegated evidence.",
    "oolong-000696": "Both verification subcalls returned NOT_FOUND; the final comparison was determined by direct REPL counting only.",
    "frames-0122": "Both subcalls returned NOT_FOUND; the root answered from direct context snippets.",
    "frames-0232": "All Jeep/codename subcalls returned NOT_FOUND; the root had already guessed/derived the answer without useful subcall evidence.",
    "frames-0044": "Both G40 station subcalls returned NOT_FOUND; the answer came from root-led search, not delegation.",
    "frames-0423": "All mayor/hometown subcalls returned NOT_FOUND; the answer came from root-led search, not delegation.",
}


# Hand-reviewed empty-subcall decisions. These are intentionally explicit:
# empty subcall responses are never synthesized or modified, but the trace can
# still be SFT-worthy when the trajectory obtains enough evidence elsewhere.
MANUAL_EMPTY_SUBCALL_DECISIONS: dict[str, dict[str, str]] = {
    "CU_P-0153-450308": {
        "status": "unrecovered",
        "recovery": "repl_only_after_empty_subcall",
        "rationale": "The trace had useful subcalls before the empty response, but after the empty file-extraction subcall it recovered only with deterministic REPL extraction and made no later non-empty subcall.",
    },
    "frames-0430": {
        "status": "unrecovered",
        "recovery": "repl_only_after_empty_subcall",
        "rationale": "The trace had earlier non-empty subcalls, but the empty screenplay subcalls were followed by finalization without any later non-empty subcall.",
    },
    "oolong-000265": {
        "status": "unrecovered",
        "recovery": "repl_only_after_empty_subcall",
        "rationale": "After the empty label-count subcall, the trace finalized from deterministic REPL counting and did not make a later non-empty subcall.",
    },
    "oolong-000338": {
        "status": "unrecovered",
        "recovery": "repl_only_after_empty_subcall",
        "rationale": "After the empty user-label subcall, the trace finalized from REPL-derived label distribution and did not make a later non-empty subcall.",
    },
    "oolong-000531": {
        "status": "unrecovered",
        "recovery": "repl_only_after_empty_subcall",
        "rationale": "After the empty dataset-structure subcall, the trace recovered only by counting user IDs in REPL and made no later non-empty subcall.",
    },
    "oolong-000664": {
        "status": "recovered",
        "recovery": "later_subcall_and_repl_verification",
        "rationale": "The first label-format subcall is empty, but a later subcall returns label evidence and the root also computes exact counts in REPL.",
    },
    "oolong-000786": {
        "status": "recovered",
        "recovery": "later_subcall_and_repl_verification",
        "rationale": "The first label-token subcall is empty, but later subcall output confirms format while exact class counts are computed in REPL.",
    },
    "oolong-000869": {
        "status": "recovered",
        "recovery": "direct_repl_verification",
        "rationale": "Several chunk-count subcalls are empty/partial, but the final count comes from a direct full-context REPL count of User: 72838.",
    },
    "oolong-000907": {
        "status": "unrecovered",
        "recovery": "repl_only_after_empty_subcall",
        "rationale": "After the empty top-user chunk subcall, the trace finalized from an existing REPL count and made no later non-empty subcall.",
    },
    "oolong-000924": {
        "status": "recovered",
        "recovery": "later_subcall_and_repl_verification",
        "rationale": "Multiple exploratory subcalls are empty, but later subcall evidence and exact REPL label counts support the final label.",
    },
    "oolong-001286": {
        "status": "recovered",
        "recovery": "later_subcall_and_repl_verification",
        "rationale": "One chunk-count subcall is empty, but the full-context REPL counter determines the final JSON and later chunk evidence confirms format.",
    },
    "oolong-001302": {
        "status": "unrecovered",
        "recovery": "repl_only_after_empty_subcall",
        "rationale": "After the empty review-label subcalls, the trace finalized from REPL positive/negative counts and made no later non-empty subcall.",
    },
    "oolong-001373": {
        "status": "unrecovered",
        "recovery": "repl_only_after_empty_subcall",
        "rationale": "After the empty month-year chunk subcalls, the trace finalized from REPL regex counts and made no later non-empty subcall.",
    },
    "oolong-001396": {
        "status": "unrecovered",
        "recovery": "repl_only_after_empty_subcall",
        "rationale": "After the empty month-count subcalls, the trace finalized from direct REPL counting and made no later non-empty subcall.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate combined good RLM traces for SFT.")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        dest="inputs",
        help="Input records.jsonl path. Can be passed more than once. Defaults to the unified v2 inputs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--manual-decisions",
        type=Path,
        default=None,
        help="Manual decisions JSONL. Without this, the script writes candidate artifacts only.",
    )
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Write candidate artifacts and audit pages without final strict SFT rows.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records_with_provenance(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    input_manifest: list[dict[str, Any]] = []
    for input_index, path in enumerate(paths):
        before_hash = sha256_file(path)
        input_records = read_jsonl(path)
        input_manifest.append(
            {
                "path": str(path),
                "input_index": input_index,
                "record_count": len(input_records),
                "sha256_before": before_hash,
                "sha256_after": None,
                "unchanged": None,
            }
        )
        for row_index, record in enumerate(input_records):
            copied = copy.deepcopy(record)
            copied["_curation_input_path"] = str(path)
            copied["_curation_input_index"] = input_index
            copied["_curation_input_row_index"] = row_index
            records.append(copied)
    return records, input_manifest


def dedupe_records_by_source_id(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Preserve first occurrence by source_id.

    The unified v2 input order places already-curated/previous-good traces first,
    so this keeps the known-good copy unless a future manual workflow explicitly
    changes the input ordering or record selection.
    """
    selected: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for record in records:
        source_id = str(record.get("source_id") or record.get("example_id") or "")
        if not source_id:
            source_id = f"missing-source:{record.get('_curation_input_index')}:{record.get('_curation_input_row_index')}"
            record["source_id"] = source_id
        if source_id in selected:
            kept = selected[source_id]
            duplicates.append(
                {
                    "source_id": source_id,
                    "kept_input_path": kept.get("_curation_input_path"),
                    "kept_input_row_index": kept.get("_curation_input_row_index"),
                    "dropped_input_path": record.get("_curation_input_path"),
                    "dropped_input_row_index": record.get("_curation_input_row_index"),
                    "reason": "duplicate source_id; preserved first input occurrence",
                }
            )
            continue
        selected[source_id] = record
    return list(selected.values()), duplicates


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def remove_code_blocks(text: str) -> str:
    return FENCE_RE.sub("", text)


def code_blocks(text: str) -> list[tuple[str, str]]:
    return [(match.group(1).strip().lower(), match.group(2)) for match in FENCE_RE.finditer(text)]


def has_final_outside_code(text: str) -> bool:
    return bool(FINAL_ANY_RE.search(remove_code_blocks(text)))


def has_literal_final_outside_code(text: str) -> bool:
    return bool(FINAL_LITERAL_RE.search(remove_code_blocks(text)))


def has_final_var_outside_code(text: str) -> bool:
    return bool(FINAL_VAR_RE.search(remove_code_blocks(text)))


def has_final_inside_repl(text: str) -> bool:
    for language, body in code_blocks(text):
        if language in {"repl", "python", ""} and FINAL_ANY_RE.search(body):
            return True
    return False


def has_literal_final_inside_repl(text: str) -> bool:
    for language, body in code_blocks(text):
        if language in {"repl", "python", ""} and FINAL_LITERAL_RE.search(body):
            return True
    return False


def has_final_var_inside_repl(text: str) -> bool:
    for language, body in code_blocks(text):
        if language in {"repl", "python", ""} and FINAL_VAR_RE.search(body):
            return True
    return False


def has_clean_finalization(text: str) -> bool:
    """Runtime-aligned finalization check for SFT targets.

    Literal `FINAL(value)` is parser text and belongs outside executable code.
    `FINAL_VAR("name")` is a REPL helper and is cleanest inside a REPL block.
    """
    if has_literal_final_outside_code(text):
        return True
    if has_final_var_inside_repl(text):
        return True
    return False


def segment_has_clean_finalization(segment: dict[str, Any]) -> bool:
    if segment.get("_curation_preserve_quoted_literal_final_in_repl"):
        return True
    synthetic_rows = segment.get("_curation_sft_rows")
    if isinstance(synthetic_rows, list):
        return any(has_clean_finalization(str(row.get("response_text", ""))) for row in synthetic_rows if isinstance(row, dict))
    return has_clean_finalization(str(segment.get("response_text", "")))


def has_repl_code(text: str) -> bool:
    return any(language in {"repl", "python", ""} for language, _ in code_blocks(text))


def is_expression_final(text: str) -> bool:
    outside = remove_code_blocks(text)
    return bool(re.search(r"\bFINAL\s*\(\s*(?:json\.dumps|str|repr)\s*\(", outside))


def final_identifier_name(text: str) -> str | None:
    outside = remove_code_blocks(text).strip()
    match = FINAL_IDENTIFIER_RE.search(outside)
    if not match:
        return None
    return match.group(1)


def final_var_name(text: str) -> str | None:
    match = FINAL_VAR_RE.search(text)
    if not match:
        return None
    return match.group(2) or match.group(1).strip("'\"")


def repl_final_var_text(variable_name: str) -> str:
    return f'```repl\nFINAL_VAR("{variable_name}")\n```'


def canonical_final_text(record: dict[str, Any]) -> str:
    final_answer = str(record.get("final_answer") or record.get("answer") or "").strip()
    if FINAL_ANY_RE.search(final_answer):
        return final_answer
    return f"FINAL({final_answer})"


def preserve_quoted_literal_final_inside_repl(record: dict[str, Any], response: str) -> str | None:
    source_id = str(record.get("source_id"))
    if source_id not in PRESERVE_QUOTED_LITERAL_FINAL_IN_REPL_SOURCE_IDS:
        return None
    if not has_literal_final_inside_repl(response) or has_final_outside_code(response):
        return None
    final_answer = str(record.get("final_answer") or "").strip()
    if not (
        (final_answer.startswith('"') and final_answer.endswith('"'))
        or (final_answer.startswith("'") and final_answer.endswith("'"))
    ):
        return None
    return PRESERVE_QUOTED_LITERAL_FINAL_IN_REPL_SOURCE_IDS[source_id]


def infer_source_run(record: dict[str, Any]) -> str | None:
    for key in ("_source_run_id", "source_run_id"):
        value = record.get(key)
        if value:
            return str(value)
    source_dir = record.get("_source_run_dir")
    if source_dir:
        return Path(str(source_dir)).name
    input_path = record.get("_curation_input_path")
    if input_path:
        parts = Path(str(input_path)).parts
        for part in reversed(parts):
            if part.startswith("prime-") or part.startswith("combined-good"):
                return part
    return None


def infer_root_model(record: dict[str, Any]) -> str | None:
    for key in ("_root_model", "root_model", "model"):
        value = record.get(key)
        if value:
            return str(value)
    endpoint = record.get("endpoint")
    if isinstance(endpoint, dict) and endpoint.get("model"):
        return str(endpoint["model"])
    return None


def infer_final_variable(record: dict[str, Any], segment: dict[str, Any] | None = None) -> str | None:
    max_order = int(segment.get("order", 10**9) or 10**9) if isinstance(segment, dict) else 10**9
    assigned: list[str] = []
    for candidate_segment in trainable_segments(record):
        order = int(candidate_segment.get("order", 0) or 0)
        if order > max_order:
            continue
        for _language, body in code_blocks(str(candidate_segment.get("response_text", ""))):
            assigned.extend(match.group(1) for match in ASSIGNMENT_RE.finditer(body))
    assigned_set = set(assigned)
    for preferred in PREFERRED_FINAL_VARIABLES:
        if preferred in assigned_set:
            return preferred
    return assigned[-1] if assigned else None


def trainable_segments(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        segment
        for segment in record.get("segments", [])
        if isinstance(segment, dict) and export_sft.segment_is_trainable_sft_turn(segment)
    ]


def plain_subcall_segments(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        segment
        for segment in record.get("segments", [])
        if isinstance(segment, dict) and str(segment.get("kind")) == "plain_query"
    ]


def non_empty_plain_subcall_segments(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [segment for segment in plain_subcall_segments(record) if not segment_is_empty_subcall(segment)]


def segment_is_empty_subcall(segment: dict[str, Any]) -> bool:
    return not str(segment.get("response_text", "")).strip()


def response_mentions_empty_result(text: str) -> bool:
    lowered = text.lower()
    suspicious = [
        "empty result",
        "empty response",
        "blank result",
        "blank response",
        "returned nothing",
        "no output",
    ]
    return any(marker in lowered for marker in suspicious)


def classify_empty_subcalls(record: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], str]:
    empty_segments = [
        {
            "order": segment.get("order"),
            "call_id": segment.get("call_id"),
            "prompt_fingerprint": segment.get("prompt_fingerprint"),
        }
        for segment in plain_subcall_segments(record)
        if segment_is_empty_subcall(segment)
    ]
    if not empty_segments:
        return [], [], "no_empty_subcalls"
    if not non_empty_plain_subcall_segments(record):
        return (
            ["empty_subcall", "empty_subcall_unrecovered", "all_subcalls_empty"],
            empty_segments,
            "all plain LLM subcalls are empty, so the trace does not teach usable delegation",
        )

    manual = MANUAL_EMPTY_SUBCALL_DECISIONS.get(str(record.get("source_id")))
    if manual is not None:
        tags = ["empty_subcall", "manual_empty_subcall_review"]
        if manual["status"] == "recovered":
            tags.append("empty_subcall_recovered")
        else:
            tags.append("empty_subcall_unrecovered")
        tags.append(str(manual["recovery"]))
        return tags, empty_segments, str(manual["rationale"])

    tags = ["empty_subcall"]
    empty_orders = [int(item["order"]) for item in empty_segments if item.get("order") is not None]
    later_non_empty_subcall = False
    later_trainable_evidence = False
    depends_on_empty = False
    max_empty_order = max(empty_orders) if empty_orders else -1
    for segment in record.get("segments", []):
        if not isinstance(segment, dict):
            continue
        order = int(segment.get("order", -1) or -1)
        text = str(segment.get("response_text", ""))
        if order > max_empty_order and str(segment.get("kind")) == "plain_query" and text.strip():
            later_non_empty_subcall = True
        if order > max_empty_order and bool(segment.get("is_trainable_rlm_turn")) and has_repl_code(text):
            later_trainable_evidence = True
        if bool(segment.get("is_trainable_rlm_turn")) and response_mentions_empty_result(text):
            depends_on_empty = True

    if depends_on_empty and not (later_non_empty_subcall or later_trainable_evidence):
        tags.append("empty_subcall_unrecovered")
        return tags, empty_segments, "empty subcall appears referenced without later retry/evidence"

    if later_non_empty_subcall:
        tags.extend(["empty_subcall_recovered", "later_subcall_after_empty"])
        return tags, empty_segments, "empty subcall is followed by later non-empty subcall evidence/retry"

    if later_trainable_evidence:
        tags.extend(["empty_subcall_recovered", "repl_verification_after_empty"])
        return tags, empty_segments, "empty subcall is non-critical and followed by clear REPL verification"

    tags.append("empty_subcall_unrecovered")
    return tags, empty_segments, "empty subcall is not followed by later subcall evidence or clear REPL verification"


def root_step_for_response(record: dict[str, Any], response_text: str) -> dict[str, Any] | None:
    for step in record.get("root_steps", []):
        if isinstance(step, dict) and str(step.get("assistant", "")).strip() == response_text.strip():
            return step
    return None


def feedback_messages_for_response(record: dict[str, Any], response_text: str) -> list[dict[str, str]]:
    step = root_step_for_response(record, response_text)
    if not step:
        return []
    messages = step.get("feedback_messages")
    if isinstance(messages, list):
        return [
            {"role": str(message.get("role", "user")), "content": str(message.get("content", "")).strip()}
            for message in messages
            if isinstance(message, dict) and str(message.get("content", "")).strip()
        ]
    feedback = step.get("feedback")
    if isinstance(feedback, list) and feedback:
        return [{"role": "user", "content": "\n\n".join(str(item) for item in feedback if str(item).strip())}]
    return []


def strip_final_outside_code(text: str) -> str:
    parts: list[str] = []
    last = 0
    for match in FENCE_RE.finditer(text):
        outside = text[last : match.start()]
        outside = FINAL_ANY_RE.split(outside, maxsplit=1)[0]
        parts.append(outside.rstrip())
        parts.append(match.group(0).strip())
        last = match.end()
    outside = text[last:]
    outside = FINAL_ANY_RE.split(outside, maxsplit=1)[0]
    parts.append(outside.rstrip())
    return "\n\n".join(part for part in parts if part.strip()).strip()


def first_correct_bare_answer_segment(segments: list[dict[str, Any]], final_answer: str) -> dict[str, Any] | None:
    normalized_final = final_answer.strip()
    for segment in segments:
        response = str(segment.get("response_text", "")).strip()
        if response and response == normalized_final:
            return segment
    for segment in segments:
        response = str(segment.get("response_text", "")).strip()
        if response and not has_repl_code(response) and not has_final_outside_code(response):
            return segment
    return None


def apply_finalization_repairs(record: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    tags: list[str] = []
    patches: list[dict[str, Any]] = []
    exclusions: list[str] = []
    final_text = canonical_final_text(record)
    final_answer = str(record.get("final_answer") or record.get("answer") or "").strip()
    segments = trainable_segments(record)
    if not segments:
        tags.append("no_trainable_segments")
        exclusions.append("no trainable RLM turns")
        return tags, patches, exclusions

    def patch_segment(segment: dict[str, Any], new_response: str, repair_type: str, rationale: str) -> None:
        old_response = str(segment.get("response_text", ""))
        if old_response.strip() == new_response.strip():
            return
        repair_id = f"{record.get('source_id')}:patch:{len(patches)}"
        patches.append(
            {
                "repair_id": repair_id,
                "source_id": record.get("source_id"),
                "segment_order": segment.get("order"),
                "repair_type": repair_type,
                "old_response_text": old_response,
                "new_response_text": new_response,
                "rationale": rationale,
            }
        )
        segment["response_text"] = new_response
        segment.setdefault("curation", {})["repair_type"] = repair_type
        segment["curation"]["repair_id"] = repair_id
        segment["curation"]["old_response_text"] = old_response

    for segment in segments:
        response = str(segment.get("response_text", ""))
        preserve_rationale = preserve_quoted_literal_final_inside_repl(record, response)
        if preserve_rationale:
            tags.append("literal_final_in_repl_preserved_quoted")
            segment["_curation_preserve_quoted_literal_final_in_repl"] = True
            segment.setdefault("curation", {})["preserve_literal_final_in_repl"] = True
            segment["curation"]["preserve_reason"] = preserve_rationale
        elif is_expression_final(response):
            tags.append("expression_final_repaired")
            patch_segment(
                segment,
                final_text,
                "expression_final",
                "FINAL contained an unevaluated expression; replaced with stored accepted final answer.",
            )
        elif has_literal_final_inside_repl(response) and not has_final_outside_code(response):
            tags.append("literal_final_in_repl_repaired")
            patch_segment(
                segment,
                final_text,
                "literal_final_in_repl",
                "Moved/collapsed literal FINAL from executable code into a final-only assistant response.",
            )
        elif has_final_var_outside_code(response) and not has_final_var_inside_repl(response) and not has_repl_code(response):
            variable_name = final_var_name(response)
            if variable_name:
                tags.append("final_var_moved_into_repl")
                patch_segment(
                    segment,
                    repl_final_var_text(variable_name),
                    "final_var_outside_repl",
                    "Moved FINAL_VAR into a REPL block because FINAL_VAR is a Python REPL helper.",
                )
        elif has_repl_code(response) and has_final_outside_code(response):
            analysis_response = strip_final_outside_code(response)
            feedback = feedback_messages_for_response(record, response)
            synthetic_prompt = export_sft.sanitize_prompt_messages(segment.get("prompt_messages", []))
            if analysis_response and feedback:
                final_response = final_text
                variable_name = final_var_name(response)
                if variable_name:
                    final_response = repl_final_var_text(variable_name)
                synthetic_prompt = synthetic_prompt + [{"role": "assistant", "content": analysis_response}] + feedback
                segment["_curation_sft_rows"] = [
                    {
                        "response_text": analysis_response,
                        "prompt_messages": export_sft.sanitize_prompt_messages(segment.get("prompt_messages", [])),
                        "row_role": "analysis_before_final",
                    },
                    {
                        "response_text": final_response,
                        "prompt_messages": synthetic_prompt,
                        "row_role": "synthetic_final_after_feedback",
                    },
                ]
                segment.setdefault("curation", {})["repair_type"] = "code_plus_final_split"
                tags.append("code_plus_final_split")
                patches.append(
                    {
                        "repair_id": f"{record.get('source_id')}:patch:{len(patches)}",
                        "source_id": record.get("source_id"),
                        "segment_order": segment.get("order"),
                        "repair_type": "code_plus_final_split",
                        "old_response_text": response,
                        "new_response_text": None,
                        "rationale": "Split same-turn code plus FINAL into an analysis SFT row and a synthetic final row using existing REPL feedback.",
                    }
                )
            else:
                tags.append("code_plus_final_collapsed")
                collapsed_response = final_text
                variable_name = final_var_name(response)
                if variable_name:
                    collapsed_response = repl_final_var_text(variable_name)
                patch_segment(
                    segment,
                    collapsed_response,
                    "code_plus_final_collapse",
                    "Could not recover feedback for a clean split; kept only the stored accepted final answer.",
                )
        else:
            identifier = final_identifier_name(response)
            if identifier and final_answer and identifier != final_answer:
                if identifier in set(ASSIGNMENT_RE.findall("\n".join(body for _language, body in code_blocks(response)))) or identifier == infer_final_variable(record, segment):
                    tags.append("final_identifier_repaired_to_final_var")
                    patch_segment(
                        segment,
                        repl_final_var_text(identifier),
                        "final_identifier_to_final_var",
                        "FINAL(identifier) would be parsed literally; repaired to REPL FINAL_VAR for the existing variable.",
                    )
                else:
                    tags.append("final_identifier_repaired_to_literal")
                    patch_segment(
                        segment,
                        final_text,
                        "final_identifier_to_literal",
                        "FINAL(identifier) would be parsed literally and no safe variable provenance was found; replaced with stored accepted final answer.",
                    )

    forced_finalize = any(bool(step.get("forced_finalize")) for step in record.get("root_steps", []) if isinstance(step, dict))
    final_wrapped_segments = [segment for segment in segments if segment_has_clean_finalization(segment)]
    if forced_finalize:
        tags.append("forced_finalize")
        bare_segment = first_correct_bare_answer_segment(segments, final_answer)
        if bare_segment is not None:
            tags.append("bare_answer_loop_repaired")
            patch_segment(
                bare_segment,
                final_text,
                "bare_answer_loop",
                "Forced finalization followed repeated bare answers; rewrote first accepted bare answer as FINAL(actual_answer).",
            )
            started = False
            for segment in segments:
                if segment is bare_segment:
                    started = True
                    continue
                if started:
                    response = str(segment.get("response_text", "")).strip()
                    if response == final_answer or str(segment.get("kind")) == "root_finalize_turn":
                        segment.setdefault("curation", {})["exclude_from_sft"] = True
                        segment["curation"]["exclude_reason"] = "redundant forced-finalization loop turn"
        elif not final_wrapped_segments:
            tags.append("unrepairable_finalization")
            exclusions.append("forced finalization with no safe bare-answer segment to repair")

    segments = trainable_segments(record)
    if not any(segment_has_clean_finalization(segment) for segment in segments):
        last_segment = segments[-1]
        response = str(last_segment.get("response_text", "")).strip()
        if response and not has_repl_code(response):
            tags.append("missing_final_wrapper_repaired")
            patch_segment(
                last_segment,
                final_text,
                "missing_final_wrapper",
                "Last accepted answer lacked FINAL wrapper; wrapped stored accepted final answer.",
            )
        else:
            variable_name = infer_final_variable(record, last_segment)
            if variable_name:
                tags.append("missing_final_var_repaired")
                feedback = feedback_messages_for_response(record, response)
                final_response = repl_final_var_text(variable_name)
                if response and feedback:
                    analysis_response = strip_final_outside_code(response) or response
                    synthetic_prompt = (
                        export_sft.sanitize_prompt_messages(last_segment.get("prompt_messages", []))
                        + [{"role": "assistant", "content": analysis_response}]
                        + feedback
                    )
                    last_segment["_curation_sft_rows"] = [
                        {
                            "response_text": analysis_response,
                            "prompt_messages": export_sft.sanitize_prompt_messages(last_segment.get("prompt_messages", [])),
                            "row_role": "analysis_before_final",
                        },
                        {
                            "response_text": final_response,
                            "prompt_messages": synthetic_prompt,
                            "row_role": "synthetic_final_after_feedback",
                        },
                    ]
                    last_segment.setdefault("curation", {})["repair_type"] = "missing_final_var_split"
                    patches.append(
                        {
                            "repair_id": f"{record.get('source_id')}:patch:{len(patches)}",
                            "source_id": record.get("source_id"),
                            "segment_order": last_segment.get("order"),
                            "repair_type": "missing_final_var_split",
                            "old_response_text": response,
                            "new_response_text": None,
                            "rationale": "Split analysis code from a synthetic FINAL_VAR turn using existing REPL feedback.",
                        }
                    )
                else:
                    patch_segment(
                        last_segment,
                        final_response,
                        "missing_final_var",
                        "Last response lacked clean finalization but had a clear REPL variable; added FINAL_VAR in a REPL block.",
                    )
            else:
                tags.append("unrepairable_finalization")
                exclusions.append("no clean FINAL wrapper and last response is not safely wrappable")

    return sorted(set(tags)), patches, exclusions


def strict_sft_status(record: dict[str, Any], tags: list[str], exclusions: list[str]) -> tuple[bool, str]:
    if not export_sft.record_is_accepted(record):
        return False, "not accepted by exact match or judge"
    if not non_empty_plain_subcall_segments(record):
        return False, "does not use any non-empty plain LLM subcall"
    if "empty_subcall_unrecovered" in tags:
        return False, "empty subcall appears unrecovered"
    if any(tag == "unrepairable_finalization" for tag in tags):
        return False, "; ".join(exclusions) or "unrepairable finalization"
    if exclusions:
        return False, "; ".join(exclusions)
    return True, "accepted, uses subcalls, and has clean or repaired finalization"


def curate_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    curated = copy.deepcopy(record)
    tags: list[str] = []
    rationale: list[str] = []
    empty_tags, empty_segments, empty_rationale = classify_empty_subcalls(curated)
    tags.extend(empty_tags)
    if empty_segments:
        rationale.append(empty_rationale)
    final_tags, patches, exclusions = apply_finalization_repairs(curated)
    tags.extend(final_tags)
    if int(curated.get("num_llm_subcalls", 0) or 0) > HIGH_SUBCALL_THRESHOLD:
        tags.extend(["overdelegated", "high_subcall_count"])
    strict, status_reason = strict_sft_status(curated, tags, exclusions)
    manual_audit_only_reason = MANUAL_AUDIT_ONLY_SOURCE_DECISIONS.get(str(curated.get("source_id")))
    if manual_audit_only_reason:
        strict = False
        tags.extend(["audit_only_manual_exclusion", "weak_subcall_signal"])
        status_reason = manual_audit_only_reason
    if strict:
        tags.append("strict_sft")
    else:
        tags.append("audit_only")

    tags = sorted(set(tags))
    source_run_id = infer_source_run(curated)
    root_model = infer_root_model(curated)
    source_input_path = curated.get("_curation_input_path")
    manual_decision_id = f"{curated.get('source_id')}:record"
    curation = {
        "version": CURATION_VERSION,
        "status": "strict_sft" if strict else "audit_only",
        "tags": tags,
        "rationale": "; ".join(rationale + [status_reason]),
        "empty_subcalls": empty_segments,
        "patch_count": len(patches),
        "manual_decision_id": manual_decision_id,
        "source_input_path": source_input_path,
        "source_run_id": source_run_id,
        "root_model": root_model,
        "non_empty_llm_subcall_count": len(non_empty_plain_subcall_segments(curated)),
    }
    curated["curation"] = curation
    manifest = {
        "source_id": curated.get("source_id"),
        "example_id": curated.get("example_id"),
        "source_input_path": source_input_path,
        "source_run_id": source_run_id,
        "root_model": root_model,
        "curation_version": CURATION_VERSION,
        "status": curation["status"],
        "manual_decision_id": manual_decision_id,
        "tags": ",".join(tags),
        "rationale": curation["rationale"],
        "num_llm_subcalls": int(curated.get("num_llm_subcalls", 0) or 0),
        "non_empty_llm_subcall_count": len(non_empty_plain_subcall_segments(curated)),
        "num_trainable_segments": len(trainable_segments(curated)),
        "empty_subcall_count": len(empty_segments),
        "patch_count": len(patches),
        "final_answer": curated.get("final_answer"),
        "exact_match": bool(curated.get("exact_match")),
        "judge_score": curated.get("judge_score"),
    }
    return curated, manifest, patches


def sft_rows_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("curation", {}).get("status") != "strict_sft":
        return []
    rows: list[dict[str, Any]] = []
    source_id = str(record.get("source_id", record.get("example_id", "")))
    for segment in record.get("segments", []):
        if not isinstance(segment, dict) or not export_sft.segment_is_trainable_sft_turn(segment):
            continue
        curation = segment.get("curation", {})
        if isinstance(curation, dict) and curation.get("exclude_from_sft"):
            continue
        synthetic_rows = segment.get("_curation_sft_rows")
        if isinstance(synthetic_rows, list):
            for index, synthetic in enumerate(synthetic_rows):
                rows.append(
                    make_sft_row(
                        record,
                        segment,
                        source_id=source_id,
                        response_text=str(synthetic["response_text"]).strip(),
                        prompt_messages=synthetic["prompt_messages"],
                        row_role=str(synthetic.get("row_role", "curated")),
                        synthetic_index=index,
                    )
                )
            continue
        rows.append(
            make_sft_row(
                record,
                segment,
                source_id=source_id,
                response_text=str(segment["response_text"]).strip(),
                prompt_messages=segment["prompt_messages"],
                row_role="original_or_repaired",
                synthetic_index=None,
            )
        )
    return rows


def make_sft_row(
    record: dict[str, Any],
    segment: dict[str, Any],
    *,
    source_id: str,
    response_text: str,
    prompt_messages: list[dict[str, Any]],
    row_role: str,
    synthetic_index: int | None,
) -> dict[str, Any]:
    segment_order = int(segment.get("order", 0) or 0)
    row_id = f"{source_id}:{segment_order}"
    if synthetic_index is not None:
        row_id += f":{synthetic_index}"
    sanitized_prompt = sanitize_curated_prompt_messages(record, prompt_messages)
    return {
        "row_id": row_id,
        "source_id": source_id,
        "example_id": record.get("example_id"),
        "source_input_path": record.get("curation", {}).get("source_input_path"),
        "source_run_id": record.get("curation", {}).get("source_run_id"),
        "root_model": record.get("curation", {}).get("root_model"),
        "curation_version": CURATION_VERSION,
        "curation_status": record.get("curation", {}).get("status"),
        "manual_decision_id": record.get("curation", {}).get("manual_decision_id"),
        "segment_order": segment_order,
        "segment_kind": str(segment.get("kind")),
        "row_role": row_role,
        "prompt": sanitized_prompt,
        "completion": [{"role": "assistant", "content": response_text}],
        "final_answer": str(record.get("final_answer", "")),
        "exact_match": bool(record.get("exact_match")),
        "judge_score": record.get("judge_score"),
        "prompt_token_count": int(segment.get("prompt_token_count", 0) or 0),
        "completion_token_count": int(segment.get("completion_token_count", 0) or 0),
        "curation_tags": record.get("curation", {}).get("tags", []),
        "prompt_variant": record.get("prompt_variant"),
        "prompt_rewritten_to_current_variant": bool(sanitized_prompt and sanitized_prompt[0].get("role") == "system"),
    }


def sanitize_curated_prompt_messages(record: dict[str, Any], prompt_messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = export_sft.sanitize_prompt_messages(prompt_messages)
    if not messages or messages[0].get("role") != "system":
        return messages
    prompt_variant = str(record.get("prompt_variant") or DEFAULT_PROMPT_VARIANT)
    try:
        messages[0] = {"role": "system", "content": get_system_prompt_template(prompt_variant)}
    except ValueError:
        messages[0] = {"role": "system", "content": get_system_prompt_template(DEFAULT_PROMPT_VARIANT)}
    return messages


def render_audit_sample(record: dict[str, Any], patches: list[dict[str, Any]]) -> str:
    lines = [
        f"# Trace Audit: {record.get('source_id')}",
        "",
        f"- status: `{record.get('curation', {}).get('status')}`",
        f"- tags: `{', '.join(record.get('curation', {}).get('tags', []))}`",
        f"- final_answer: `{record.get('final_answer')}`",
        f"- llm_subcalls: `{record.get('num_llm_subcalls')}`",
        "",
        "## Question",
        "",
        str(record.get("question", "")).strip(),
        "",
    ]
    if patches:
        lines.extend(["## Finalization Patches", ""])
        for patch in patches:
            lines.extend(
                [
                    f"### Segment {patch.get('segment_order')} - {patch.get('repair_type')}",
                    "",
                    "**Before**",
                    "",
                    "```text",
                    str(patch.get("old_response_text", "")).strip(),
                    "```",
                    "",
                    "**After**",
                    "",
                    "```text",
                    str(patch.get("new_response_text", "")).strip() if patch.get("new_response_text") is not None else "<split into curated SFT rows>",
                    "```",
                    "",
                    str(patch.get("rationale", "")).strip(),
                    "",
                ]
            )
    lines.extend(["## Curated Trainable Turns", ""])
    for segment in trainable_segments(record):
        curation = segment.get("curation", {}) if isinstance(segment.get("curation"), dict) else {}
        excluded = " excluded_from_sft" if curation.get("exclude_from_sft") else ""
        synthetic_rows = segment.get("_curation_sft_rows")
        if isinstance(synthetic_rows, list):
            lines.extend([f"### Segment {segment.get('order')} {segment.get('kind')} curated synthetic rows", ""])
            for index, synthetic in enumerate(synthetic_rows):
                lines.extend(
                    [
                        f"#### Synthetic Row {index}: {synthetic.get('row_role')}",
                        "",
                        "```text",
                        str(synthetic.get("response_text", "")).strip()[:4000],
                        "```",
                        "",
                    ]
                )
            continue
        lines.extend(
            [
                f"### Segment {segment.get('order')} {segment.get('kind')}{excluded}",
                "",
                "```text",
                str(segment.get("response_text", "")).strip()[:4000],
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_compact_audit_page(record: dict[str, Any], patches: list[dict[str, Any]]) -> str:
    lines = [
        f"# Compact Trace Audit: {record.get('source_id')}",
        "",
        f"- status: `{record.get('curation', {}).get('status')}`",
        f"- model: `{record.get('curation', {}).get('root_model')}`",
        f"- source: `{record.get('curation', {}).get('source_input_path')}`",
        f"- tags: `{', '.join(record.get('curation', {}).get('tags', []))}`",
        f"- final_answer: `{record.get('final_answer')}`",
        f"- exact_match: `{record.get('exact_match')}`",
        f"- judge_score: `{record.get('judge_score')}`",
        f"- llm_subcalls: `{record.get('num_llm_subcalls')}`, non_empty: `{len(non_empty_plain_subcall_segments(record))}`",
        "",
        "## Question",
        "",
        str(record.get("question", "")).strip()[:2000],
        "",
    ]
    if patches:
        lines.extend(["## Patches", ""])
        for patch in patches:
            lines.extend(
                [
                    f"- `{patch.get('repair_id')}` segment `{patch.get('segment_order')}` `{patch.get('repair_type')}`: {patch.get('rationale')}",
                ]
            )
        lines.append("")
    empty_segments = record.get("curation", {}).get("empty_subcalls", [])
    if empty_segments:
        lines.extend(["## Empty Subcalls", ""])
        for item in empty_segments:
            lines.append(f"- order `{item.get('order')}`, call `{item.get('call_id')}`, prompt `{item.get('prompt_fingerprint')}`")
        lines.append("")
    lines.extend(["## Sequential Trace", ""])
    for segment in record.get("segments", []):
        if not isinstance(segment, dict):
            continue
        kind = str(segment.get("kind"))
        order = segment.get("order")
        if kind == "plain_query":
            prompt = ""
            request = segment.get("request")
            if isinstance(request, dict):
                prompt = str(request.get("prompt") or request.get("messages") or "")
            response = str(segment.get("response_text", ""))
            lines.extend(
                [
                    f"### {order}: plain llm subcall",
                    "",
                    "**Prompt excerpt**",
                    "",
                    "```text",
                    prompt[:1200],
                    "```",
                    "",
                    "**Response excerpt**",
                    "",
                    "```text",
                    response[:1200] if response.strip() else "<EMPTY>",
                    "```",
                    "",
                ]
            )
        elif bool(segment.get("is_trainable_rlm_turn")):
            curation = segment.get("curation", {}) if isinstance(segment.get("curation"), dict) else {}
            excluded = " excluded_from_sft" if curation.get("exclude_from_sft") else ""
            lines.extend(
                [
                    f"### {order}: {kind}{excluded}",
                    "",
                    "```text",
                    str(segment.get("response_text", "")).strip()[:2500],
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_id",
        "example_id",
        "source_input_path",
        "source_run_id",
        "root_model",
        "curation_version",
        "status",
        "manual_decision_id",
        "tags",
        "rationale",
        "num_llm_subcalls",
        "non_empty_llm_subcall_count",
        "num_trainable_segments",
        "empty_subcall_count",
        "patch_count",
        "final_answer",
        "exact_match",
        "judge_score",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_quality_report(manifest: list[dict[str, Any]], rows: list[dict[str, Any]], patches: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(row["status"] for row in manifest)
    tag_counts: Counter[str] = Counter()
    for row in manifest:
        for tag in str(row["tags"]).split(","):
            if tag:
                tag_counts[tag] += 1
    excluded = [row for row in manifest if row["status"] != "strict_sft"]
    return {
        "num_input_records": len(manifest),
        "status_counts": dict(status_counts),
        "tag_counts": dict(tag_counts),
        "num_sft_rows_strict": len(rows),
        "num_sources_strict": len({row["source_id"] for row in rows}),
        "num_finalization_patches": len(patches),
        "patch_counts_by_repair_type": dict(Counter(str(patch.get("repair_type")) for patch in patches)),
        "empty_subcall_decisions": {
            "recovered": tag_counts.get("empty_subcall_recovered", 0),
            "unrecovered": tag_counts.get("empty_subcall_unrecovered", 0),
            "all_empty": tag_counts.get("all_subcalls_empty", 0),
        },
        "excluded_records": excluded,
    }


def write_quality_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Curated SFT Trace Quality Report",
        "",
        f"- Input records: {report['num_input_records']}",
        f"- Strict SFT records: {report['status_counts'].get('strict_sft', 0)}",
        f"- Audit-only records: {report['status_counts'].get('audit_only', 0)}",
        f"- Strict SFT rows: {report['num_sft_rows_strict']}",
        f"- Strict SFT source IDs: {report['num_sources_strict']}",
        f"- Finalization patches: {report['num_finalization_patches']}",
        "",
        "## Patch Counts",
        "",
    ]
    for repair_type, count in sorted(report.get("patch_counts_by_repair_type", {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{repair_type}`: {count}")
    lines.extend(
        [
            "",
            "## Empty Subcall Decisions",
            "",
        ]
    )
    for key, count in sorted(report.get("empty_subcall_decisions", {}).items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(
        [
            "",
            "## Tag Counts",
            "",
        ]
    )
    for tag, count in sorted(report["tag_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{tag}`: {count}")
    lines.extend(["", "## Excluded Records", ""])
    if not report["excluded_records"]:
        lines.append("None.")
    else:
        for row in report["excluded_records"]:
            lines.append(f"- `{row['source_id']}`: {row['rationale']} ({row['tags']})")
    path.write_text("\n".join(lines).rstrip() + "\n")


def build_manual_audit_queue(manifest: list[dict[str, Any]], patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patched_sources = {str(patch.get("source_id")) for patch in patches}
    queue: list[dict[str, Any]] = []
    for row in manifest:
        tags = set(str(row.get("tags", "")).split(","))
        reasons: list[str] = []
        source_id = str(row.get("source_id"))
        if source_id in patched_sources:
            reasons.append("finalization_repair")
        if int(row.get("empty_subcall_count", 0) or 0) > 0:
            reasons.append("empty_subcall_decision")
        if "literal_final_in_repl_preserved_quoted" in tags:
            reasons.append("quoted_literal_final_preserved")
        if row.get("status") != "strict_sft":
            reasons.append("audit_only_exclusion")
        if "code_plus_final_split" in tags or "missing_final_var_repaired" in tags:
            reasons.append("synthetic_sft_row")
        if reasons:
            queue.append(
                {
                    "decision_id": row.get("manual_decision_id"),
                    "source_id": source_id,
                    "status": row.get("status"),
                    "reasons": sorted(set(reasons)),
                    "tags": row.get("tags"),
                    "rationale": row.get("rationale"),
                    "patch_ids": [patch.get("repair_id") for patch in patches if str(patch.get("source_id")) == source_id],
                }
            )
    return queue


def write_decision_template(path: Path, queue: list[dict[str, Any]]) -> None:
    rows = [
        {
            "decision_id": item["decision_id"],
            "source_id": item["source_id"],
            "decision": "pending",
            "rationale": "",
            "affected_repair_ids": item.get("patch_ids", []),
            "audit_reasons": item.get("reasons", []),
        }
        for item in queue
    ]
    write_jsonl(path, rows)


def load_manual_decisions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    decisions: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        decision_id = str(row.get("decision_id") or row.get("source_id") or "")
        if not decision_id:
            continue
        decisions[decision_id] = row
    return decisions


def apply_manual_decisions(curated: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]) -> None:
    for record in curated:
        curation = record.get("curation")
        if not isinstance(curation, dict):
            continue
        decision = decisions.get(str(curation.get("manual_decision_id")))
        if not decision:
            continue
        curation["manual_audit_decision"] = decision
        curation["manual_audit_decision_id"] = decision.get("decision_id")
        curation["manual_audit_rationale"] = decision.get("rationale")
        decision_value = str(decision.get("decision", "")).strip().lower()
        if decision_value == "exclude":
            curation["status"] = "audit_only"
            curation.setdefault("tags", [])
            if "manually_excluded" not in curation["tags"]:
                curation["tags"].append("manually_excluded")
            curation["rationale"] = (curation.get("rationale", "") + "; manually excluded: " + str(decision.get("rationale", ""))).strip("; ")


def validate_manual_decisions(queue: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    allowed = {"approve", "exclude"}
    for item in queue:
        decision_id = str(item.get("decision_id"))
        decision = decisions.get(decision_id)
        if not decision:
            errors.append(f"missing manual decision for {decision_id}")
            continue
        value = str(decision.get("decision", "")).strip().lower()
        if value not in allowed:
            errors.append(f"invalid manual decision for {decision_id}: {value!r}")
        if not str(decision.get("rationale", "")).strip():
            errors.append(f"missing manual rationale for {decision_id}")
    return errors


def validate_strict_rows(curated: list[dict[str, Any]], rows: list[dict[str, Any]], queue: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    records_by_source = {str(record.get("source_id")): record for record in curated}
    required_decision_ids = {str(item.get("decision_id")) for item in queue}
    for row in rows:
        source_id = str(row.get("source_id"))
        record = records_by_source.get(source_id)
        if record is None:
            errors.append(f"strict row {row.get('row_id')} has no source record")
            continue
        if not row.get("prompt"):
            errors.append(f"strict row {row.get('row_id')} has missing prompt")
        completion = row.get("completion")
        if not isinstance(completion, list) or not completion or not str(completion[0].get("content", "")).strip():
            errors.append(f"strict row {row.get('row_id')} has empty completion")
        tags = set(record.get("curation", {}).get("tags", []))
        if not non_empty_plain_subcall_segments(record):
            errors.append(f"strict row {row.get('row_id')} source has zero non-empty subcalls")
        if "empty_subcall_unrecovered" in tags:
            errors.append(f"strict row {row.get('row_id')} source has unrecovered empty subcall")
        decision_id = str(record.get("curation", {}).get("manual_decision_id"))
        if decision_id in required_decision_ids and decision_id not in decisions:
            errors.append(f"strict row {row.get('row_id')} source lacks required manual decision")
    return errors


def write_manual_review_markdown(path: Path, manifest: list[dict[str, Any]], patches: list[dict[str, Any]]) -> None:
    patch_counts = Counter(str(patch["repair_type"]) for patch in patches)
    rows_by_source = {str(row["source_id"]): row for row in manifest}
    lines = [
        "# Manual Curation Notes",
        "",
        "I used the curator script as an audit scaffold, then reviewed the flagged trajectory classes directly.",
        "The curation does not invent missing subcall responses. Empty subcall text remains empty in `records.curated.jsonl`.",
        "",
        "## Finalization Decisions",
        "",
        f"- Bare-answer loops repaired: {patch_counts.get('bare_answer_loop', 0)}. I kept the verified trajectory up to the first correct bare answer, rewrote that answer as `FINAL(actual_answer)`, and excluded repeated forced-finalization turns from SFT rows.",
        f"- Literal FINAL inside REPL repaired: {patch_counts.get('literal_final_in_repl', 0)}. Literal `FINAL(...)` is parsed as text outside executable code, so these are converted to final-only text.",
        f"- Quoted literal FINAL inside REPL preserved: {sum(1 for row in manifest if 'literal_final_in_repl_preserved_quoted' in str(row.get('tags', '')))}. These stored final answers include quotes, so the original final turn is kept rather than converted to an outside-REPL literal target with quotes.",
        f"- FINAL_VAR moved into REPL: {patch_counts.get('final_var_outside_repl', 0)}. `FINAL_VAR` is a Python REPL helper, so variable-final repairs prefer a small REPL block.",
        f"- FINAL(identifier) repaired: {patch_counts.get('final_identifier_to_final_var', 0) + patch_counts.get('final_identifier_to_literal', 0)}. Identifier finals are repaired to `FINAL_VAR` when the variable is clear, otherwise to the stored literal accepted answer.",
        f"- Code plus final split: {patch_counts.get('code_plus_final_split', 0)}. The analysis code is kept as one SFT row, and the final answer is moved to a synthetic final row after the already-recorded REPL feedback.",
        "",
        "## Empty Subcall Decisions",
        "",
    ]
    for source_id, decision in sorted(MANUAL_EMPTY_SUBCALL_DECISIONS.items()):
        row = rows_by_source.get(source_id, {})
        verb = "kept as" if row.get("status") == "strict_sft" else "marked"
        lines.append(
            f"- `{source_id}`: {verb} `{decision['status']}` via `{decision['recovery']}`. "
            f"{decision['rationale']} Status: `{row.get('status', 'missing')}`."
        )
    if MANUAL_AUDIT_ONLY_SOURCE_DECISIONS:
        lines.extend(["", "## Weak Subcall Exclusions", ""])
        for source_id, rationale in sorted(MANUAL_AUDIT_ONLY_SOURCE_DECISIONS.items()):
            row = rows_by_source.get(source_id, {})
            lines.append(f"- `{source_id}`: marked `audit_only`. {rationale} Status: `{row.get('status', 'missing')}`.")
    lines.extend(
        [
            "",
            "## Overdelegation",
            "",
            "High-subcall traces are tagged but kept eligible for strict SFT. They still show delegation behavior, and efficiency can be shaped later during RL.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_audit_samples(output_dir: Path, curated: list[dict[str, Any]], patches: list[dict[str, Any]]) -> None:
    sample_dir = output_dir / "audit_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for stale_sample in sample_dir.glob("*.md"):
        stale_sample.unlink()
    patches_by_source: dict[str, list[dict[str, Any]]] = {}
    for patch in patches:
        patches_by_source.setdefault(str(patch.get("source_id")), []).append(patch)

    wanted = {
        "bare_answer_loop_repaired": None,
        "literal_final_in_repl_repaired": None,
        "code_plus_final_split": None,
        "empty_subcall_recovered": None,
        "audit_only": None,
        "overdelegated": None,
    }
    for record in curated:
        tags = set(record.get("curation", {}).get("tags", []))
        for key in list(wanted):
            if wanted[key] is None and (key in tags or record.get("curation", {}).get("status") == key):
                wanted[key] = record
    for key, record in wanted.items():
        if record is None:
            continue
        source_id = str(record.get("source_id"))
        (sample_dir / f"{key}__{safe_filename(source_id)}.md").write_text(
            render_audit_sample(record, patches_by_source.get(source_id, []))
        )


def write_audit_pages(output_dir: Path, curated: list[dict[str, Any]], patches: list[dict[str, Any]]) -> None:
    page_dir = output_dir / "audit_pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    for stale_page in page_dir.glob("*.md"):
        stale_page.unlink()
    patches_by_source: dict[str, list[dict[str, Any]]] = {}
    for patch in patches:
        patches_by_source.setdefault(str(patch.get("source_id")), []).append(patch)
    for record in curated:
        source_id = str(record.get("source_id"))
        (page_dir / f"{safe_filename(source_id)}.md").write_text(render_compact_audit_page(record, patches_by_source.get(source_id, [])))


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120]


def curate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    curated_records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    patches: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    for record in records:
        curated, manifest_row, record_patches = curate_record(record)
        curated_records.append(curated)
        manifest.append(manifest_row)
        patches.extend(record_patches)
        strict_rows.extend(sft_rows_for_record(curated))
    return curated_records, manifest, patches, strict_rows


def main() -> None:
    args = parse_args()
    input_paths = args.inputs or DEFAULT_V2_INPUTS
    input_hashes_before = {str(path): sha256_file(path) for path in input_paths}
    records_with_provenance, input_manifest = read_records_with_provenance(input_paths)
    records, duplicate_records = dedupe_records_by_source_id(records_with_provenance)
    curated, manifest, patches, strict_rows = curate_records(records)
    broad_rows = list(strict_rows)
    audit_queue = build_manual_audit_queue(manifest, patches)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "records.candidate.jsonl", curated)
    write_jsonl(args.output_dir / "manual_patches.candidate.jsonl", patches)
    write_jsonl(args.output_dir / "manual_audit_queue.jsonl", audit_queue)
    write_decision_template(args.output_dir / "manual_review.decisions.template.jsonl", audit_queue)
    write_jsonl(args.output_dir / "duplicate_records.jsonl", duplicate_records)
    write_audit_pages(args.output_dir, curated, patches)
    write_audit_samples(args.output_dir, curated, patches)

    decisions = load_manual_decisions(args.manual_decisions)
    decision_errors: list[str] = []
    validation_errors: list[str] = []
    final_export_written = False
    if args.manual_decisions is not None and not args.candidate_only:
        decision_errors = validate_manual_decisions(audit_queue, decisions)
        if decision_errors:
            write_json(args.output_dir / "validation_errors.json", {"manual_decision_errors": decision_errors})
            raise SystemExit("Manual decision validation failed; see validation_errors.json")
        apply_manual_decisions(curated, decisions)
        strict_rows = []
        for record in curated:
            strict_rows.extend(sft_rows_for_record(record))
        broad_rows = list(strict_rows)
        validation_errors = validate_strict_rows(curated, strict_rows, audit_queue, decisions)
        if validation_errors:
            write_json(args.output_dir / "validation_errors.json", {"strict_row_errors": validation_errors})
            raise SystemExit("Strict row validation failed; see validation_errors.json")
        write_jsonl(args.output_dir / "records.curated.jsonl", curated)
        write_jsonl(args.output_dir / "sft_rows.strict.jsonl", strict_rows)
        write_jsonl(args.output_dir / "sft_rows.broad.jsonl", broad_rows)
        write_jsonl(args.output_dir / "manual_patches.jsonl", patches)
        write_jsonl(args.output_dir / "manual_review.decisions.jsonl", list(decisions.values()))
        final_export_written = True

    write_jsonl(args.output_dir / "curation_manifest.jsonl", manifest)
    write_json(args.output_dir / "curation_manifest.json", {"records": manifest})
    write_manifest_csv(args.output_dir / "curation_manifest.csv", manifest)
    report = build_quality_report(manifest, strict_rows, patches)
    for row in input_manifest:
        after_hash = sha256_file(Path(row["path"]))
        row["sha256_after"] = after_hash
        row["unchanged"] = after_hash == row["sha256_before"] == input_hashes_before[row["path"]]
    report["curation_version"] = CURATION_VERSION
    report["input_files"] = input_manifest
    report["num_unique_records_after_dedupe"] = len(records)
    report["num_duplicate_records_dropped"] = len(duplicate_records)
    report["manual_audit_queue_count"] = len(audit_queue)
    report["manual_decisions_supplied"] = len(decisions)
    report["final_export_written"] = final_export_written
    report["original_inputs_unchanged"] = all(row["unchanged"] for row in input_manifest)
    if not final_export_written:
        report["candidate_only_reason"] = (
            "Run again with --manual-decisions after Codex/manual audit to write records.curated.jsonl and strict SFT rows."
        )
    write_json(args.output_dir / "quality_report.json", report)
    write_quality_markdown(args.output_dir / "quality_report.md", report)
    write_manual_review_markdown(args.output_dir / "manual_review.md", manifest, patches)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
