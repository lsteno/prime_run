from __future__ import annotations

import argparse
import copy
import csv
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


DEFAULT_INPUT = Path("outputs/rlm_traces/combined-good-sft-traces-v1/records.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/rlm_traces/combined-good-sft-traces-v1/curated-v1")
HIGH_SUBCALL_THRESHOLD = 20

FENCE_RE = re.compile(r"```([A-Za-z0-9_-]*)\n(.*?)```", re.DOTALL)
FINAL_RE = re.compile(r"\bFINAL(?:_VAR)?\s*\(", re.DOTALL)


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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
    return bool(FINAL_RE.search(remove_code_blocks(text)))


def has_final_inside_repl(text: str) -> bool:
    for language, body in code_blocks(text):
        if language in {"repl", "python", ""} and FINAL_RE.search(body):
            return True
    return False


def has_repl_code(text: str) -> bool:
    return any(language in {"repl", "python", ""} for language, _ in code_blocks(text))


def is_expression_final(text: str) -> bool:
    outside = remove_code_blocks(text)
    return bool(re.search(r"\bFINAL\s*\(\s*(?:json\.dumps|str|repr)\s*\(", outside))


def canonical_final_text(record: dict[str, Any]) -> str:
    final_answer = str(record.get("final_answer") or record.get("answer") or "").strip()
    if FINAL_RE.search(final_answer):
        return final_answer
    return f"FINAL({final_answer})"


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

    tags.append("empty_subcall_recovered")
    return tags, empty_segments, "empty subcall is non-critical or followed by later evidence/retry"


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
        outside = FINAL_RE.split(outside, maxsplit=1)[0]
        parts.append(outside.rstrip())
        parts.append(match.group(0).strip())
        last = match.end()
    outside = text[last:]
    outside = FINAL_RE.split(outside, maxsplit=1)[0]
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
        patches.append(
            {
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
        segment["curation"]["old_response_text"] = old_response

    for segment in segments:
        response = str(segment.get("response_text", ""))
        if is_expression_final(response):
            tags.append("expression_final_repaired")
            patch_segment(
                segment,
                final_text,
                "expression_final",
                "FINAL contained an unevaluated expression; replaced with stored accepted final answer.",
            )
        elif has_final_inside_repl(response) and not has_final_outside_code(response):
            tags.append("final_in_repl_repaired")
            patch_segment(
                segment,
                final_text,
                "final_in_repl",
                "Moved/collapsed FINAL from executable code into a final-only assistant response.",
            )
        elif has_repl_code(response) and has_final_outside_code(response):
            analysis_response = strip_final_outside_code(response)
            feedback = feedback_messages_for_response(record, response)
            synthetic_prompt = export_sft.sanitize_prompt_messages(segment.get("prompt_messages", []))
            if analysis_response and feedback:
                synthetic_prompt = synthetic_prompt + [{"role": "assistant", "content": analysis_response}] + feedback
                segment["_curation_sft_rows"] = [
                    {
                        "response_text": analysis_response,
                        "prompt_messages": export_sft.sanitize_prompt_messages(segment.get("prompt_messages", [])),
                        "row_role": "analysis_before_final",
                    },
                    {
                        "response_text": final_text,
                        "prompt_messages": synthetic_prompt,
                        "row_role": "synthetic_final_after_feedback",
                    },
                ]
                segment.setdefault("curation", {})["repair_type"] = "code_plus_final_split"
                tags.append("code_plus_final_split")
                patches.append(
                    {
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
                patch_segment(
                    segment,
                    final_text,
                    "code_plus_final_collapse",
                    "Could not recover feedback for a clean split; kept only the stored accepted final answer.",
                )

    forced_finalize = any(bool(step.get("forced_finalize")) for step in record.get("root_steps", []) if isinstance(step, dict))
    final_wrapped_segments = [segment for segment in segments if has_final_outside_code(str(segment.get("response_text", "")))]
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
    if not any(has_final_outside_code(str(segment.get("response_text", ""))) for segment in segments):
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
            tags.append("unrepairable_finalization")
            exclusions.append("no clean FINAL wrapper and last response is not safely wrappable")

    return sorted(set(tags)), patches, exclusions


def strict_sft_status(record: dict[str, Any], tags: list[str], exclusions: list[str]) -> tuple[bool, str]:
    if not export_sft.record_is_accepted(record):
        return False, "not accepted by exact match or judge"
    if int(record.get("num_llm_subcalls", 0) or 0) < 1:
        return False, "does not use any plain LLM subcall"
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
    if strict:
        tags.append("strict_sft")
    else:
        tags.append("audit_only")

    tags = sorted(set(tags))
    curation = {
        "status": "strict_sft" if strict else "audit_only",
        "tags": tags,
        "rationale": "; ".join(rationale + [status_reason]),
        "empty_subcalls": empty_segments,
        "patch_count": len(patches),
    }
    curated["curation"] = curation
    manifest = {
        "source_id": curated.get("source_id"),
        "example_id": curated.get("example_id"),
        "source_run_id": curated.get("_source_run_id"),
        "root_model": curated.get("_root_model") or curated.get("endpoint", {}).get("model"),
        "status": curation["status"],
        "tags": ",".join(tags),
        "rationale": curation["rationale"],
        "num_llm_subcalls": int(curated.get("num_llm_subcalls", 0) or 0),
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
    return {
        "row_id": row_id,
        "source_id": source_id,
        "example_id": record.get("example_id"),
        "segment_order": segment_order,
        "segment_kind": str(segment.get("kind")),
        "row_role": row_role,
        "prompt": export_sft.sanitize_prompt_messages(prompt_messages),
        "completion": [{"role": "assistant", "content": response_text}],
        "final_answer": str(record.get("final_answer", "")),
        "exact_match": bool(record.get("exact_match")),
        "judge_score": record.get("judge_score"),
        "prompt_token_count": int(segment.get("prompt_token_count", 0) or 0),
        "completion_token_count": int(segment.get("completion_token_count", 0) or 0),
        "curation_tags": record.get("curation", {}).get("tags", []),
    }


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


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_id",
        "example_id",
        "source_run_id",
        "root_model",
        "status",
        "tags",
        "rationale",
        "num_llm_subcalls",
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
        "## Tag Counts",
        "",
    ]
    for tag, count in sorted(report["tag_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{tag}`: {count}")
    lines.extend(["", "## Excluded Records", ""])
    if not report["excluded_records"]:
        lines.append("None.")
    else:
        for row in report["excluded_records"]:
            lines.append(f"- `{row['source_id']}`: {row['rationale']} ({row['tags']})")
    path.write_text("\n".join(lines).rstrip() + "\n")


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
        f"- FINAL inside REPL repaired: {patch_counts.get('final_in_repl', 0)}. These were protocol mistakes after the answer had been reached; I converted the final turn to final-only text.",
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
        "final_in_repl_repaired": None,
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
    original_bytes = args.input.read_bytes()
    records = read_jsonl(args.input)
    curated, manifest, patches, strict_rows = curate_records(records)
    broad_rows = list(strict_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "records.curated.jsonl", curated)
    write_jsonl(args.output_dir / "sft_rows.strict.jsonl", strict_rows)
    write_jsonl(args.output_dir / "sft_rows.broad.jsonl", broad_rows)
    write_jsonl(args.output_dir / "manual_patches.jsonl", patches)
    write_jsonl(args.output_dir / "curation_manifest.jsonl", manifest)
    write_json(args.output_dir / "curation_manifest.json", {"records": manifest})
    write_manifest_csv(args.output_dir / "curation_manifest.csv", manifest)
    report = build_quality_report(manifest, strict_rows, patches)
    report["original_input_unchanged"] = args.input.read_bytes() == original_bytes
    write_json(args.output_dir / "quality_report.json", report)
    write_quality_markdown(args.output_dir / "quality_report.md", report)
    write_manual_review_markdown(args.output_dir / "manual_review.md", manifest, patches)
    write_audit_samples(args.output_dir, curated, patches)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
