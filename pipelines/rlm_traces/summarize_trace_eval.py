from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_VARIANT = "sanjaya_text_depth1_llm_only_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a single RLM trace eval records.jsonl file.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--records", type=Path, help="Path to records.jsonl.")
    source.add_argument("--run-dir", type=Path, help="Trace run directory containing the prompt-variant subdir.")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, help="Prompt variant directory name when using --run-dir.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults beside records.jsonl.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def segment_token_count(segment: dict[str, Any], key: str, fallback_key: str) -> int:
    value = segment.get(key)
    if value is None:
        value = segment.get(fallback_key)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def segments(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("segments") or record.get("rlm_segments")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [segment for segment in value if isinstance(segment, dict)]


def is_error_like_subcall_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    markers = (
        "NOT_FOUND",
        "RESOURCE_EXHAUSTED",
        "RATE_LIMIT",
        "429",
        "500",
        "503",
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
        "DEADLINE_EXCEEDED",
    )
    return any(marker in stripped for marker in markers)


def segment_stats(record: dict[str, Any]) -> dict[str, float]:
    prompt_tokens = 0
    completion_tokens = 0
    trainable_tokens = 0
    plain_subcall_tokens = 0
    plain_subcalls = 0
    empty_plain_subcalls = 0
    error_like_plain_subcalls = 0

    for segment in segments(record):
        segment_prompt_tokens = segment_token_count(segment, "prompt_token_count", "prompt_tokens")
        segment_completion_tokens = segment_token_count(segment, "completion_token_count", "completion_tokens")
        segment_total = segment_prompt_tokens + segment_completion_tokens
        prompt_tokens += segment_prompt_tokens
        completion_tokens += segment_completion_tokens
        if segment.get("is_trainable_rlm_turn"):
            trainable_tokens += segment_total
        if segment.get("kind") == "plain_query":
            plain_subcalls += 1
            plain_subcall_tokens += segment_total
            response_text = str(segment.get("response_text") or "")
            if not response_text.strip():
                empty_plain_subcalls += 1
            if is_error_like_subcall_text(response_text):
                error_like_plain_subcalls += 1

    return {
        "segment_prompt_tokens": float(prompt_tokens),
        "segment_completion_tokens": float(completion_tokens),
        "segment_total_tokens": float(prompt_tokens + completion_tokens),
        "segment_trainable_tokens": float(trainable_tokens),
        "segment_plain_subcall_tokens": float(plain_subcall_tokens),
        "segment_plain_subcalls": float(plain_subcalls),
        "empty_plain_subcalls": float(empty_plain_subcalls),
        "error_like_plain_subcalls": float(error_like_plain_subcalls),
    }


def record_is_correct(record: dict[str, Any]) -> bool:
    return not record.get("error") and (bool(record.get("exact_match")) or number(record.get("judge_score"), math.nan) == 1.0)


def record_num_llm_subcalls(record: dict[str, Any], stats: dict[str, float]) -> int:
    value = record.get("num_llm_subcalls")
    if value is None:
        value = record.get("num_subcalls")
    try:
        return int(value or stats["segment_plain_subcalls"])
    except (TypeError, ValueError):
        return int(stats["segment_plain_subcalls"])


def record_num_rlm_subcalls(record: dict[str, Any]) -> int:
    value = record.get("num_rlm_subcalls")
    if value is None:
        value = record.get("rlm_subcall_count")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_record: list[dict[str, Any]] = []
    for record in records:
        stats = segment_stats(record)
        num_llm_subcalls = record_num_llm_subcalls(record, stats)
        cost_total_tokens = number(
            record.get("total_rollout_tokens")
            or record.get("total_model_tokens")
            or record.get("cost_total_tokens"),
            default=stats["segment_total_tokens"],
        )
        cost_plain_subcall_tokens = number(record.get("cost_plain_subcall_tokens"), default=stats["segment_plain_subcall_tokens"])
        cost_trainable_tokens = number(record.get("cost_trainable_tokens"), default=stats["segment_trainable_tokens"])
        num_rlm_subcalls = record_num_rlm_subcalls(record)
        per_record.append(
            {
                "source_id": str(record.get("source_id", record.get("example_id", ""))),
                "error": bool(record.get("error")),
                "has_final_answer": bool(str(record.get("final_answer") or "").strip()),
                "exact_match": bool(record.get("exact_match")),
                "judge_correct": number(record.get("judge_score"), math.nan) == 1.0,
                "correct": record_is_correct(record),
                "num_llm_subcalls": num_llm_subcalls,
                "used_llm_subcalls": num_llm_subcalls > 0,
                "num_rlm_subcalls": num_rlm_subcalls,
                "used_rlm_subcalls": num_rlm_subcalls > 0,
                "total_tokens": cost_total_tokens,
                "plain_subcall_tokens": cost_plain_subcall_tokens,
                "trainable_tokens": cost_trainable_tokens,
                **stats,
            }
        )

    total = len(per_record)
    subcall_total = int(sum(row["num_llm_subcalls"] for row in per_record))
    empty_subcalls = int(sum(row["empty_plain_subcalls"] for row in per_record))
    error_like_subcalls = int(sum(row["error_like_plain_subcalls"] for row in per_record))
    summary = {
        "num_records": total,
        "pass_at_1": rate(sum(1 for row in per_record if row["correct"]), total),
        "exact_match_rate": rate(sum(1 for row in per_record if row["exact_match"]), total),
        "judge_correct_rate": rate(sum(1 for row in per_record if row["judge_correct"]), total),
        "error_rate": rate(sum(1 for row in per_record if row["error"]), total),
        "no_final_rate": rate(sum(1 for row in per_record if not row["has_final_answer"]), total),
        "no_final_or_error_rate": rate(sum(1 for row in per_record if row["error"] or not row["has_final_answer"]), total),
        "mean_llm_subcalls": mean([float(row["num_llm_subcalls"]) for row in per_record]),
        "llm_subcall_usage_rate": rate(sum(1 for row in per_record if row["used_llm_subcalls"]), total),
        "used_rlm_subcalls_rate": rate(sum(1 for row in per_record if row["used_rlm_subcalls"]), total),
        "empty_subcall_rate": float(empty_subcalls / subcall_total) if subcall_total else 0.0,
        "error_like_subcall_rate": float(error_like_subcalls / subcall_total) if subcall_total else 0.0,
        "total_llm_subcalls": subcall_total,
        "empty_plain_subcalls": empty_subcalls,
        "error_like_plain_subcalls": error_like_subcalls,
        "mean_total_tokens": mean([float(row["total_tokens"]) for row in per_record]),
        "mean_plain_subcall_tokens": mean([float(row["plain_subcall_tokens"]) for row in per_record]),
        "mean_trainable_tokens": mean([float(row["trainable_tokens"]) for row in per_record]),
    }
    return {"summary": summary, "per_record": per_record}


def write_outputs(payload: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    per_record = payload["per_record"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (out_dir / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    if per_record:
        with (out_dir / "per_record.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_record[0].keys()))
            writer.writeheader()
            writer.writerows(per_record)

    lines = [
        "# GPT-5.4 RLM Trace Eval Summary",
        "",
        f"- Records: `{summary['num_records']}`",
        f"- Pass@1 exact-or-judge: `{summary['pass_at_1']:.4f}`",
        f"- Exact match rate: `{summary['exact_match_rate']:.4f}`",
        f"- Judge-correct rate: `{summary['judge_correct_rate']:.4f}`",
        f"- No-final/error rate: `{summary['no_final_or_error_rate']:.4f}`",
        f"- Mean LLM subcalls: `{summary['mean_llm_subcalls']:.2f}`",
        f"- LLM-subcall usage rate: `{summary['llm_subcall_usage_rate']:.4f}`",
        f"- Empty subcall rate: `{summary['empty_subcall_rate']:.4f}`",
        f"- Error-like subcall rate: `{summary['error_like_subcall_rate']:.4f}`",
        f"- Used recursive RLM subcalls rate: `{summary['used_rlm_subcalls_rate']:.4f}`",
        f"- Mean total tokens: `{summary['mean_total_tokens']:.1f}`",
        f"- Mean plain-subcall tokens: `{summary['mean_plain_subcall_tokens']:.1f}`",
        f"- Mean trainable/root tokens: `{summary['mean_trainable_tokens']:.1f}`",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    records_path = args.records if args.records is not None else args.run_dir / args.variant / "records.jsonl"
    out_dir = args.out_dir or records_path.parent / "eval_summary"
    payload = summarize_records(read_jsonl(records_path))
    write_outputs(payload, out_dir)
    print((out_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
