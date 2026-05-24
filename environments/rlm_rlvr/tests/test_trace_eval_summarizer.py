from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.rlm_traces import summarize_trace_eval


def test_trace_eval_summarizer_counts_correctness_subcalls_and_tokens(tmp_path: Path) -> None:
    records = [
        {
            "source_id": "exact",
            "error": None,
            "final_answer": "42",
            "exact_match": True,
            "judge_score": None,
            "num_llm_subcalls": 1,
            "used_recursion": False,
            "total_model_tokens": 100,
            "segments": [
                {
                    "kind": "root",
                    "is_trainable_rlm_turn": True,
                    "prompt_token_count": 20,
                    "completion_token_count": 10,
                },
                {
                    "kind": "plain_query",
                    "prompt_token_count": 50,
                    "completion_token_count": 20,
                    "response_text": "useful evidence",
                },
            ],
        },
        {
            "source_id": "judge",
            "error": None,
            "final_answer": "forty two",
            "exact_match": False,
            "judge_score": 1.0,
            "num_llm_subcalls": 2,
            "used_recursion": False,
            "segments": [
                {
                    "kind": "plain_query",
                    "prompt_token_count": 10,
                    "completion_token_count": 0,
                    "response_text": "",
                },
                {
                    "kind": "plain_query",
                    "prompt_token_count": 15,
                    "completion_token_count": 1,
                    "response_text": "NOT_FOUND",
                },
            ],
        },
        {
            "source_id": "bad",
            "error": "RuntimeError: boom",
            "final_answer": "",
            "exact_match": False,
            "judge_score": None,
            "num_llm_subcalls": 0,
            "num_rlm_subcalls": 1,
            "segments": [],
        },
    ]
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    payload = summarize_trace_eval.summarize_records(summarize_trace_eval.read_jsonl(records_path))
    summary = payload["summary"]

    assert summary["num_records"] == 3
    assert summary["pass_at_1"] == 2 / 3
    assert summary["exact_match_rate"] == 1 / 3
    assert summary["judge_correct_rate"] == 1 / 3
    assert summary["no_final_or_error_rate"] == 1 / 3
    assert summary["mean_llm_subcalls"] == 1.0
    assert summary["llm_subcall_usage_rate"] == 2 / 3
    assert summary["used_rlm_subcalls_rate"] == 1 / 3
    assert summary["empty_subcall_rate"] == 1 / 3
    assert summary["error_like_subcall_rate"] == 2 / 3
    assert summary["mean_total_tokens"] == (100 + 26 + 0) / 3


def test_trace_eval_summarizer_writes_expected_files(tmp_path: Path) -> None:
    payload = summarize_trace_eval.summarize_records(
        [
            {
                "source_id": "a",
                "error": None,
                "final_answer": "1",
                "exact_match": True,
                "judge_score": None,
                "num_llm_subcalls": 1,
                "segments": [
                    {
                        "kind": "plain_query",
                        "prompt_token_count": 3,
                        "completion_token_count": 2,
                        "response_text": "ok",
                    }
                ],
            }
        ]
    )

    summarize_trace_eval.write_outputs(payload, tmp_path)

    assert json.loads((tmp_path / "summary.json").read_text())["pass_at_1"] == 1.0
    assert (tmp_path / "aggregate.csv").exists()
    assert (tmp_path / "per_record.csv").exists()
    assert "Pass@1" in (tmp_path / "report.md").read_text()
