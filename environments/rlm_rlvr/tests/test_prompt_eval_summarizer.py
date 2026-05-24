from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_summarizer_module():
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "summarize_rlm_prompt_eval.py"
    spec = importlib.util.spec_from_file_location("summarize_rlm_prompt_eval", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_token_totals_support_trace_export_counts() -> None:
    summarizer = _load_summarizer_module()

    prompt_tokens, completion_tokens = summarizer.token_totals(
        {
            "segments": [
                {
                    "prompt_token_count": 5,
                    "completion_token_count": 10,
                    "trainable_token_count": 3,
                }
            ]
        }
    )

    assert prompt_tokens == 5
    assert completion_tokens == 3


def test_token_totals_support_live_trace_counts() -> None:
    summarizer = _load_summarizer_module()

    prompt_tokens, completion_tokens = summarizer.token_totals(
        {
            "segments": [
                {
                    "prompt_tokens": 7,
                    "completion_tokens": 9,
                    "trainable_completion_tokens": 4,
                }
            ]
        }
    )

    assert prompt_tokens == 7
    assert completion_tokens == 4


def test_token_totals_support_raw_segment_arrays() -> None:
    summarizer = _load_summarizer_module()

    prompt_tokens, completion_tokens = summarizer.token_totals(
        {
            "rlm_segments": [
                {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 4, 5],
                    "completion_mask": [True, False, True],
                }
            ]
        }
    )

    assert prompt_tokens == 2
    assert completion_tokens == 2


def test_live_trace_directory_loads_sample_status_and_tokens(tmp_path) -> None:
    summarizer = _load_summarizer_module()
    trace_dir = tmp_path / "default"
    trace_dir.mkdir()
    trace_path = trace_dir / "sample-1.json"
    trace_path.write_text(
        json.dumps(
            {
                "prompt_variant": "default",
                "sample": {"source_id": "sample-1"},
                "status": {
                    "used_recursion": True,
                    "used_llm_subcalls": True,
                    "used_rlm_subcalls": False,
                    "num_subcalls": 2,
                    "num_llm_subcalls": 2,
                    "num_rlm_subcalls": 0,
                    "max_depth_reached": 1,
                },
                "segments": [{"prompt_tokens": 10, "completion_tokens": 3}],
            }
        )
    )

    frame = summarizer.load_run("default", trace_dir)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["source_id"] == "sample-1"
    assert row["used_recursion"] == 1.0
    assert row["used_llm_subcalls"] == 1.0
    assert row["used_rlm_subcalls"] == 0.0
    assert row["num_subcalls"] == 2.0
    assert row["num_llm_subcalls"] == 2.0
    assert row["num_rlm_subcalls"] == 0.0
    assert row["max_depth_reached"] == 1.0
    assert row["prompt_tokens"] == 10.0
    assert row["completion_tokens"] == 3.0


def test_jsonl_table_loads_tokens_from_serialized_segments(tmp_path) -> None:
    summarizer = _load_summarizer_module()
    result_path = tmp_path / "results.jsonl"
    pd.DataFrame(
        [
            {
                "source_id": "sample-1",
                "reward": 1.0,
                "used_recursion": False,
                "used_llm_subcalls": 0,
                "used_rlm_subcalls": 0,
                "num_subcalls": 0,
                "num_llm_subcalls": 0,
                "num_rlm_subcalls": 0,
                "max_depth_reached": 0,
                "rlm_segments": json.dumps(
                    [
                        {
                            "prompt_ids": [1, 2, 3],
                            "completion_ids": [4, 5],
                            "completion_mask": [True, False],
                        }
                    ]
                ),
            }
        ]
    ).to_json(result_path, orient="records", lines=True)

    frame = summarizer.load_run("default", result_path)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["prompt_tokens"] == 3.0
    assert row["completion_tokens"] == 1.0
