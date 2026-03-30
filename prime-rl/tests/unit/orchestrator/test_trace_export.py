from __future__ import annotations

import json
from pathlib import Path

from prime_rl.orchestrator.trace_export import export_rollout_traces


def test_export_rollout_traces_includes_dataset_answer(tmp_path: Path) -> None:
    trace_path = export_rollout_traces(
        rollouts=[
            {
                "example_id": 7,
                "task": "rlm_rlvr",
                "answer": "ground truth answer",
                "reward": 0.01,
                "error": None,
                "final_answer": "model answer",
                "is_truncated": False,
                "stop_condition": "completed",
                "sampling_args": {"max_tokens": 2048},
                "timing": {},
                "rlm_trace": [],
                "metrics": {},
                "rlm_segments": [],
            }
        ],
        step=2,
        output_dir=tmp_path,
    )

    payload = json.loads(trace_path.read_text())

    assert payload["rollouts"][0]["answer"] == "ground truth answer"
    assert payload["rollouts"][0]["rlm_answer"] == "model answer"


def test_export_rollout_traces_falls_back_to_completion_for_rlm_answer(tmp_path: Path) -> None:
    trace_path = export_rollout_traces(
        rollouts=[
            {
                "example_id": 8,
                "task": "rlm_rlvr",
                "answer": "expected answer",
                "reward": 0.0,
                "error": None,
                "final_answer": None,
                "completion": [{"role": "assistant", "content": "partial model answer"}],
                "is_truncated": False,
                "stop_condition": "completed",
                "sampling_args": {},
                "timing": {},
                "rlm_trace": [],
                "metrics": {},
                "rlm_segments": [],
            }
        ],
        step=3,
        output_dir=tmp_path,
    )

    payload = json.loads(trace_path.read_text())

    assert payload["rollouts"][0]["rlm_answer"] == "partial model answer"


def test_export_rollout_traces_includes_judge_debug_payload(tmp_path: Path) -> None:
    trace_path = export_rollout_traces(
        rollouts=[
            {
                "example_id": 9,
                "task": "rlm_rlvr",
                "answer": "ground truth answer",
                "reward": 1.0,
                "error": None,
                "final_answer": "model answer",
                "completion": [{"role": "assistant", "content": "model answer"}],
                "trajectory": [
                    {
                        "extras": {
                            "rlm_debug": {
                                "expected_answers": ["ground truth answer", "alias"],
                                "judge_score": 1.0,
                                "judge_raw_response": "1",
                                "judge_parse_error": None,
                            }
                        }
                    }
                ],
                "is_truncated": False,
                "stop_condition": "completed",
                "sampling_args": {},
                "timing": {},
                "rlm_trace": [],
                "metrics": {},
                "rlm_segments": [],
            }
        ],
        step=4,
        output_dir=tmp_path,
    )

    payload = json.loads(trace_path.read_text())

    assert payload["rollouts"][0]["expected_answers"] == ["ground truth answer", "alias"]
    assert payload["rollouts"][0]["judge_score"] == 1.0
    assert payload["rollouts"][0]["judge_raw_response"] == "1"
    assert payload["rollouts"][0]["judge_parse_error"] is None
