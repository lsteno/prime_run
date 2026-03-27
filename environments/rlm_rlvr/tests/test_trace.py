from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlm_rlvr.trace import build_recursive_trace, segment_from_trajectory_step


def test_segment_from_trajectory_step_preserves_training_fields() -> None:
    step = {
        "completion": [{"role": "assistant", "content": "done"}],
        "tokens": {
            "prompt_ids": [1, 2],
            "completion_ids": [3, 4],
            "completion_logprobs": [-0.1, -0.2],
            "completion_mask": [True, False],
        },
    }

    segment = segment_from_trajectory_step(
        step,
        order=7,
        depth=2,
        kind="recursive_turn",
        default_temperature=0.7,
    )

    assert segment is not None
    assert segment["order"] == 7
    assert segment["depth"] == 2
    assert segment["kind"] == "recursive_turn"
    assert segment["prompt_ids"] == [1, 2]
    assert segment["completion_ids"] == [3, 4]
    assert segment["completion_logprobs"] == [-0.1, -0.2]
    assert segment["completion_mask"] == [True, False]
    assert segment["temperature"] == 0.7
    assert segment["response_text"] == "done"


def test_build_recursive_trace_normalizes_metadata() -> None:
    trace = build_recursive_trace(
        [
            {
                "call_id": 3,
                "depth": 2,
                "prompt": "summarize file A",
                "response": "summary",
                "parent_turn": 1,
                "remaining_depth": 0,
                "batch_id": "b1",
                "request_id": "r1",
                "metadata": {
                    "prompt_tokens": 11,
                    "completion_tokens": 5,
                    "tool_call_count": 1,
                    "num_turns": 2,
                    "max_turns_reached": False,
                },
            }
        ]
    )

    assert trace == [
        {
            "call_id": 3,
            "depth": 2,
            "prompt": "summarize file A",
            "response": "summary",
            "parent_turn": 1,
            "remaining_depth": 0,
            "batch_id": "b1",
            "request_id": "r1",
            "prompt_tokens": 11,
            "completion_tokens": 5,
            "tool_call_count": 1,
            "num_turns": 2,
            "max_turns_reached": False,
        }
    ]
