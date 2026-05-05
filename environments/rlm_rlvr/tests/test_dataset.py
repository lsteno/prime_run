from __future__ import annotations

import json

from rlm_rlvr.dataset import _rows_to_dataset
from rlm_rlvr.env import RLMRLVREnv


def test_dataset_info_preserves_compact_eval_metadata() -> None:
    dataset = _rows_to_dataset(
        [
            {
                "id": "example-1",
                "dataset": "oolong",
                "task": "TASK_TYPE.MOST_FREQ",
                "prompt": "question",
                "context": "context",
                "answer": "answer",
                "answer_type": "ANSWER_TYPE.LABEL",
                "context_token_count": 123,
                "metadata": {
                    "source_dataset": "sst2",
                    "task_group": "counting",
                    "context_window_text_with_labels": "large text not copied into debug metadata",
                },
            }
        ],
        dataset_name="local",
    )

    info = json.loads(dataset[0]["info"])

    assert info["source_id"] == "example-1"
    assert info["dataset_name"] == "oolong"
    assert info["source_task"] == "TASK_TYPE.MOST_FREQ"
    assert info["answer_type"] == "ANSWER_TYPE.LABEL"
    assert info["context_token_count"] == 123
    assert info["metadata"]["source_dataset"] == "sst2"
    assert info["metadata"]["task_group"] == "counting"


def test_sample_metadata_omits_large_context_metadata() -> None:
    metadata = RLMRLVREnv._sample_metadata(
        {
            "source_id": "example-1",
            "dataset_name": "oolong",
            "source_task": "TASK_TYPE.MOST_FREQ",
            "answer_type": "ANSWER_TYPE.LABEL",
            "context_token_count": 123,
            "metadata": {
                "source_dataset": "sst2",
                "task_group": "counting",
                "context_window_text_with_labels": "large text",
            },
        }
    )

    assert metadata["source_id"] == "example-1"
    assert metadata["metadata"] == {
        "source_dataset": "sst2",
        "task_group": "counting",
    }
