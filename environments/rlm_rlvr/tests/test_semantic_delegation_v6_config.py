from __future__ import annotations

import csv
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.rlm_rlvr.generate_semantic_delegation_v6_configs import (  # noqa: E402
    BASE_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    FILENAME,
    OUT_DIR,
    RUN_ID,
    build_config,
)


def test_semantic_delegation_v6_generated_config_is_current() -> None:
    assert (OUT_DIR / FILENAME).read_text() == build_config(BASE_CONFIG.read_text())


def test_semantic_delegation_v6_config_uses_packet_credit_and_fixed_eval() -> None:
    config = tomllib.loads((OUT_DIR / FILENAME).read_text())
    assert config["max_steps"] == 50
    assert config["ckpt"]["interval"] == 10
    assert config["trainer"]["loss"]["semantic_child_fraction"] == 0.4
    assert config["orchestrator"]["batch_size"] == 64
    assert config["orchestrator"]["rollouts_per_example"] == 8
    assert config["orchestrator"]["max_trainable_llm_subcalls_per_rollout"] == 3
    assert config["wandb"]["name"] == RUN_ID
    assert config["wandb"]["entity"] == "lsteno-university-of-twente"
    train_args = config["orchestrator"]["env"][0]["args"]
    assert train_args["dataset_id"] == DATASET_ID
    assert train_args["dataset_revision"] == DATASET_REVISION
    assert train_args["efficiency_penalty_mode"] == "accuracy_stratified_group"
    assert train_args["efficiency_tie_break_max"] == 0.05
    assert train_args["efficiency_root_token_multiplier"] == 8.0
    assert train_args["subcall_max_tokens"] == 512
    assert train_args["max_batched_subcalls"] == 20
    assert "llm_subcall_model" not in train_args
    assert (
        config["orchestrator"]["buffer"]["difficulty_metric"]
        == "semantic_chunk_accuracy_metric"
    )
    assert [
        phase["weights"]
        for phase in config["orchestrator"]["buffer"]["curriculum_phases"]
    ] == [
        {"base": 0.65, "sentiment_small": 0.35},
        {
            "base": 0.65,
            "sentiment_small": 0.10,
            "sentiment_medium": 0.15,
            "advanced_small": 0.10,
        },
        {
            "base": 0.65,
            "sentiment_medium": 0.15,
            "advanced_small": 0.10,
            "advanced_medium": 0.10,
        },
    ]
    evals = {item["name"]: item for item in config["orchestrator"]["eval"]["env"]}
    assert evals["semantic_eval"]["args"]["dataset_eval_split"] == "semantic_eval"
    assert evals["semantic_eval"]["args"]["dataset_revision"] == DATASET_REVISION
    assert evals["semantic_eval"]["num_examples"] == 64
    assert evals["mixed_eval"]["args"]["dataset_eval_split"] == "eval"
    assert evals["mixed_eval"]["num_examples"] == 32


def test_semantic_delegation_v6_manifest_contains_only_the_pilot() -> None:
    with (OUT_DIR / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["run_id"] == RUN_ID
    assert rows[0]["status"] == "pending"
    assert rows[0]["dataset_id"] == DATASET_ID
