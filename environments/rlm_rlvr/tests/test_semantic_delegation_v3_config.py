from __future__ import annotations

import csv
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.rlm_rlvr.generate_semantic_delegation_v3_curriculum_config import (  # noqa: E402
    BASE_CONFIG,
    CONFIG_FILENAME,
    OUT_DIR,
    PRIVATE_DATASET_ID,
    RUN_ID,
    build_config,
)


def test_semantic_delegation_v3_generated_config_is_current() -> None:
    config_path = OUT_DIR / CONFIG_FILENAME
    assert config_path.read_text() == build_config(BASE_CONFIG.read_text())


def test_semantic_delegation_v3_config_matches_curriculum_plan() -> None:
    config = tomllib.loads((OUT_DIR / CONFIG_FILENAME).read_text())
    orchestrator = config["orchestrator"]
    buffer = orchestrator["buffer"]
    assert config["max_steps"] == 150
    assert config["deployment"]["num_infer_gpus"] == 4
    assert config["deployment"]["num_train_gpus"] == 4
    assert orchestrator["batch_size"] == 64
    assert orchestrator["rollouts_per_example"] == 4
    assert orchestrator["max_trainable_llm_subcalls_per_rollout"] == 3
    assert orchestrator["semantic_child_local_advantage_weight"] == 0.8
    assert buffer["difficulty_metric"] == "semantic_progress_metric"
    assert buffer["hash_keys"] == ["example_id"]
    assert buffer["online_filter_hard"] is True
    assert buffer["online_filter_easy"] is False
    assert buffer["easy_threshold"] == 1.0
    assert buffer["curriculum_bucket_path"] == "metadata.curriculum_bucket"
    assert buffer["curriculum_phases"] == [
        {"start_step": 0, "end_step": 29, "weights": {"base": 0.65, "semantic_4": 0.35}},
        {
            "start_step": 30,
            "end_step": 79,
            "weights": {"base": 0.65, "semantic_4": 0.0875, "semantic_8": 0.2625},
        },
        {
            "start_step": 80,
            "end_step": 149,
            "weights": {
                "base": 0.65,
                "semantic_4": 0.035,
                "semantic_8": 0.07,
                "semantic_16": 0.21,
                "semantic_global": 0.035,
            },
        },
    ]

    for env_config in [orchestrator["env"][0], orchestrator["eval"]["env"][0]]:
        args = env_config["args"]
        assert args["dataset_id"] == PRIVATE_DATASET_ID
        assert "data_paths" not in args
        assert "eval_data_paths" not in args
        assert "llm_subcall_model" not in args
        assert args["max_depth"] == 0
        assert args["train_plain_llm_subcalls"] is True
        assert args["max_total_subcalls"] == 24
        assert args["max_batched_subcalls"] == 16
        assert args["subcall_batch_overflow_mode"] == "reject"
        assert args["efficiency_penalty_applies_to"] == "correct_only"
        assert args["max_turn_penalty"] == 0.5

    train_args = orchestrator["env"][0]["args"]
    assert train_args["efficiency_penalty_mode"] == "adaptive_group"
    assert train_args["adaptive_efficiency_beta_max"] == 1.0
    assert train_args["adaptive_efficiency_gamma"] == 2.0
    assert train_args["adaptive_efficiency_solve_rate_floor"] == 0.25
    assert train_args["adaptive_efficiency_cost_basis"] == "weighted_turn_tokens"
    assert train_args["efficiency_root_token_multiplier"] == 8.0
    assert train_args["efficiency_plain_subcall_token_multiplier"] == 1.0
    assert config["wandb"]["entity"] == "lsteno-university-of-twente"
    assert config["wandb"]["project"] == "rlm-rlvr"
    assert config["wandb"]["name"] == RUN_ID


def test_semantic_delegation_v3_manifest_points_at_private_dataset() -> None:
    with (OUT_DIR / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["run_id"] == RUN_ID
    assert rows[0]["dataset_id"] == PRIVATE_DATASET_ID
    assert rows[0]["status"] == "pending"
