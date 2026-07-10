from __future__ import annotations

import csv
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.rlm_rlvr.generate_semantic_delegation_v5_configs import (  # noqa: E402
    BASE_CONFIG,
    OUT_DIR,
    PRIVATE_DATASET_ID,
    RUNS,
    build_config,
    config_filename,
    run_id,
)


def test_semantic_delegation_v5_generated_configs_are_current() -> None:
    base = BASE_CONFIG.read_text()
    for max_steps, label, _ in RUNS:
        assert (OUT_DIR / config_filename(max_steps, label)).read_text() == build_config(
            base, max_steps=max_steps, label=label
        )


def test_semantic_delegation_v5_configs_use_natural_credit_and_root_cost_floor() -> None:
    for max_steps, label, _ in RUNS:
        config = tomllib.loads((OUT_DIR / config_filename(max_steps, label)).read_text())
        assert config["max_steps"] == max_steps
        assert config["trainer"]["loss"]["normalization"] == "branch_sequence"
        assert config["trainer"]["loss"]["semantic_child_fraction"] == 0.5
        assert config["deployment"]["num_infer_gpus"] == 4
        assert config["deployment"]["num_train_gpus"] == 4
        assert config["wandb"]["entity"] == "lsteno-university-of-twente"
        assert config["wandb"]["project"] == "rlm-rlvr"
        assert config["wandb"]["name"] == run_id(max_steps, label)
        train_args = config["orchestrator"]["env"][0]["args"]
        eval_args = config["orchestrator"]["eval"]["env"][0]["args"]
        assert train_args["dataset_id"] == PRIVATE_DATASET_ID
        assert eval_args["dataset_id"] == PRIVATE_DATASET_ID
        assert "semantic_record_map_min_records" not in train_args
        assert "semantic_record_map_min_records" not in eval_args
        assert train_args["train_plain_llm_subcalls"] is True
        assert train_args["adaptive_efficiency_beta_min"] == 0.10
        assert train_args["adaptive_efficiency_beta_max"] == 1.0
        assert train_args["efficiency_penalty_applies_to"] == "all_rollouts"
        assert "llm_subcall_model" not in train_args


def test_semantic_delegation_v5_manifest_requires_review_before_full_run() -> None:
    with (OUT_DIR / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run_type"] for row in rows] == ["pilot", "full"]
    assert [row["status"] for row in rows] == ["pending", "gated"]
    assert all(row["dataset_id"] == PRIVATE_DATASET_ID for row in rows)
