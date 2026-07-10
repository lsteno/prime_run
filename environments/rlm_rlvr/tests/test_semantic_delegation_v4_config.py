from __future__ import annotations

import csv
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.rlm_rlvr.generate_semantic_delegation_v4_configs import (  # noqa: E402
    BASE_CONFIG,
    OUT_DIR,
    PRIVATE_DATASET_ID,
    RUNS,
    build_config,
    config_filename,
    run_id,
)


def test_semantic_delegation_v4_generated_configs_are_current() -> None:
    base = BASE_CONFIG.read_text()
    for max_steps, label in RUNS:
        assert (OUT_DIR / config_filename(max_steps, label)).read_text() == build_config(
            base, max_steps=max_steps, label=label
        )


def test_semantic_delegation_v4_configs_use_verified_dual_contract_and_branch_loss() -> None:
    for max_steps, label in RUNS:
        config = tomllib.loads((OUT_DIR / config_filename(max_steps, label)).read_text())
        assert config["max_steps"] == max_steps
        assert config["trainer"]["loss"]["normalization"] == "branch_sequence"
        assert config["trainer"]["loss"]["semantic_child_fraction"] == 0.5
        assert config["deployment"]["num_infer_gpus"] == 4
        assert config["deployment"]["num_train_gpus"] == 4
        assert config["wandb"]["entity"] == "lsteno-university-of-twente"
        assert config["wandb"]["project"] == "rlm-rlvr"
        assert config["wandb"]["name"] == run_id(max_steps, label)
        for env_config in [config["orchestrator"]["env"][0], config["orchestrator"]["eval"]["env"][0]]:
            args = env_config["args"]
            assert args["dataset_id"] == PRIVATE_DATASET_ID
            assert args["semantic_record_map_min_records"] == 8
            assert args["train_plain_llm_subcalls"] is True
            assert "llm_subcall_model" not in args


def test_semantic_delegation_v4_manifest_gates_full_run() -> None:
    with (OUT_DIR / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run_type"] for row in rows] == ["pilot", "full"]
    assert [row["status"] for row in rows] == ["pending", "gated"]
    assert all(row["dataset_id"] == PRIVATE_DATASET_ID for row in rows)
