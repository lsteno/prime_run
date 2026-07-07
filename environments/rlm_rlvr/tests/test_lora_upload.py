from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
UPLOAD_SCRIPT = ROOT / "scripts/rlm_rlvr/upload_lora_adapter.py"


def _load_upload_module():
    spec = importlib.util.spec_from_file_location("upload_lora_adapter", UPLOAD_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h100_depth1_sweep_repo_id_uses_compact_matrix_name() -> None:
    upload = _load_upload_module()
    repo_id = upload.repo_id_for_run(
        "rlm-rlvr-qwen3-4b-depth1-llmonly-h100-r4-a8-lr1e-6-s150-bal35f40v1",
        "lsteno/qwen3-rlm-depth1-h100-sameroot",
    )

    assert repo_id == "lsteno/qwen3-rlm-depth1-h100-sameroot-r4-a8-lr1e-6-s150-bal35f40v1-lora"
    assert len(repo_id) <= upload.HF_REPO_ID_MAX_LENGTH


def test_unknown_long_run_id_repo_id_is_hf_length_safe() -> None:
    upload = _load_upload_module()
    repo_id = upload.repo_id_for_run(
        "rlm-rlvr-" + "very-long-unrecognized-run-" * 8,
        "lsteno/qwen3-rlm-depth1-h100-sameroot",
    )

    assert len(repo_id) <= upload.HF_REPO_ID_MAX_LENGTH
    assert not repo_id.endswith("-")
    assert not repo_id.endswith(".")
