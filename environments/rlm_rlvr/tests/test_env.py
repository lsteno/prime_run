import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("verifiers.envs.experimental.rlm_env")

import rlm_rlvr.env as env_module
from rlm_rlvr.env import RLMRLVREnv, load_environment


def test_load_environment_returns_canonical_rlm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_module.vf, "ensure_keys", lambda keys: None)
    monkeypatch.setattr(
        env_module,
        "build_datasets",
        lambda **kwargs: (lambda: [], lambda: []),
    )
    monkeypatch.setattr(env_module, "build_rubric", lambda **kwargs: env_module.vf.Rubric(funcs=[]))

    env = load_environment(
        dataset_id=None,
        max_examples=1,
        max_eval_examples=1,
    )

    assert isinstance(env, RLMRLVREnv)
    assert env.env_id == "rlm_rlvr"
