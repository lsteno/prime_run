from pathlib import Path
from types import SimpleNamespace

from prime_rl.configs.shared import LogExtrasConfig, WandbWithExtrasConfig
from prime_rl.utils.monitor import wandb as wandb_module
from prime_rl.utils.monitor.wandb import WandbMonitor


class _FakeTable:
    def __init__(self, columns=None, log_mode=None, dataframe=None):
        self.columns = columns
        self.log_mode = log_mode
        self.dataframe = dataframe
        self.rows = []

    def add_data(self, *values):
        self.rows.append(values)


class _FakeTokenizer:
    def decode(self, token_ids):
        del token_ids
        return "x" * 32


def test_wandb_sample_logging_omits_input_ids_and_truncates_messages(monkeypatch, tmp_path: Path) -> None:
    logged = []
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(wandb_module.wandb, "init", lambda **kwargs: SimpleNamespace(id="test-run"))
    monkeypatch.setattr(wandb_module.wandb, "Table", _FakeTable)
    monkeypatch.setattr(wandb_module.wandb, "log", lambda data, step=None, commit=None: logged.append((data, step, commit)))

    config = WandbWithExtrasConfig(
        project="test-project",
        log_extras=LogExtrasConfig(
            samples=True,
            interval=1,
            sample_max_chars=8,
            sample_include_input_ids=False,
            final_samples=False,
        ),
    )
    monitor = WandbMonitor(
        config=config,
        output_dir=tmp_path,
        tokenizer=_FakeTokenizer(),
        run_config=None,
    )

    assert "input_ids" not in monitor.samples_cols

    rollout = {
        "example_id": 1,
        "task": "rlm_rlvr",
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1, 2, 3],
                    "completion_ids": [4, 5],
                },
                "extras": {
                    "rlm_debug": {
                        "expected_answers": ["42"],
                        "judge_score": 1,
                        "judge_raw_response": "[exact_match]",
                    }
                },
            }
        ],
        "answer": "42",
        "final_answer": "42",
        "reward": 1.0,
    }

    monitor.log_samples([rollout], step=1)

    assert monitor.samples[0]["messages"] == "x" * 8 + "\n...[truncated 24 chars; full trace is on disk]"
    assert "input_ids" not in monitor.samples[0]
    assert logged[-1][0] == {"samples": monitor.samples_table}


def test_wandb_final_samples_disabled_by_default(monkeypatch, tmp_path: Path) -> None:
    logged = []
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(wandb_module.wandb, "init", lambda **kwargs: SimpleNamespace(id="test-run"))
    monkeypatch.setattr(wandb_module.wandb, "Table", _FakeTable)
    monkeypatch.setattr(wandb_module.wandb, "log", lambda data, step=None, commit=None: logged.append((data, step, commit)))

    monitor = WandbMonitor(
        config=WandbWithExtrasConfig(project="test-project"),
        output_dir=tmp_path,
        tokenizer=_FakeTokenizer(),
        run_config=None,
    )
    monitor.samples.append({"step": 1})
    monitor.log_final_samples()

    assert logged == []
