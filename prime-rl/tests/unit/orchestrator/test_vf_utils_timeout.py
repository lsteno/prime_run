import asyncio

import verifiers as vf

from prime_rl.orchestrator.vf_utils import run_group, run_rollout


def _example() -> dict:
    return {
        "example_id": 7,
        "task": "rlm_rlvr",
        "prompt": [{"role": "user", "content": "question"}],
        "answer": "answer",
        "info": {"source": "test"},
    }


def test_run_rollout_timeout_returns_non_trainable_error_rollout():
    async def run() -> None:
        class SlowEnv:
            async def run_rollout(self, *args, **kwargs):
                await asyncio.sleep(1)

        output = await run_rollout(
            env=SlowEnv(),
            client=object(),
            model_name="model",
            example=_example(),
            sampling_args={"temperature": 0.7},
            rollout_timeout_seconds=0.01,
        )

        assert output["example_id"] == 7
        assert output["task"] == "rlm_rlvr"
        assert output["trajectory"] == []
        assert output["reward"] == 0.0
        assert output["stop_condition"] == "rollout_timeout"
        assert output["metrics"]["rollout/timeout"] == 1.0
        assert output["error"]["error_chain_repr"]

    asyncio.run(run())


def test_run_rollout_without_timeout_returns_underlying_output():
    async def run() -> None:
        expected = vf.RolloutOutput(
            example_id=7,
            task="rlm_rlvr",
            trajectory=[],
            error=None,
            reward=1.0,
            metrics={},
        )

        class FastEnv:
            async def run_rollout(self, *args, **kwargs):
                return expected

        output = await run_rollout(
            env=FastEnv(),
            client=object(),
            model_name="model",
            example=_example(),
            sampling_args={"temperature": 0.7},
            rollout_timeout_seconds=None,
        )

        assert output is expected

    asyncio.run(run())


def test_run_group_timeout_returns_one_error_rollout_per_requested_rollout():
    async def run() -> None:
        class SlowEnv:
            async def run_group(self, *args, **kwargs):
                await asyncio.sleep(1)

        outputs = await run_group(
            env=SlowEnv(),
            client=object(),
            model_name="model",
            example=_example(),
            rollouts_per_example=3,
            sampling_args={"temperature": 0.7},
            rollout_timeout_seconds=0.01,
        )

        assert len(outputs) == 3
        assert all(output["trajectory"] == [] for output in outputs)
        assert all(output["stop_condition"] == "rollout_timeout" for output in outputs)
        assert all(output["error"] is not None for output in outputs)

    asyncio.run(run())
