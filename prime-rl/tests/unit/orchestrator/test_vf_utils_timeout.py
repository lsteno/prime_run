import asyncio

import verifiers as vf

from prime_rl.orchestrator.vf_utils import (
    EnvClientPool,
    EnvWorkerHandle,
    ManagedEnvClientPool,
    REQUIRED_STATE_COLUMNS,
    get_stable_free_port_pair,
    run_group,
    run_rollout,
)


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


def test_env_client_pool_distributes_concurrent_rollouts_to_least_busy_workers():
    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class FakeClient:
            def __init__(self, address: str):
                self.address = address
                self.pending_requests = {}
                self.count = 0

            async def wait_for_server_startup(self, timeout=None):
                return None

            async def close(self):
                return None

            async def run_rollout(self, *args, **kwargs):
                self.count += 1
                if sum(client.count for client in clients) == 6:
                    started.set()
                await release.wait()
                return vf.RolloutOutput(
                    example_id=self.count,
                    task="rlm_rlvr",
                    trajectory=[],
                    error=None,
                    reward=0.0,
                    metrics={},
                )

        clients = [FakeClient(f"tcp://127.0.0.1:{port}") for port in (5551, 5552, 5553)]
        pool = EnvClientPool(clients, name="rlm_rlvr")
        tasks = [
            asyncio.create_task(
                pool.run_rollout(
                    input=vf.RolloutInput(**_example()),
                    client_config=object(),
                    model="model",
                    sampling_args={"temperature": 0.7},
                )
            )
            for _ in range(6)
        ]
        await asyncio.wait_for(started.wait(), timeout=1)
        assert [client.count for client in clients] == [2, 2, 2]
        release.set()
        await asyncio.gather(*tasks)

    asyncio.run(run())


def test_managed_env_client_pool_skips_quarantined_workers():
    async def run() -> None:
        class FakeClient:
            def __init__(self, address: str):
                self.address = address
                self.pending_requests = {}

            async def wait_for_server_startup(self, timeout=None):
                return None

            async def close(self):
                return None

        handles = [
            EnvWorkerHandle(0, "w0", "tcp://127.0.0.1:60100", None, FakeClient("tcp://127.0.0.1:60100")),
            EnvWorkerHandle(1, "w1", "tcp://127.0.0.1:60102", None, FakeClient("tcp://127.0.0.1:60102")),
        ]
        handles[0].quarantined = True
        pool = ManagedEnvClientPool(handles, name="rlm_rlvr")

        reservation = await pool.reserve_worker()
        try:
            assert reservation.worker_id == 1
            assert reservation.worker_name == "w1"
        finally:
            await reservation.release()

    asyncio.run(run())


def test_managed_env_client_pool_routes_active_reservation():
    async def run() -> None:
        class FakeClient:
            def __init__(self, address: str):
                self.address = address
                self.pending_requests = {}
                self.count = 0

            async def wait_for_server_startup(self, timeout=None):
                return None

            async def close(self):
                return None

            async def run_rollout(self, *args, **kwargs):
                self.count += 1
                return vf.RolloutOutput(
                    example_id=self.count,
                    task="rlm_rlvr",
                    trajectory=[],
                    error=None,
                    reward=0.0,
                    metrics={},
                )

        clients = [FakeClient("tcp://127.0.0.1:60110"), FakeClient("tcp://127.0.0.1:60112")]
        pool = ManagedEnvClientPool(
            [
                EnvWorkerHandle(0, "w0", clients[0].address, None, clients[0]),
                EnvWorkerHandle(1, "w1", clients[1].address, None, clients[1]),
            ],
            name="rlm_rlvr",
        )
        reservation = await pool.reserve_worker()
        token = pool.activate_reservation(reservation)
        try:
            await pool.run_rollout(
                input=vf.RolloutInput(**_example()),
                client_config=object(),
                model="model",
                sampling_args={"temperature": 0.7},
            )
        finally:
            pool.reset_active_reservation(token)
            await reservation.release()

        assert clients[reservation.worker_id].count == 1
        assert clients[1 - reservation.worker_id].count == 0

    asyncio.run(run())


def test_required_state_columns_include_rlm_protocol_metrics():
    assert "used_repl" in REQUIRED_STATE_COLUMNS
    assert "used_llm_subcalls" in REQUIRED_STATE_COLUMNS
    assert "num_llm_subcalls" in REQUIRED_STATE_COLUMNS
    assert "used_rlm_subcalls" in REQUIRED_STATE_COLUMNS
    assert "num_rlm_subcalls" in REQUIRED_STATE_COLUMNS
    assert "max_depth_reached" in REQUIRED_STATE_COLUMNS


def test_stable_free_port_pair_uses_non_ephemeral_pair():
    port = get_stable_free_port_pair()
    assert 61000 <= port <= 65533
    assert port % 2 == 0
