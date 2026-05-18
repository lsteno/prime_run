import asyncio
from collections import deque
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import verifiers as vf

import prime_rl.orchestrator.scheduler as scheduler_module
from prime_rl.orchestrator.scheduler import GroupState, InflightRolloutInfo, Scheduler
from prime_rl.orchestrator.vf_utils import EnvWorkerHandle, ManagedEnvClientPool


def test_update_off_policy_does_not_increment_interleaved_on_policy_tasks():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_off_policy_steps = 1
        scheduler.cancelled_rollouts_count = 0
        scheduler.logger = MagicMock()

        client = SimpleNamespace(api_base_url="http://test")
        stale_task = asyncio.create_task(asyncio.sleep(60))
        survivor_task = asyncio.create_task(asyncio.sleep(60))
        interleaved_task = None

        scheduler.inflight_requests = {
            stale_task: InflightRolloutInfo(off_policy_steps=1, client_config=client, task="test", group_id=1),
            survivor_task: InflightRolloutInfo(off_policy_steps=0, client_config=client, task="test", group_id=2),
        }

        async def drop_group(group_id: int) -> int:
            tasks_to_remove = [
                task for task, info in list(scheduler.inflight_requests.items()) if info.group_id == group_id
            ]
            for task in tasks_to_remove:
                scheduler.inflight_requests.pop(task, None)
                task.cancel()

            await asyncio.sleep(0)

            nonlocal interleaved_task
            if interleaved_task is None:
                interleaved_task = asyncio.create_task(asyncio.sleep(60))
                scheduler.inflight_requests[interleaved_task] = InflightRolloutInfo(
                    off_policy_steps=0,
                    client_config=client,
                    task="test",
                    group_id=3,
                )
            return len(tasks_to_remove)

        scheduler.drop_group = drop_group

        await scheduler._update_off_policy()

        assert stale_task not in scheduler.inflight_requests
        assert scheduler.inflight_requests[survivor_task].off_policy_steps == 1
        assert interleaved_task is not None
        assert scheduler.inflight_requests[interleaved_task].off_policy_steps == 0
        assert scheduler.cancelled_rollouts_count == 1

        for task in (stale_task, survivor_task, interleaved_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_schedule_rollout_uses_task_retry_config():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.rate_limiter = None
        scheduler.groups = {
            1: GroupState(
                example={"task": "rlm_rlvr"},
                pending_slots=deque([0]),
                pinned_client=None,
            )
        }
        scheduler.inflight_requests = {}
        scheduler.env = object()
        scheduler.model_name = "test-model"
        scheduler.sampling_args = {"temperature": 0.7}
        scheduler.max_retries_by_task = {"rlm_rlvr": 3}
        scheduler.ckpt_step = 0
        scheduler.step = 0
        scheduler.current_phase = "train"
        scheduler.config = SimpleNamespace(
            rollout_timeout_seconds=12.5,
            env_worker_recovery=SimpleNamespace(enabled=False),
            attempt_logging=SimpleNamespace(enabled=False),
            output_dir=None,
        )

        client = SimpleNamespace(api_base_url="http://test", extra_headers={})

        async def select_client():
            return client

        scheduler._select_least_loaded_client = select_client
        scheduler._get_env_worker_pool = lambda task: None

        captured: dict[str, int] = {}
        original_run_rollout = scheduler_module.run_rollout

        async def fake_run_rollout(**kwargs):
            captured["max_retries"] = kwargs["max_retries"]
            captured["rollout_timeout_seconds"] = kwargs["rollout_timeout_seconds"]
            return {"trajectory": [], "error": None}

        scheduler_module.run_rollout = fake_run_rollout
        try:
            await scheduler.schedule_rollout(group_id=1)
            assert scheduler.inflight_requests
            task = next(iter(scheduler.inflight_requests))
            await task
        finally:
            scheduler_module.run_rollout = original_run_rollout

        assert captured["max_retries"] == 3
        assert captured["rollout_timeout_seconds"] == 12.5

    asyncio.run(run())


def test_schedule_rollout_records_worker_attempt_metadata():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.rate_limiter = None
        scheduler.groups = {
            1: GroupState(
                example={"task": "rlm_rlvr", "example_id": 99, "info": {"source_id": "src-99"}},
                pending_slots=deque([2]),
                pinned_client=None,
            )
        }
        scheduler.inflight_requests = {}
        scheduler.env = object()
        scheduler.model_name = "test-model"
        scheduler.sampling_args = {"temperature": 0.7}
        scheduler.max_retries_by_task = {"rlm_rlvr": 0}
        scheduler.ckpt_step = 7
        scheduler.step = 11
        scheduler.current_phase = "train"
        scheduler.config = SimpleNamespace(
            rollout_timeout_seconds=12.5,
            env_worker_recovery=SimpleNamespace(enabled=True),
            attempt_logging=SimpleNamespace(enabled=False),
            output_dir=None,
        )

        client = SimpleNamespace(api_base_url="http://test", extra_headers={})

        async def select_client():
            return client

        scheduler._select_least_loaded_client = select_client

        class FakeWorker:
            address = "worker://0"
            pending_requests = {}

            async def wait_for_server_startup(self, timeout=None):
                return None

            async def close(self):
                return None

            async def run_rollout(self, *args, **kwargs):
                return vf.RolloutOutput(
                    example_id=99,
                    task="rlm_rlvr",
                    trajectory=[{"role": "assistant", "content": "x"}],
                    error=None,
                    reward=0.0,
                    metrics={},
                )

        pool = ManagedEnvClientPool(
            [
                EnvWorkerHandle(
                    worker_id=0,
                    worker_name="rlm_rlvr_w0",
                    address="worker://0",
                    process=None,
                    client=FakeWorker(),
                )
            ],
            name="rlm_rlvr",
        )
        scheduler._get_env_worker_pool = lambda task: pool

        original_run_rollout = scheduler_module.run_rollout

        async def fake_run_rollout(**kwargs):
            return await pool.run_rollout(
                input=object(),
                client_config=client,
                model="model",
                sampling_args={},
            )

        scheduler_module.run_rollout = fake_run_rollout
        try:
            await scheduler.schedule_rollout(group_id=1)
            assert scheduler.inflight_requests
            task, info = next(iter(scheduler.inflight_requests.items()))
            assert info.attempt_id
            assert info.worker_id == 0
            assert info.worker_generation == 0
            assert info.env_worker_name == "rlm_rlvr_w0"
            assert info.slot_index == 2
            assert info.attempt_number == 1
            await task
            await scheduler._release_worker_reservation(info)
        finally:
            scheduler_module.run_rollout = original_run_rollout

    asyncio.run(run())


def test_reschedule_or_drop_slot_cools_down_group_after_max_attempts():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.groups = {
            3: GroupState(
                example={"task": "rlm_rlvr", "example_id": 5, "prompt": []},
                pending_slots=deque([]),
            )
        }
        scheduler.inflight_requests = {}
        scheduler.current_phase = "train"
        scheduler.logger = MagicMock()
        scheduler.attempt_group_drops = 0
        scheduler.attempt_drops_by_task = defaultdict(int)
        scheduler.cancelled_rollouts_count = 0
        scheduler.config = SimpleNamespace(
            env_worker_recovery=SimpleNamespace(
                enabled=True,
                max_rollout_attempts_per_slot=4,
                max_attempts_cooldown_steps=5,
            ),
            attempt_logging=SimpleNamespace(enabled=False),
            output_dir=None,
        )
        cooldown_calls = []

        class FakeBuffer:
            def put_example_on_hard_cooldown(self, example, step, cooldown_steps):
                cooldown_calls.append((example["example_id"], step, cooldown_steps))
                return True

        scheduler.buffer = FakeBuffer()

        info = InflightRolloutInfo(
            off_policy_steps=0,
            client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
            task="rlm_rlvr",
            group_id=3,
            slot_index=0,
            attempt_number=4,
        )

        await scheduler._reschedule_or_drop_slot(info, step=12, reason="rollout_timeout", count_as_reschedule=True)

        assert 3 not in scheduler.groups
        assert cooldown_calls == [(5, 12, 5)]

    asyncio.run(run())
