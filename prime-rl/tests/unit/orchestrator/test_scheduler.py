import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import prime_rl.orchestrator.scheduler as scheduler_module
from prime_rl.orchestrator.scheduler import InflightRolloutInfo, Scheduler


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
            1: SimpleNamespace(
                example={"task": "rlm_rlvr"},
                rollouts_to_schedule=1,
                pinned_client=None,
            )
        }
        scheduler.inflight_requests = {}
        scheduler.env = object()
        scheduler.model_name = "test-model"
        scheduler.sampling_args = {"temperature": 0.7}
        scheduler.max_retries_by_task = {"rlm_rlvr": 3}

        client = SimpleNamespace(api_base_url="http://test", extra_headers={})

        async def select_client():
            return client

        scheduler._select_least_loaded_client = select_client

        captured: dict[str, int] = {}
        original_run_rollout = scheduler_module.run_rollout

        async def fake_run_rollout(**kwargs):
            captured["max_retries"] = kwargs["max_retries"]
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

    asyncio.run(run())
