import asyncio
import json
import time
from collections import deque
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
        await asyncio.gather(task, return_exceptions=True)

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


def test_scheduler_metadata_helpers_accept_json_string_info():
    scheduler = Scheduler.__new__(Scheduler)
    example = {
        "task": "rlm_rlvr",
        "info": json.dumps({"source_id": "frames-123", "dataset_name": "frames"}),
    }

    assert scheduler._source_id(example) == "frames-123"
    assert scheduler._dataset_name(example) == "frames"


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


def test_enforce_rollout_deadlines_times_out_running_attempt_and_reschedules_slot():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.groups = {
            3: GroupState(
                example={"task": "rlm_rlvr", "example_id": 5, "prompt": []},
                pending_slots=deque([]),
            )
        }
        task = asyncio.create_task(asyncio.sleep(60))
        scheduler.inflight_requests = {
            task: InflightRolloutInfo(
                off_policy_steps=0,
                client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
                task="rlm_rlvr",
                group_id=3,
                slot_index=0,
                attempt_number=1,
                start_time_perf=time.perf_counter() - 10.0,
            )
        }
        scheduler.current_phase = "train"
        scheduler.logger = MagicMock()
        scheduler.sampling_args = {"temperature": 0.7}
        scheduler.rollouts_per_example = 4
        scheduler.total_rollouts_by_task = defaultdict(int)
        scheduler.timeout_rollouts_by_task = defaultdict(int)
        scheduler.empty_rollouts_by_task = defaultdict(int)
        scheduler.errored_rollouts_by_task = defaultdict(int)
        scheduler.attempt_timeouts = 0
        scheduler.attempt_timeouts_by_task = defaultdict(int)
        scheduler.attempts_by_task = defaultdict(int)
        scheduler.attempt_reschedules = 0
        scheduler.config = SimpleNamespace(
            rollout_timeout_seconds=1.0,
            env_worker_recovery=SimpleNamespace(
                enabled=True,
                max_rollout_attempts_per_slot=4,
                max_attempts_cooldown_steps=5,
            ),
            attempt_logging=SimpleNamespace(enabled=False),
            output_dir=None,
        )
        restart_calls = []

        async def fake_restart(info, step):
            restart_calls.append((info.group_id, step))

        scheduler._restart_worker_for_timeout = fake_restart

        await scheduler._enforce_rollout_deadlines(step=12)

        assert task not in scheduler.inflight_requests
        assert scheduler.timeout_rollouts_by_task["rlm_rlvr"] == 1
        assert scheduler.attempt_timeouts == 1
        assert scheduler.attempt_reschedules == 1
        assert list(scheduler.groups[3].pending_slots) == [0]
        assert restart_calls == [(3, 12)]

        await asyncio.sleep(0)

    asyncio.run(run())


def test_train_rollout_timeout_drops_whole_group_on_first_timeout():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.groups = {
            3: GroupState(
                example={"task": "rlm_rlvr", "example_id": 5, "prompt": []},
                pending_slots=deque([]),
                completed_rollouts={1: vf.RolloutOutput(
                    example_id=5,
                    task="rlm_rlvr",
                    trajectory=[{"role": "assistant", "content": "ok"}],
                    error=None,
                    reward=0.0,
                    metrics={},
                )},
            )
        }
        timed_out_task = asyncio.create_task(asyncio.sleep(60))
        sibling_task = asyncio.create_task(asyncio.sleep(60))
        scheduler.inflight_requests = {
            timed_out_task: InflightRolloutInfo(
                off_policy_steps=0,
                client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
                task="rlm_rlvr",
                group_id=3,
                slot_index=0,
                attempt_number=1,
                start_time_perf=time.perf_counter() - 10.0,
            ),
            sibling_task: InflightRolloutInfo(
                off_policy_steps=0,
                client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
                task="rlm_rlvr",
                group_id=3,
                slot_index=2,
                attempt_number=1,
                start_time_perf=time.perf_counter(),
            ),
        }
        scheduler.task_done_perf = {}
        scheduler.current_phase = "train"
        scheduler.logger = MagicMock()
        scheduler.sampling_args = {"temperature": 0.7}
        scheduler.rollouts_per_example = 4
        scheduler.total_rollouts_by_task = defaultdict(int)
        scheduler.timeout_rollouts_by_task = defaultdict(int)
        scheduler.empty_rollouts_by_task = defaultdict(int)
        scheduler.errored_rollouts_by_task = defaultdict(int)
        scheduler.attempt_timeouts = 0
        scheduler.attempt_timeouts_by_task = defaultdict(int)
        scheduler.attempts_by_task = defaultdict(int)
        scheduler.attempt_reschedules = 0
        scheduler.attempt_group_drops = 0
        scheduler.attempt_group_drops_first_timeout = 0
        scheduler.timeout_cooldown_groups = 0
        scheduler.attempt_drops_by_task = defaultdict(int)
        scheduler.attempt_first_timeout_drops_by_task = defaultdict(int)
        scheduler.config = SimpleNamespace(
            rollout_timeout_seconds=1.0,
            env_worker_recovery=SimpleNamespace(
                enabled=True,
                max_rollout_attempts_per_slot=4,
                max_attempts_cooldown_steps=5,
                drop_group_on_first_timeout=True,
                first_timeout_cooldown_steps=5,
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
        restart_calls = []

        async def fake_restart(info, step):
            restart_calls.append((info.group_id, step))

        scheduler._restart_worker_for_timeout = fake_restart

        await scheduler._enforce_rollout_deadlines(step=12)

        assert 3 not in scheduler.groups
        assert timed_out_task not in scheduler.inflight_requests
        assert sibling_task not in scheduler.inflight_requests
        assert sibling_task.cancelled()
        assert cooldown_calls == [(5, 12, 5)]
        assert restart_calls == [(3, 12)]
        assert scheduler.attempt_reschedules == 0
        assert scheduler.attempt_group_drops == 1
        assert scheduler.attempt_group_drops_first_timeout == 1
        assert scheduler.timeout_cooldown_groups == 1
        assert scheduler.attempt_first_timeout_drops_by_task["rlm_rlvr"] == 1

    asyncio.run(run())


def test_enforce_rollout_deadlines_does_not_timeout_completed_success():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        start_time = time.perf_counter() - 10.0
        task = asyncio.create_task(asyncio.sleep(0, result={"stop_condition": "has_final_env_response"}))
        await task
        scheduler.inflight_requests = {
            task: InflightRolloutInfo(
                off_policy_steps=0,
                client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
                task="rlm_rlvr",
                group_id=3,
                slot_index=0,
                attempt_number=1,
                start_time_perf=start_time,
            )
        }
        scheduler.task_done_perf = {task: start_time + 0.5}
        scheduler.config = SimpleNamespace(rollout_timeout_seconds=1.0)
        scheduler._handle_rollout_timeout = MagicMock()

        await scheduler._enforce_rollout_deadlines(step=12)

        assert task in scheduler.inflight_requests
        scheduler._handle_rollout_timeout.assert_not_called()

    asyncio.run(run())


def test_late_success_after_timeout_drops_train_group_on_first_timeout():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        start_time = time.perf_counter() - 10.0
        task = asyncio.create_task(asyncio.sleep(0, result={"stop_condition": "has_final_env_response"}))
        await task
        scheduler.groups = {
            3: GroupState(
                example={"task": "rlm_rlvr", "example_id": 5, "prompt": []},
                pending_slots=deque([]),
            )
        }
        scheduler.inflight_requests = {
            task: InflightRolloutInfo(
                off_policy_steps=0,
                client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
                task="rlm_rlvr",
                group_id=3,
                slot_index=0,
                attempt_number=1,
                start_time_perf=start_time,
            )
        }
        scheduler.task_done_perf = {task: start_time + 2.0}
        scheduler.config = SimpleNamespace(
            rollout_timeout_seconds=1.0,
            attempt_logging=SimpleNamespace(enabled=False),
            output_dir=None,
            env_worker_recovery=SimpleNamespace(
                enabled=True,
                max_rollout_attempts_per_slot=4,
                max_attempts_cooldown_steps=5,
                drop_group_on_first_timeout=True,
                first_timeout_cooldown_steps=5,
            ),
        )
        scheduler.current_phase = "train"
        scheduler.logger = MagicMock()
        scheduler.scheduler_attempts_finished = 0
        scheduler.attempt_late_success_after_timeout = 0
        scheduler.attempt_late_success_after_timeout_by_task = defaultdict(int)
        scheduler.attempts_by_task = defaultdict(int)
        scheduler.attempt_reschedules = 0
        scheduler.attempt_group_drops = 0
        scheduler.attempt_group_drops_first_timeout = 0
        scheduler.timeout_cooldown_groups = 0
        scheduler.attempt_drops_by_task = defaultdict(int)
        scheduler.attempt_first_timeout_drops_by_task = defaultdict(int)
        scheduler._release_worker_reservation = AsyncMock()
        scheduler._restart_worker_for_timeout = AsyncMock()
        cooldown_calls = []

        class FakeBuffer:
            def put_example_on_hard_cooldown(self, example, step, cooldown_steps):
                cooldown_calls.append((example["example_id"], step, cooldown_steps))
                return True

        scheduler.buffer = FakeBuffer()

        await scheduler._enforce_rollout_deadlines(step=12)

        assert task not in scheduler.inflight_requests
        assert 3 not in scheduler.groups
        assert cooldown_calls == [(5, 12, 5)]
        assert scheduler.attempt_late_success_after_timeout == 1
        assert scheduler.attempt_reschedules == 0
        assert scheduler.attempt_group_drops_first_timeout == 1

    asyncio.run(run())


def test_enforce_rollout_deadlines_marks_late_success_without_training_it():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        start_time = time.perf_counter() - 10.0
        task = asyncio.create_task(asyncio.sleep(0, result={"stop_condition": "has_final_env_response"}))
        await task
        scheduler.groups = {
            3: GroupState(
                example={"task": "rlm_rlvr", "example_id": 5, "prompt": []},
                pending_slots=deque([]),
            )
        }
        scheduler.inflight_requests = {
            task: InflightRolloutInfo(
                off_policy_steps=0,
                client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
                task="rlm_rlvr",
                group_id=3,
                slot_index=0,
                attempt_number=1,
                start_time_perf=start_time,
            )
        }
        scheduler.task_done_perf = {task: start_time + 2.0}
        scheduler.config = SimpleNamespace(
            rollout_timeout_seconds=1.0,
            attempt_logging=SimpleNamespace(enabled=False),
            output_dir=None,
            env_worker_recovery=SimpleNamespace(
                enabled=True,
                max_rollout_attempts_per_slot=4,
                max_attempts_cooldown_steps=5,
            ),
        )
        scheduler.current_phase = "train"
        scheduler.scheduler_attempts_finished = 0
        scheduler.attempt_late_success_after_timeout = 0
        scheduler.attempt_late_success_after_timeout_by_task = defaultdict(int)
        scheduler.attempts_by_task = defaultdict(int)
        scheduler.attempt_reschedules = 0
        scheduler._release_worker_reservation = AsyncMock()

        await scheduler._enforce_rollout_deadlines(step=12)

        assert task not in scheduler.inflight_requests
        assert list(scheduler.groups[3].pending_slots) == [0]
        assert scheduler.attempt_late_success_after_timeout == 1
        assert scheduler.attempt_reschedules == 1

    asyncio.run(run())


def test_allowed_inflight_uses_remaining_need_plus_cushion():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.max_inflight_rollouts = 100
    scheduler.batch_size = 64
    scheduler.token_batch_size = None
    scheduler.scheduler_allowed_inflight = 0
    scheduler.config = SimpleNamespace(async_scheduling=SimpleNamespace(inflight_completion_cushion=20))

    assert scheduler._allowed_inflight_for_progress(0) == 84
    assert scheduler._allowed_inflight_for_progress(60) == 24
    assert scheduler._allowed_inflight_for_progress(64) == 20


def test_fill_inflight_requests_respects_allowed_inflight():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_inflight_rollouts = 100
        scheduler.rollouts_per_example = 4
        scheduler.inflight_requests = {}
        scheduler.groups = {}
        scheduler.next_group_id = 0

        class FakeBuffer:
            def sample_examples(self, n):
                return [{"task": "rlm_rlvr", "example_id": len(scheduler.groups), "prompt": []}]

        scheduler.buffer = FakeBuffer()

        async def fake_schedule_rollout(group_id: int):
            task = asyncio.create_task(asyncio.sleep(60))
            scheduler.inflight_requests[task] = InflightRolloutInfo(
                off_policy_steps=0,
                client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
                task="rlm_rlvr",
                group_id=group_id,
            )
            return True

        scheduler.schedule_rollout = fake_schedule_rollout

        await scheduler._fill_inflight_requests(allowed_inflight=3)

        assert len(scheduler.inflight_requests) == 3

        await asyncio.gather(*(asyncio.create_task(asyncio.sleep(0)) for _ in []))
        for task in scheduler.inflight_requests:
            task.cancel()
        await asyncio.gather(*scheduler.inflight_requests.keys(), return_exceptions=True)

    asyncio.run(run())


def test_managed_env_pool_try_reserve_respects_per_worker_cap():
    async def run() -> None:
        class FakeWorker:
            address = "worker://0"
            pending_requests = {}

            async def wait_for_server_startup(self, timeout=None):
                return None

            async def close(self):
                return None

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

        first = await pool.try_reserve_worker(max_requests_per_worker=1)
        assert first is not None
        second = await pool.try_reserve_worker(max_requests_per_worker=1)
        assert second is None
        assert pool.worker_loads()["rlm_rlvr_w0"] == 1
        assert pool.at_capacity_count(1) == 1

        await first.release()
        third = await pool.try_reserve_worker(max_requests_per_worker=1)
        assert third is not None
        await third.release()

    asyncio.run(run())


def test_trim_carryover_keeps_bounded_oldest_attempts_and_drops_other_groups():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.current_phase = "train"
        scheduler.config = SimpleNamespace(
            output_dir=None,
            rollout_timeout_seconds=400,
            attempt_logging=SimpleNamespace(enabled=False),
            async_scheduling=SimpleNamespace(
                max_cross_step_carryover=2,
                restart_workers_for_stale_cancel=False,
                batch_complete_cancel_grace_seconds=0,
            ),
            env_worker_recovery=SimpleNamespace(enabled=True),
        )
        scheduler.groups = {
            idx: GroupState(example={"task": "rlm_rlvr", "example_id": idx, "prompt": []}, pending_slots=deque([]))
            for idx in range(4)
        }
        scheduler.inflight_requests = {}
        scheduler.scheduler_cancelled_batch_complete = 0
        scheduler.scheduler_cancelled_batch_complete_by_worker = defaultdict(int)

        for idx in range(4):
            task = asyncio.create_task(asyncio.sleep(60))
            scheduler.inflight_requests[task] = InflightRolloutInfo(
                off_policy_steps=0,
                client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
                task="rlm_rlvr",
                group_id=idx,
                slot_index=0,
                start_time_perf=100.0 + idx,
                worker_id=idx,
                worker_generation=0,
                env_worker_name=f"w{idx}",
            )

        scheduler._release_worker_reservation = AsyncMock()
        scheduler._restart_worker_for_batch_cancel = AsyncMock()

        await scheduler._trim_carryover_at_batch_completion(step=3)

        kept_group_ids = {info.group_id for info in scheduler.inflight_requests.values()}
        assert len(scheduler.inflight_requests) == 2
        assert kept_group_ids == {0, 1}
        assert set(scheduler.groups) == {0, 1}
        assert scheduler.scheduler_cancelled_batch_complete == 2

        for task in scheduler.inflight_requests:
            task.cancel()
        await asyncio.gather(*scheduler.inflight_requests.keys(), return_exceptions=True)

    asyncio.run(run())


def test_stale_carryover_is_discarded_after_max_carryover_steps():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(async_scheduling=SimpleNamespace(max_carryover_steps=1))
    current = InflightRolloutInfo(
        off_policy_steps=0,
        client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
        task="rlm_rlvr",
        scheduled_step=9,
    )
    one_step_old = InflightRolloutInfo(
        off_policy_steps=0,
        client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
        task="rlm_rlvr",
        scheduled_step=8,
    )
    two_steps_old = InflightRolloutInfo(
        off_policy_steps=0,
        client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
        task="rlm_rlvr",
        scheduled_step=7,
    )

    assert scheduler._is_stale_carryover(current, step=9) is False
    assert scheduler._is_stale_carryover(one_step_old, step=9) is False
    assert scheduler._is_stale_carryover(two_steps_old, step=9) is True


def test_completed_deferred_group_is_enqueued_for_background_scoring():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.deferred_group_scoring_tasks = {"rlm_rlvr"}
        scheduler.group_scoring_tasks = {}
        scheduler.group_scoring_semaphore = asyncio.Semaphore(1)
        scheduler.batch_size = 4
        scheduler.token_batch_size = None
        scheduler.rollouts_per_example = 1
        scheduler.current_phase = "train"
        scheduler.logger = MagicMock()
        scheduler.scheduler_attempts_finished = 0
        scheduler.scheduler_carryover_accepted = 0
        scheduler.total_rollouts_by_task = defaultdict(int)
        scheduler.empty_rollouts_by_task = defaultdict(int)
        scheduler.errored_rollouts_by_task = defaultdict(int)
        scheduler.attempts_by_task = defaultdict(int)
        scheduler.attempt_duration_seconds = []
        scheduler.attempt_wall_seconds = []
        scheduler.attempt_scheduler_consume_lag_seconds = []
        scheduler.group_scoring_started = 0
        scheduler.group_scoring_runtime_seconds = []
        scheduler.group_scoring_queue_wait_seconds = []
        scheduler.group_scoring_lag_seconds = []
        scheduler.config = SimpleNamespace(
            rollout_timeout_seconds=400,
            verification=SimpleNamespace(enabled=True),
            group_scoring=SimpleNamespace(enabled=True, max_concurrency=1, max_pending_groups=64),
            async_scheduling=SimpleNamespace(max_carryover_steps=1),
            attempt_logging=SimpleNamespace(enabled=False),
            env_worker_recovery=SimpleNamespace(enabled=False),
            output_dir=None,
        )
        scheduler.buffer = MagicMock()
        scheduler._release_worker_reservation = AsyncMock()

        rollout = vf.RolloutOutput(
            example_id=12,
            task="rlm_rlvr",
            trajectory=[{"role": "assistant", "content": "FINAL(1)"}],
            error=None,
            reward=0.0,
            metrics={},
        )
        task = asyncio.create_task(asyncio.sleep(0, result=rollout))
        await task
        scheduler.task_done_perf = {task: time.perf_counter()}
        scheduler.groups = {
            7: GroupState(
                example={"task": "rlm_rlvr", "example_id": 12, "prompt": []},
                pending_slots=deque([]),
            )
        }
        scheduler.inflight_requests = {
            task: InflightRolloutInfo(
                off_policy_steps=0,
                client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
                task="rlm_rlvr",
                group_id=7,
                slot_index=0,
                attempt_number=1,
                start_time_perf=time.perf_counter() - 0.1,
                scheduled_step=3,
            )
        }

        score_started = asyncio.Event()

        async def fake_score_group(rollouts):
            score_started.set()
            await asyncio.sleep(60)
            return rollouts

        scheduler._score_group_if_deferred = fake_score_group

        progress = await scheduler._process_finished_task(
            task,
            step=3,
            batch_rollouts=[],
            batch_progress=0,
            pbar=MagicMock(),
        )

        assert progress == 0
        assert 7 not in scheduler.groups
        assert len(scheduler.group_scoring_tasks) == 1
        assert scheduler.group_scoring_started == 1
        scheduler.buffer.update.assert_not_called()
        scheduler._release_worker_reservation.assert_awaited()
        await score_started.wait()

        scoring_task = next(iter(scheduler.group_scoring_tasks))
        scoring_task.cancel()
        await asyncio.gather(scoring_task, return_exceptions=True)

    asyncio.run(run())


def test_finished_background_group_scoring_updates_buffer():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.group_scoring_tasks = {}
        scheduler.group_scoring_semaphore = asyncio.Semaphore(1)
        scheduler.batch_size = 4
        scheduler.token_batch_size = None
        scheduler.rollouts_per_example = 1
        scheduler.current_phase = "train"
        scheduler.logger = MagicMock()
        scheduler.group_scoring_finished = 0
        scheduler.group_scoring_failed = 0
        scheduler.group_scoring_stale = 0
        scheduler.group_scoring_runtime_seconds = []
        scheduler.group_scoring_queue_wait_seconds = []
        scheduler.group_scoring_lag_seconds = []
        scheduler.config = SimpleNamespace(
            async_scheduling=SimpleNamespace(max_carryover_steps=1),
            env_worker_recovery=SimpleNamespace(enabled=False),
            attempt_logging=SimpleNamespace(enabled=False),
            output_dir=None,
        )
        update_calls = []

        class FakeBuffer:
            rollout_buffer = []

            def update(self, rollouts, step=None):
                update_calls.append((rollouts, step))

        scheduler.buffer = FakeBuffer()

        rollout = vf.RolloutOutput(
            example_id=12,
            task="rlm_rlvr",
            trajectory=[{"role": "assistant", "content": "FINAL(1)"}],
            error=None,
            reward=1.0,
            metrics={},
        )

        async def scoring_result():
            start = time.perf_counter()
            await asyncio.sleep(0)
            end = time.perf_counter()
            return [rollout], start, end

        task = asyncio.create_task(scoring_result())
        scheduler.group_scoring_tasks[task] = scheduler_module.GroupScoringInfo(
            group_id=7,
            scheduled_step=3,
            source_id="src-12",
            example_id=12,
            dataset_name="frames",
            task="rlm_rlvr",
            created_time_perf=time.perf_counter(),
            created_time_iso="now",
            rollout_count=1,
            example={"task": "rlm_rlvr", "example_id": 12, "prompt": []},
        )
        await task

        progress = await scheduler._process_finished_scoring_task(
            task,
            step=3,
            batch_rollouts=[],
            batch_progress=4,
            pbar=MagicMock(),
        )

        assert progress == 4
        assert update_calls == [([rollout], 3)]
        assert scheduler.group_scoring_finished == 1
        assert scheduler.group_scoring_tasks == {}
        assert scheduler.group_scoring_runtime_seconds
        assert scheduler.group_scoring_queue_wait_seconds
        assert scheduler.group_scoring_lag_seconds

    asyncio.run(run())


def test_scheduling_pauses_when_group_scoring_pending_cap_is_reached():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.config = SimpleNamespace(group_scoring=SimpleNamespace(enabled=True, max_pending_groups=1))
        scheduler.group_scoring_tasks = {asyncio.create_task(asyncio.sleep(60)): object()}
        scheduler.group_scoring_max_pending_reached = 0
        scheduler.max_inflight_rollouts = 100
        scheduler.inflight_requests = {}

        scheduled = await scheduler._schedule_next_request(allowed_inflight=10)

        assert scheduled is False
        assert scheduler.group_scoring_max_pending_reached == 1

        for task in scheduler.group_scoring_tasks:
            task.cancel()
        await asyncio.gather(*scheduler.group_scoring_tasks.keys(), return_exceptions=True)

    asyncio.run(run())
