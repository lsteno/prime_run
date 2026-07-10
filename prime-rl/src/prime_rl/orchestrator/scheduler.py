from __future__ import annotations

import asyncio
import json
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import verifiers as vf
from aiolimiter import AsyncLimiter

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.utils import get_sampling_args
from prime_rl.orchestrator.vf_utils import (
    EnvWorkerReservation,
    ManagedEnvClientPool,
    get_seq_len,
    make_timeout_rollout,
    run_rollout,
)
from prime_rl.utils.async_utils import safe_cancel, safe_cancel_all
from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import ProgressTracker, get_logger
from prime_rl.utils.temp_scheduling import compute_temperature
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_latest_ckpt_step,
    get_step_path,
    wait_for_path,
)


@dataclass(frozen=True)
class InflightRolloutInfo:
    """Metadata for an in-flight request."""

    off_policy_steps: int
    client_config: vf.ClientConfig
    task: str
    group_id: int | None = None
    attempt_id: str | None = None
    slot_index: int | None = None
    attempt_number: int = 1
    worker_id: int | None = None
    worker_generation: int | None = None
    env_worker_name: str | None = None
    env_address: str | None = None
    source_id: str | None = None
    example_id: int | None = None
    dataset_name: str | None = None
    start_time_perf: float = 0.0
    start_time_iso: str | None = None
    reschedule_count: int = 0
    policy_ckpt_step: int = 0
    worker_reservation: EnvWorkerReservation | None = None
    scheduled_step: int = 0


@dataclass
class GroupState:
    """Tracks the state of a rollout group (one example × N rollouts)."""

    example: dict
    pending_slots: deque[int]
    completed_rollouts: dict[int, vf.RolloutOutput] = field(default_factory=dict)
    attempts_by_slot: dict[int, int] = field(default_factory=dict)
    dropped: bool = False
    pinned_client: vf.ClientConfig | None = None


@dataclass(frozen=True)
class GroupScoringInfo:
    """Metadata for a completed group being scored off the rollout hot path."""

    group_id: int
    scheduled_step: int
    source_id: str | None
    example_id: int | None
    dataset_name: str | None
    task: str
    created_time_perf: float
    created_time_iso: str
    rollout_count: int
    example: dict


class Scheduler:
    """
    Asynchronously manages scheduling of rollout requests and policy updates.
    Keeps a constant number of rollouts in-flight (continuous batching) and
    updates the policy as soon as it becomes available.

    References:
    - AReal: https://arxiv.org/abs/2505.24298v1
    - PipelineRL: https://arxiv.org/abs/2509.19128v1
    """

    def __init__(
        self,
        env: vf.Environment,
        inference_pool: InferencePool,
        buffer: Buffer,
        config: OrchestratorConfig,
        max_inflight_rollouts: int,
        max_async_level: int,
        max_off_policy_steps: int,
        strict_async_level: bool,
        tasks_per_minute: int | None,
        lora_name: str | None = None,
        deferred_group_scoring_tasks: set[str] | None = None,
    ):
        self.logger = get_logger()
        if tasks_per_minute is not None:
            self.rate_limiter = AsyncLimiter(max_rate=tasks_per_minute, time_period=60)
        else:
            self.rate_limiter = None
        self.env = env
        self.buffer = buffer
        self.config = config
        self.batch_size = config.batch_size
        self.token_batch_size = config.token_batch_size
        self.rollouts_per_example = config.rollouts_per_example
        self.max_inflight_rollouts = max_inflight_rollouts
        self.max_async_level = max_async_level
        self.max_off_policy_steps = max_off_policy_steps
        self.strict_async_level = strict_async_level
        self.lora_name = lora_name
        initial_temp = compute_temperature(step=0, sampling_config=config.sampling, max_steps=config.max_steps)
        self.sampling_args = get_sampling_args(config.sampling, temperature=initial_temp)
        self.model_name = self.config.model.name
        self.json_logging = config.log.json_logging
        self.max_retries_by_task = {env_config.resolved_name: env_config.max_retries for env_config in config.env}

        # Inference pool - used for admin operations (adapter sync) and metrics
        self.inference_pool = inference_pool

        self.deferred_group_scoring_tasks = set(deferred_group_scoring_tasks or ())
        if self.deferred_group_scoring_tasks:
            task_list = ", ".join(sorted(self.deferred_group_scoring_tasks))
            self.logger.info(f"Deferred group scoring active for task(s): {task_list}")

        # Track in-flight requests: task -> info
        self.inflight_requests: dict[asyncio.Task, InflightRolloutInfo] = {}
        self.task_done_perf: dict[asyncio.Task, float] = {}

        # Track in-progress groups while rollouts are generated independently.
        self.next_group_id = 0
        self.groups: dict[int, GroupState] = {}
        self.group_scoring_tasks: dict[asyncio.Task, GroupScoringInfo] = {}
        self.group_scoring_done_perf: dict[asyncio.Task, float] = {}
        self.group_scoring_semaphore = asyncio.Semaphore(config.group_scoring.max_concurrency)

        self.step, self.ckpt_step = 0, 0
        self.checkpoint_ready = asyncio.Event()
        self.checkpoint_ready.set()
        self.update_weights_time, self.wait_for_ckpt_time = 0, 0
        self.update_policy_task: asyncio.Task | None = None
        self.cancelled_rollouts_count = 0
        self.empty_rollouts_by_task: dict[str, int] = defaultdict(int)
        self.errored_rollouts_by_task: dict[str, int] = defaultdict(int)
        self.timeout_rollouts_by_task: dict[str, int] = defaultdict(int)
        self.total_rollouts_by_task: dict[str, int] = defaultdict(int)
        self.last_batch_generation_time = 0.0
        self.current_phase = "train"
        self.attempt_duration_seconds: list[float] = []
        self.attempt_wall_seconds: list[float] = []
        self.attempt_scheduler_consume_lag_seconds: list[float] = []
        self.attempt_timeouts = 0
        self.attempt_late_success_after_timeout = 0
        self.attempt_reschedules = 0
        self.attempt_group_drops = 0
        self.attempt_group_drops_first_timeout = 0
        self.timeout_cooldown_groups = 0
        self.attempts_by_task: dict[str, int] = defaultdict(int)
        self.attempt_timeouts_by_task: dict[str, int] = defaultdict(int)
        self.attempt_late_success_after_timeout_by_task: dict[str, int] = defaultdict(int)
        self.attempt_drops_by_task: dict[str, int] = defaultdict(int)
        self.attempt_first_timeout_drops_by_task: dict[str, int] = defaultdict(int)
        self.worker_restart_count = 0
        self.worker_restart_count_by_name: dict[str, int] = defaultdict(int)
        self.scheduler_attempts_started = 0
        self.scheduler_attempts_finished = 0
        self.scheduler_carryover_count = 0
        self.scheduler_carryover_accepted = 0
        self.scheduler_stale_after_batch_complete = 0
        self.scheduler_cancelled_batch_complete = 0
        self.scheduler_cancelled_batch_complete_by_worker: dict[str, int] = defaultdict(int)
        self.scheduler_allowed_inflight = max_inflight_rollouts
        self.group_scoring_started = 0
        self.group_scoring_finished = 0
        self.group_scoring_failed = 0
        self.group_scoring_stale = 0
        self.group_scoring_max_pending_reached = 0
        self.group_scoring_runtime_seconds: list[float] = []
        self.group_scoring_queue_wait_seconds: list[float] = []
        self.group_scoring_lag_seconds: list[float] = []

    @property
    def uses_token_batching(self) -> bool:
        return self.token_batch_size is not None

    @property
    def batch_target(self) -> int:
        if self.uses_token_batching:
            assert self.token_batch_size is not None
            return self.token_batch_size
        assert self.batch_size is not None
        return self.batch_size

    def get_batch_progress_increment(self, rollouts: list[vf.RolloutOutput]) -> int:
        if self.uses_token_batching:
            return sum(get_seq_len(rollout) for rollout in rollouts)
        return len(rollouts)

    def finalize_batch_rollouts(self, rollouts: list[vf.RolloutOutput]) -> list[vf.RolloutOutput]:
        if self.batch_size is None:
            return rollouts
        return rollouts[: self.batch_size]

    def set_sampling_args(self, sampling_args: dict) -> None:
        """Update sampling args for future rollout requests."""
        self.sampling_args = sampling_args

    async def cancel_inflight_rollouts(self):
        """Cancel all in-flight rollout requests."""
        count = len(self.inflight_requests)
        for info in list(self.inflight_requests.values()):
            await self._release_worker_reservation(info)
        await safe_cancel_all(list(self.inflight_requests))
        self.inflight_requests.clear()
        self.task_done_perf.clear()
        self.groups.clear()
        if getattr(self, "group_scoring_tasks", None):
            await safe_cancel_all(list(self.group_scoring_tasks))
            self.group_scoring_tasks.clear()
            self.group_scoring_done_perf.clear()
        self.cancelled_rollouts_count += count

    @staticmethod
    def _client_identity(c: vf.ClientConfig) -> tuple[str, str | None]:
        return (c.api_base_url, c.extra_headers.get("X-data-parallel-rank"))

    async def _select_least_loaded_client(self) -> vf.ClientConfig:
        """Select the client with the fewest in-flight tasks.

        Uses (api_base_url, dp_rank) as identity rather than client_idx so that
        load tracking survives elastic pool refreshes (which reassign indices).
        """
        clients = self.inference_pool.clients
        while not clients:
            await asyncio.sleep(1)
            clients = self.inference_pool.clients
        inflight = Counter(self._client_identity(info.client_config) for info in self.inflight_requests.values())
        return min(clients, key=lambda c: inflight[self._client_identity(c)])

    @staticmethod
    def _example_info(example: dict) -> dict[str, Any]:
        info = example.get("info") or {}
        if isinstance(info, dict):
            return info
        if isinstance(info, str):
            try:
                decoded = json.loads(info)
            except json.JSONDecodeError:
                return {}
            return decoded if isinstance(decoded, dict) else {}
        return {}

    def _source_id(self, example: dict) -> str | None:
        info = self._example_info(example)
        for key in ("source_id", "id", "row_id"):
            if key in example:
                return str(example[key])
            if key in info:
                return str(info[key])
        return None

    def _dataset_name(self, example: dict) -> str | None:
        info = self._example_info(example)
        for key in ("dataset", "dataset_name", "source_dataset"):
            if key in example:
                return str(example[key])
            if key in info:
                return str(info[key])
        return None

    def _attempt_log_path(self, step: int) -> Any:
        return self.config.output_dir / "attempts" / self.current_phase / f"step_{step:06d}.jsonl"

    def _log_attempt_event(self, event: str, payload: dict[str, Any]) -> None:
        if not self.config.attempt_logging.enabled:
            return
        record = {"event": event, **payload}
        path = self._attempt_log_path(int(payload.get("step", self.step)))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def _rollout_protocol_stats(rollout: vf.RolloutOutput) -> dict[str, Any]:
        metrics = rollout.get("metrics") or {}
        token_usage = rollout.get("token_usage") or {}
        return {
            "token_usage": token_usage,
            "used_repl": rollout.get("used_repl", metrics.get("used_repl")),
            "used_llm_subcalls": rollout.get("used_llm_subcalls", metrics.get("used_llm_subcalls")),
            "num_llm_subcalls": rollout.get("num_llm_subcalls", metrics.get("num_llm_subcalls")),
            "used_rlm_subcalls": rollout.get("used_rlm_subcalls", metrics.get("used_rlm_subcalls")),
            "num_rlm_subcalls": rollout.get("num_rlm_subcalls", metrics.get("num_rlm_subcalls")),
            "max_depth_reached": rollout.get("max_depth_reached", metrics.get("max_depth_reached")),
        }

    async def _release_worker_reservation(self, info: InflightRolloutInfo) -> None:
        if info.worker_reservation is not None:
            await info.worker_reservation.release()

    def _get_env_worker_pool(self, task: str) -> ManagedEnvClientPool | None:
        try:
            env_for_task = self.env.get_env_for_task(task)
        except Exception:
            env_for_task = self.env
        env_client = getattr(env_for_task, "env_client", None)
        if isinstance(env_client, ManagedEnvClientPool):
            return env_client
        return None

    def _env_worker_pools(self) -> list[ManagedEnvClientPool]:
        envs = getattr(self.env, "envs", None)
        if envs is None:
            envs = [self.env]
        pools: list[ManagedEnvClientPool] = []
        seen: set[int] = set()
        for env in envs:
            env_client = getattr(env, "env_client", None)
            if isinstance(env_client, ManagedEnvClientPool) and id(env_client) not in seen:
                pools.append(env_client)
                seen.add(id(env_client))
        return pools

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))))
        return ordered[idx]

    def _record_task_done(self, task: asyncio.Task) -> None:
        if task not in getattr(self, "inflight_requests", {}):
            return
        if not hasattr(self, "task_done_perf"):
            self.task_done_perf = {}
        self.task_done_perf[task] = time.perf_counter()

    def _attempt_timings(self, task: asyncio.Task | None, info: InflightRolloutInfo) -> dict[str, float]:
        log_time_perf = time.perf_counter()
        if not hasattr(self, "task_done_perf"):
            self.task_done_perf = {}
        done_time_perf = self.task_done_perf.pop(task, None) if task is not None else None
        if done_time_perf is None:
            done_time_perf = log_time_perf
        worker_runtime_s = max(0.0, done_time_perf - info.start_time_perf)
        wall_s = max(0.0, log_time_perf - info.start_time_perf)
        scheduler_consume_lag_s = max(0.0, log_time_perf - done_time_perf)
        return {
            "worker_runtime_s": worker_runtime_s,
            "scheduler_consume_lag_s": scheduler_consume_lag_s,
            "wall_s": wall_s,
        }

    def _worker_runtime_exceeded_timeout(self, task: asyncio.Task | None, info: InflightRolloutInfo) -> bool:
        timeout = self.config.rollout_timeout_seconds
        if timeout is None:
            return False
        if not hasattr(self, "task_done_perf"):
            self.task_done_perf = {}
        done_time_perf = self.task_done_perf.get(task) if task is not None else None
        if done_time_perf is None:
            return self._rollout_deadline_exceeded(info)
        return (done_time_perf - info.start_time_perf) >= timeout

    def _rollout_deadline_exceeded(self, info: InflightRolloutInfo, now: float | None = None) -> bool:
        timeout = self.config.rollout_timeout_seconds
        if timeout is None or info.start_time_perf <= 0:
            return False
        return ((now or time.perf_counter()) - info.start_time_perf) >= timeout

    def _timeout_rollout_for_info(self, info: InflightRolloutInfo) -> vf.RolloutOutput:
        example: dict[str, Any] = {"task": info.task}
        if info.group_id is not None and info.group_id in self.groups:
            example = self.groups[info.group_id].example
        elif info.example_id is not None:
            example["example_id"] = info.example_id
        timeout = self.config.rollout_timeout_seconds
        assert timeout is not None
        return make_timeout_rollout(example, self.sampling_args, timeout)

    async def drop_group(
        self,
        group_id: int,
        *,
        step: int | None = None,
        reason: str | None = None,
        cooldown_steps: int | None = None,
    ) -> int:
        """Drop a group and cancel any remaining in-flight rollouts for it."""
        tasks_to_cancel = []
        group = self.groups.pop(group_id, None)
        if group is not None:
            group.dropped = True
            if step is not None and self.config.env_worker_recovery.enabled:
                resolved_cooldown_steps = (
                    cooldown_steps
                    if cooldown_steps is not None
                    else self.config.env_worker_recovery.max_attempts_cooldown_steps
                )
                self.buffer.put_example_on_hard_cooldown(
                    group.example,
                    step=step,
                    cooldown_steps=resolved_cooldown_steps,
                )
                self._log_attempt_event(
                    "group_dropped",
                    {
                        "phase": self.current_phase,
                        "step": step,
                        "group_id": group_id,
                        "source_id": self._source_id(group.example),
                        "example_id": group.example.get("example_id"),
                        "dataset_name": self._dataset_name(group.example),
                        "task": group.example.get("task"),
                        "reason": reason or "group_dropped",
                        "group_completed_count": len(group.completed_rollouts),
                        "cooldown_steps": resolved_cooldown_steps,
                    },
                )
        for task, info in list(self.inflight_requests.items()):
            if info.group_id != group_id:
                continue
            self.inflight_requests.pop(task, None)
            self.task_done_perf.pop(task, None)
            await self._release_worker_reservation(info)
            tasks_to_cancel.append(task)
        await safe_cancel_all(tasks_to_cancel)
        return len(tasks_to_cancel)

    async def schedule_rollout(self, group_id: int) -> bool:
        """Asynchronously schedules a rollout request."""
        group = self.groups.get(group_id)
        if group is None or group.dropped or not group.pending_slots:
            return False
        worker_pool = self._get_env_worker_pool(group.example["task"])
        worker_reservation = None
        worker_token = None
        if self.config.env_worker_recovery.enabled and worker_pool is not None:
            async_scheduling = getattr(self.config, "async_scheduling", None)
            max_requests_per_worker = getattr(async_scheduling, "max_requests_per_env_worker", None)
            if max_requests_per_worker is None:
                worker_reservation = await worker_pool.reserve_worker()
            else:
                worker_reservation = await worker_pool.try_reserve_worker(
                    max_requests_per_worker=max_requests_per_worker
                )
                if worker_reservation is None:
                    return False
            worker_token = worker_pool.activate_reservation(worker_reservation)
        if self.rate_limiter:
            await self.rate_limiter.acquire()
            group = self.groups.get(group_id)
            if group is None or group.dropped or not group.pending_slots:
                if worker_reservation is not None:
                    await worker_reservation.release()
                if worker_pool is not None and worker_token is not None:
                    worker_pool.reset_active_reservation(worker_token)
                return False
        if group.pinned_client is not None:
            client_config = group.pinned_client
        else:
            client_config = await self._select_least_loaded_client()
            if group_id not in self.groups:
                if worker_reservation is not None:
                    await worker_reservation.release()
                if worker_pool is not None and worker_token is not None:
                    worker_pool.reset_active_reservation(worker_token)
                return False
            group.pinned_client = client_config
        if not group.pending_slots:
            if worker_reservation is not None:
                await worker_reservation.release()
            if worker_pool is not None and worker_token is not None:
                worker_pool.reset_active_reservation(worker_token)
            return False
        slot_index = group.pending_slots.popleft()
        attempt_number = group.attempts_by_slot.get(slot_index, 0) + 1
        group.attempts_by_slot[slot_index] = attempt_number
        attempt_id = uuid4().hex
        start_time_perf = time.perf_counter()
        start_time_iso = datetime.now(UTC).isoformat()
        info = InflightRolloutInfo(
            off_policy_steps=0,
            client_config=client_config,
            task=group.example["task"],
            group_id=group_id,
            attempt_id=attempt_id,
            slot_index=slot_index,
            attempt_number=attempt_number,
            worker_id=worker_reservation.worker_id if worker_reservation else None,
            worker_generation=worker_reservation.worker_generation if worker_reservation else None,
            env_worker_name=worker_reservation.worker_name if worker_reservation else None,
            env_address=worker_reservation.address if worker_reservation else None,
            source_id=self._source_id(group.example),
            example_id=group.example.get("example_id"),
            dataset_name=self._dataset_name(group.example),
            start_time_perf=start_time_perf,
            start_time_iso=start_time_iso,
            reschedule_count=max(0, attempt_number - 1),
            policy_ckpt_step=self.ckpt_step,
            worker_reservation=worker_reservation,
            scheduled_step=self.step,
        )
        self.scheduler_attempts_started = getattr(self, "scheduler_attempts_started", 0) + 1
        self._log_attempt_event(
            "attempt_started",
            {
                "attempt_id": attempt_id,
                "phase": self.current_phase,
                "step": self.step,
                "scheduled_step": self.step,
                "group_id": group_id,
                "slot_index": slot_index,
                "attempt_number": attempt_number,
                "source_id": info.source_id,
                "example_id": info.example_id,
                "dataset_name": info.dataset_name,
                "task": info.task,
                "worker_id": info.worker_id,
                "worker_name": info.env_worker_name,
                "worker_generation": info.worker_generation,
                "env_address": info.env_address,
                "inference_client": self._client_identity(client_config),
                "policy_ckpt_step": info.policy_ckpt_step,
                "off_policy_steps_at_start": info.off_policy_steps,
                "start_time_iso": start_time_iso,
                "timeout_seconds": self.config.rollout_timeout_seconds,
                "reschedule_count": info.reschedule_count,
                "group_completed_count": len(group.completed_rollouts),
            },
        )
        async def run_rollout_and_record_done() -> vf.RolloutOutput:
            try:
                return await run_rollout(
                    env=self.env,
                    client=client_config,
                    example=group.example,
                    model_name=self.model_name,
                    sampling_args=self.sampling_args,
                    max_retries=self.max_retries_by_task.get(group.example["task"], 0),
                    rollout_timeout_seconds=self.config.rollout_timeout_seconds,
                )
            finally:
                current_task = asyncio.current_task()
                if current_task is not None:
                    self._record_task_done(current_task)

        run_rollout_task = asyncio.create_task(run_rollout_and_record_done())
        if worker_pool is not None and worker_token is not None:
            worker_pool.reset_active_reservation(worker_token)
        self.inflight_requests[run_rollout_task] = info
        return True

    @property
    def inflight_rollout_count(self) -> int:
        return len(self.inflight_requests)

    @property
    def inflight_sample_count(self) -> int:
        return self.inflight_rollout_count + sum(len(g.pending_slots) for g in self.groups.values())

    def _allowed_inflight_for_progress(self, batch_progress: int) -> int:
        cushion = self.config.async_scheduling.inflight_completion_cushion
        if cushion is None:
            allowed = self.max_inflight_rollouts
        else:
            remaining_needed = max(0, self.batch_target - batch_progress)
            allowed = min(self.max_inflight_rollouts, remaining_needed + cushion)
        self.scheduler_allowed_inflight = allowed
        return allowed

    async def _schedule_next_request(self, *, allowed_inflight: int | None = None) -> bool:
        group_scoring_config = getattr(getattr(self, "config", None), "group_scoring", None)
        if (
            group_scoring_config is not None
            and group_scoring_config.enabled
            and len(getattr(self, "group_scoring_tasks", {})) >= group_scoring_config.max_pending_groups
        ):
            self.group_scoring_max_pending_reached = getattr(self, "group_scoring_max_pending_reached", 0) + 1
            return False

        inflight_cap = self.max_inflight_rollouts if allowed_inflight is None else allowed_inflight
        remaining_capacity = inflight_cap - self.inflight_rollout_count

        if remaining_capacity <= 0:
            return False

        for group_id, group in self.groups.items():
            if group.pending_slots and not group.dropped:
                return await self.schedule_rollout(group_id=group_id)

        example = self.buffer.sample_examples(n=1)[0]
        group_id = self.next_group_id
        self.next_group_id += 1
        self.groups[group_id] = GroupState(example=example, pending_slots=deque(range(self.rollouts_per_example)))
        return await self.schedule_rollout(group_id=group_id)

    async def _fill_inflight_requests(self, *, allowed_inflight: int | None = None) -> None:
        while await self._schedule_next_request(allowed_inflight=allowed_inflight):
            pass

    async def update_policy_loop(self):
        """Continuously checks for new policy checkpoints."""
        while True:
            await self.maybe_update_policy()
            await asyncio.sleep(1)

    async def maybe_update_policy(self):
        """Updates the policy to the latest available checkpoint. Aborts rollout requests that are older than the max retention steps."""
        latest_ckpt_step = get_latest_ckpt_step(get_broadcast_dir(self.config.output_dir)) or 0
        async_away_ckpt_step = max(self.step - self.max_async_level, 0)
        next_ckpt_step = (
            async_away_ckpt_step if self.strict_async_level else max(async_away_ckpt_step, latest_ckpt_step)
        )

        if next_ckpt_step > self.ckpt_step:
            if next_ckpt_step == async_away_ckpt_step:
                self.logger.info(
                    f"Orchestrator paused: waiting for trainer process to complete checkpoint {next_ckpt_step} "
                    f"(>{self.max_async_level} step(s) ahead). Training is progressing normally."
                )
                self.checkpoint_ready.clear()
                wait_for_ckpt_start_time = time.perf_counter()
                await wait_for_path(get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step) / "STABLE")
                self.wait_for_ckpt_time = time.perf_counter() - wait_for_ckpt_start_time
                self.logger.info(
                    f"Orchestrator resumed: checkpoint {next_ckpt_step} ready (after {self.wait_for_ckpt_time:.2f}s)"
                )

            self.logger.debug(
                f"Got new policy with step {next_ckpt_step}. Updating weights and cancelling old rollout requests."
            )

            # Update weights on inference servers
            update_weights_start_time = time.perf_counter()
            weights_path = get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step)
            await self.inference_pool.update_weights(weights_path, lora_name=self.lora_name, step=next_ckpt_step)
            self.update_weights_time = time.perf_counter() - update_weights_start_time
            self.logger.debug(f"Updated weights to step {next_ckpt_step} in {self.update_weights_time:.2f}s")

            if self.lora_name is not None:
                self.model_name = self.lora_name
                self.inference_pool.update_model_name(self.model_name)

            self.checkpoint_ready.set()

            await self._update_off_policy()
            self.ckpt_step = next_ckpt_step

    async def _update_off_policy(self) -> None:
        stale_group_ids = {
            info.group_id
            for info in self.inflight_requests.values()
            if info.group_id is not None and info.off_policy_steps >= self.max_off_policy_steps
        }
        tasks_to_increment = [
            task
            for task, info in list(self.inflight_requests.items())
            if info.group_id is None or info.group_id not in stale_group_ids
        ]

        counts = await asyncio.gather(*(self.drop_group(gid) for gid in stale_group_ids))
        removed = sum(counts)
        for task in tasks_to_increment:
            info = self.inflight_requests.get(task)
            if info is None:
                continue
            self.inflight_requests[task] = replace(info, off_policy_steps=info.off_policy_steps + 1)

        self.cancelled_rollouts_count += removed
        if removed:
            self.logger.warning(
                f"Cancelled {removed} old rollout requests (will refill naturally). "
                f"Consider increasing max_off_policy_steps to avoid this."
            )

    def _should_defer_group_scoring(self, task: str) -> bool:
        return task in self.deferred_group_scoring_tasks and self.config.verification.enabled

    def _should_score_group_in_background(self, task: str) -> bool:
        group_scoring_config = getattr(self.config, "group_scoring", None)
        return (
            group_scoring_config is not None
            and group_scoring_config.enabled
            and self._should_defer_group_scoring(task)
        )

    async def _score_group_if_deferred(self, completed_rollouts: list[vf.RolloutOutput]) -> list[vf.RolloutOutput]:
        if not completed_rollouts:
            return completed_rollouts
        task = completed_rollouts[0]["task"]
        if not self._should_defer_group_scoring(task):
            return completed_rollouts
        env_for_task = self.env.get_env_for_task(task)
        await env_for_task.rubric.score_group(cast(list[vf.State], completed_rollouts))
        return completed_rollouts

    async def _score_group_with_semaphore(
        self,
        completed_rollouts: list[vf.RolloutOutput],
    ) -> tuple[list[vf.RolloutOutput], float, float]:
        async with self.group_scoring_semaphore:
            start_time = time.perf_counter()
            scored_rollouts = await self._score_group_if_deferred(completed_rollouts)
            end_time = time.perf_counter()
            return scored_rollouts, start_time, end_time

    def _enqueue_group_scoring(
        self,
        *,
        group_id: int,
        example: dict,
        completed_rollouts: list[vf.RolloutOutput],
        scheduled_step: int,
    ) -> None:
        if not hasattr(self, "group_scoring_tasks"):
            self.group_scoring_tasks = {}
        if not hasattr(self, "group_scoring_done_perf"):
            self.group_scoring_done_perf = {}
        if not hasattr(self, "group_scoring_runtime_seconds"):
            self.group_scoring_runtime_seconds = []
        if not hasattr(self, "group_scoring_queue_wait_seconds"):
            self.group_scoring_queue_wait_seconds = []
        if not hasattr(self, "group_scoring_lag_seconds"):
            self.group_scoring_lag_seconds = []
        info = GroupScoringInfo(
            group_id=group_id,
            scheduled_step=scheduled_step,
            source_id=self._source_id(example),
            example_id=example.get("example_id"),
            dataset_name=self._dataset_name(example),
            task=example["task"],
            created_time_perf=time.perf_counter(),
            created_time_iso=datetime.now(UTC).isoformat(),
            rollout_count=len(completed_rollouts),
            example=example,
        )
        task = asyncio.create_task(self._score_group_with_semaphore(completed_rollouts))
        task.add_done_callback(lambda t: self.group_scoring_done_perf.setdefault(t, time.perf_counter()))
        self.group_scoring_tasks[task] = info
        self.group_scoring_started = getattr(self, "group_scoring_started", 0) + 1

    async def _process_finished_scoring_task(
        self,
        finished_task: asyncio.Task,
        *,
        step: int,
        batch_rollouts: list[vf.RolloutOutput],
        batch_progress: int,
        pbar: ProgressTracker,
    ) -> int:
        info = self.group_scoring_tasks.pop(finished_task, None)
        if info is None:
            return batch_progress
        consume_time = time.perf_counter()
        done_time = getattr(self, "group_scoring_done_perf", {}).pop(finished_task, None)
        if done_time is None:
            done_time = consume_time
        lag_s = max(0.0, consume_time - done_time)
        if not hasattr(self, "group_scoring_lag_seconds"):
            self.group_scoring_lag_seconds = []
        self.group_scoring_lag_seconds.append(lag_s)
        if (step - info.scheduled_step) > self.config.async_scheduling.max_carryover_steps:
            self.group_scoring_stale = getattr(self, "group_scoring_stale", 0) + 1
            self._log_attempt_event(
                "group_scoring_stale",
                {
                    "phase": self.current_phase,
                    "step": step,
                    "scheduled_step": info.scheduled_step,
                    "group_id": info.group_id,
                    "source_id": info.source_id,
                    "example_id": info.example_id,
                    "dataset_name": info.dataset_name,
                    "task": info.task,
                    "scheduler_lag_ms": lag_s * 1000.0,
                    "rollout_count": info.rollout_count,
                },
            )
            return batch_progress
        try:
            scored_rollouts, scoring_start_time, scoring_end_time = finished_task.result()
        except asyncio.CancelledError:
            return batch_progress
        except Exception as e:
            self.group_scoring_failed = getattr(self, "group_scoring_failed", 0) + 1
            self.logger.warning(f"Group scoring failed for group {info.group_id} ({info.task}): {e}")
            self._log_attempt_event(
                "group_scoring_failed",
                {
                    "phase": self.current_phase,
                    "step": step,
                    "scheduled_step": info.scheduled_step,
                    "group_id": info.group_id,
                    "source_id": info.source_id,
                    "example_id": info.example_id,
                    "dataset_name": info.dataset_name,
                    "task": info.task,
                    "error": repr(e),
                    "scheduler_lag_ms": lag_s * 1000.0,
                    "rollout_count": info.rollout_count,
                },
            )
            if self.config.env_worker_recovery.enabled:
                self.buffer.put_example_on_hard_cooldown(
                    info.example,
                    step=step,
                    cooldown_steps=self.config.env_worker_recovery.max_attempts_cooldown_steps,
                )
            return batch_progress

        runtime_s = max(0.0, scoring_end_time - scoring_start_time)
        queue_wait_s = max(0.0, scoring_start_time - info.created_time_perf)
        if not hasattr(self, "group_scoring_runtime_seconds"):
            self.group_scoring_runtime_seconds = []
        if not hasattr(self, "group_scoring_queue_wait_seconds"):
            self.group_scoring_queue_wait_seconds = []
        self.group_scoring_runtime_seconds.append(runtime_s)
        self.group_scoring_queue_wait_seconds.append(queue_wait_s)
        self.group_scoring_finished = getattr(self, "group_scoring_finished", 0) + 1
        self._log_attempt_event(
            "group_scoring_finished",
            {
                "phase": self.current_phase,
                "step": step,
                "scheduled_step": info.scheduled_step,
                "group_id": info.group_id,
                "source_id": info.source_id,
                "example_id": info.example_id,
                "dataset_name": info.dataset_name,
                "task": info.task,
                "duration_ms": runtime_s * 1000.0,
                "queue_wait_ms": queue_wait_s * 1000.0,
                "scheduler_lag_ms": lag_s * 1000.0,
                "rollout_count": info.rollout_count,
            },
        )
        self.buffer.update(scored_rollouts, step=step)
        if batch_progress < self.batch_target:
            batch_progress = self._consume_rollout_buffer_into_batch(
                batch_rollouts=batch_rollouts,
                batch_progress=batch_progress,
                pbar=pbar,
            )
        return batch_progress

    async def _drain_done_group_scoring_tasks(
        self,
        *,
        step: int,
        batch_rollouts: list[vf.RolloutOutput],
        batch_progress: int,
        pbar: ProgressTracker,
    ) -> int:
        while True:
            done_tasks = [task for task in list(getattr(self, "group_scoring_tasks", {})) if task.done()]
            if not done_tasks:
                return batch_progress
            for task in done_tasks:
                batch_progress = await self._process_finished_scoring_task(
                    task,
                    step=step,
                    batch_rollouts=batch_rollouts,
                    batch_progress=batch_progress,
                    pbar=pbar,
                )

    async def _cancel_inflight_for_worker_generation(
        self,
        *,
        worker_id: int,
        worker_generation: int,
        step: int,
        reason: str,
        await_cancellation: bool = True,
    ) -> int:
        tasks_to_cancel = []
        affected_items: list[tuple[asyncio.Task, InflightRolloutInfo]] = []
        for task, info in list(self.inflight_requests.items()):
            if info.worker_id != worker_id or info.worker_generation != worker_generation:
                continue
            self.inflight_requests.pop(task, None)
            tasks_to_cancel.append(task)
            affected_items.append((task, info))
            await self._release_worker_reservation(info)

        for task, info in affected_items:
            self._log_attempt_finished(info, step=step, status="cancelled_for_worker_restart", task=task)
            await self._reschedule_or_drop_slot(info, step=step, reason=reason, count_as_reschedule=True)

        if await_cancellation:
            await safe_cancel_all(tasks_to_cancel)
        else:
            for task in tasks_to_cancel:
                task.cancel()
        return len(tasks_to_cancel)

    async def _restart_worker_generation(
        self,
        info: InflightRolloutInfo,
        *,
        step: int,
        reason: str,
        cancel_grace_seconds: float | None = None,
        cancel_existing_inflight: bool = True,
    ) -> None:
        if (
            not self.config.env_worker_recovery.enabled
            or info.worker_id is None
            or info.worker_generation is None
        ):
            return
        worker_pool = self._get_env_worker_pool(info.task)
        if worker_pool is None:
            return

        self.worker_restart_count += 1
        if info.env_worker_name:
            self.worker_restart_count_by_name[info.env_worker_name] += 1
        self._log_attempt_event(
            "worker_restart_started",
            {
                "phase": self.current_phase,
                "step": step,
                "worker_id": info.worker_id,
                "worker_name": info.env_worker_name,
                "worker_generation": info.worker_generation,
                "env_address": info.env_address,
                "reason": reason,
            },
        )
        if cancel_existing_inflight:
            await self._cancel_inflight_for_worker_generation(
                worker_id=info.worker_id,
                worker_generation=info.worker_generation,
                step=step,
                reason="worker_restart",
                await_cancellation=False,
            )
        try:
            handle = await worker_pool.restart_worker(
                worker_id=info.worker_id,
                expected_generation=info.worker_generation,
                reason=reason,
                cancel_grace_seconds=(
                    self.config.env_worker_recovery.cancel_grace_seconds
                    if cancel_grace_seconds is None
                    else cancel_grace_seconds
                ),
            )
            self._log_attempt_event(
                "worker_restart_finished",
                {
                    "phase": self.current_phase,
                    "step": step,
                    "worker_id": info.worker_id,
                    "worker_name": info.env_worker_name,
                    "old_worker_generation": info.worker_generation,
                    "worker_generation": handle.generation if handle is not None else info.worker_generation,
                    "env_address": handle.address if handle is not None else info.env_address,
                    "status": "success" if handle is not None else "skipped",
                    "reason": reason,
                },
            )
        except Exception as exc:
            self._log_attempt_event(
                "worker_restart_finished",
                {
                    "phase": self.current_phase,
                    "step": step,
                    "worker_id": info.worker_id,
                    "worker_name": info.env_worker_name,
                    "old_worker_generation": info.worker_generation,
                    "env_address": info.env_address,
                    "status": "error",
                    "error": repr(exc),
                    "reason": reason,
                },
            )
            self.logger.warning(f"Failed to restart env worker {info.env_worker_name}: {exc!r}")

    async def _restart_worker_for_timeout(self, info: InflightRolloutInfo, step: int) -> None:
        if not getattr(self.config.env_worker_recovery, "restart_on_rollout_timeout", True):
            return
        await self._restart_worker_generation(
            info,
            step=step,
            reason="rollout_timeout",
            cancel_existing_inflight=True,
        )

    def _should_drop_group_on_first_timeout(self) -> bool:
        return (
            self.current_phase == "train"
            and getattr(self.config.env_worker_recovery, "enabled", False)
            and getattr(self.config.env_worker_recovery, "drop_group_on_first_timeout", False)
        )

    async def _drop_group_on_timeout(self, info: InflightRolloutInfo, *, step: int, reason: str) -> None:
        if info.group_id is None:
            return
        if info.group_id not in self.groups:
            return
        cooldown_steps = self.config.env_worker_recovery.first_timeout_cooldown_steps
        if not hasattr(self, "attempt_group_drops"):
            self.attempt_group_drops = 0
        if not hasattr(self, "attempt_group_drops_first_timeout"):
            self.attempt_group_drops_first_timeout = 0
        if not hasattr(self, "timeout_cooldown_groups"):
            self.timeout_cooldown_groups = 0
        if not hasattr(self, "attempt_drops_by_task"):
            self.attempt_drops_by_task = defaultdict(int)
        if not hasattr(self, "attempt_first_timeout_drops_by_task"):
            self.attempt_first_timeout_drops_by_task = defaultdict(int)
        self.attempt_group_drops += 1
        self.attempt_group_drops_first_timeout += 1
        self.timeout_cooldown_groups += 1
        self.attempt_drops_by_task[info.task] += 1
        self.attempt_first_timeout_drops_by_task[info.task] += 1
        cancelled_count = await self.drop_group(
            info.group_id,
            step=step,
            reason=reason,
            cooldown_steps=cooldown_steps,
        )
        self.logger.warning(
            f"Dropped group {info.group_id} ({info.task}) after first timeout; "
            f"cancelled {cancelled_count} sibling attempt(s) and cooled down example for "
            f"{cooldown_steps} step(s). reason={reason}"
        )

    async def _handle_rollout_timeout(
        self,
        task: asyncio.Task,
        info: InflightRolloutInfo,
        *,
        step: int,
        reason: str,
        rollout: vf.RolloutOutput | None = None,
    ) -> None:
        """Enforce a rollout timeout from scheduler-visible wall-clock state."""
        self.inflight_requests.pop(task, None)
        await self._release_worker_reservation(info)
        if not task.done():
            task.cancel()

        timeout_rollout = rollout if rollout is not None else self._timeout_rollout_for_info(info)
        self.total_rollouts_by_task[info.task] += 1
        self.timeout_rollouts_by_task[info.task] += 1
        self.empty_rollouts_by_task[info.task] += int(len(timeout_rollout["trajectory"]) == 0)
        self.errored_rollouts_by_task[info.task] += int(timeout_rollout["error"] is not None)
        self._log_attempt_finished(info, step=step, status="timeout", rollout=timeout_rollout, task=task)
        await self._restart_worker_for_timeout(info, step=step)
        if self._should_drop_group_on_first_timeout():
            await self._drop_group_on_timeout(info, step=step, reason=reason)
            return
        await self._reschedule_or_drop_slot(
            info,
            step=step,
            reason=reason,
            count_as_reschedule=True,
        )
        group = self.groups.get(info.group_id) if info.group_id is not None else None
        self.logger.warning(
            f"Rollout timeout in group {info.group_id} ({info.task}) after "
            f"{self.config.rollout_timeout_seconds}s, re-scheduling "
            f"({len(group.completed_rollouts) if group is not None else 0}/{self.rollouts_per_example} complete, "
            f"attempt {info.attempt_number}/"
            f"{self.config.env_worker_recovery.max_rollout_attempts_per_slot}; reason={reason})"
        )

    async def _enforce_rollout_deadlines(self, *, step: int) -> None:
        if self.config.rollout_timeout_seconds is None:
            return
        now = time.perf_counter()
        candidates = [
            (task, info)
            for task, info in list(self.inflight_requests.items())
            if self._rollout_deadline_exceeded(info, now=now)
        ]
        for task, info in candidates:
            if task not in self.inflight_requests:
                continue
            rollout: vf.RolloutOutput | None = None
            if task.done():
                try:
                    result = task.result()
                except (asyncio.CancelledError, Exception):
                    continue
                if result.get("stop_condition") != "rollout_timeout":
                    if self._worker_runtime_exceeded_timeout(task, info):
                        self.inflight_requests.pop(task, None)
                        await self._release_worker_reservation(info)
                        self._log_attempt_finished(
                            info,
                            step=step,
                            status="late_success_after_timeout",
                            rollout=result,
                            error="worker_runtime_exceeded_timeout",
                            task=task,
                        )
                        await self._restart_worker_for_timeout(info, step=step)
                        if self._should_drop_group_on_first_timeout():
                            await self._drop_group_on_timeout(
                                info,
                                step=step,
                                reason="late_success_after_timeout",
                            )
                        else:
                            await self._reschedule_or_drop_slot(
                                info,
                                step=step,
                                reason="late_success_after_timeout",
                                count_as_reschedule=True,
                            )
                    continue
                rollout = result
            await self._handle_rollout_timeout(
                task,
                info,
                step=step,
                reason="scheduler_wall_clock_timeout",
                rollout=rollout,
            )

    def _is_stale_carryover(self, info: InflightRolloutInfo, *, step: int) -> bool:
        return (step - info.scheduled_step) > self.config.async_scheduling.max_carryover_steps

    @staticmethod
    def _worker_generation_key(info: InflightRolloutInfo) -> tuple[int, int] | None:
        if info.worker_id is None or info.worker_generation is None:
            return None
        return (info.worker_id, info.worker_generation)

    def _select_carryover_tasks(self, *, max_count: int) -> set[asyncio.Task]:
        """Select carryover attempts, preferring older worker generations.

        We keep or cancel by worker generation when possible. That avoids leaving a cancelled
        request queued ahead of a kept request on the same single-threaded env worker.
        """
        if max_count <= 0:
            return set()

        by_worker_generation: dict[object, list[tuple[asyncio.Task, InflightRolloutInfo]]] = defaultdict(list)
        for task, info in self.inflight_requests.items():
            key: object = self._worker_generation_key(info)
            if key is None:
                key = ("task", id(task))
            by_worker_generation[key].append((task, info))

        worker_groups = sorted(
            by_worker_generation.values(),
            key=lambda items: min(info.start_time_perf for _, info in items),
        )
        keep: set[asyncio.Task] = set()
        for items in worker_groups:
            # Prefer whole worker generations. If the first generation is already larger than the cap,
            # keep its oldest attempts rather than carrying over nothing.
            if len(keep) + len(items) <= max_count:
                keep.update(task for task, _ in items)
            elif not keep:
                oldest_items = sorted(items, key=lambda item: item[1].start_time_perf)[:max_count]
                keep.update(task for task, _ in oldest_items)
                break
        return keep

    async def _restart_worker_for_batch_cancel(
        self,
        info: InflightRolloutInfo,
        *,
        step: int,
        reason: str,
    ) -> None:
        if not self.config.async_scheduling.restart_workers_for_stale_cancel:
            return
        await self._restart_worker_generation(
            info,
            step=step,
            reason=reason,
            cancel_grace_seconds=self.config.async_scheduling.batch_complete_cancel_grace_seconds,
            cancel_existing_inflight=False,
        )

    async def _drop_groups_without_kept_carryover(
        self,
        *,
        kept_group_ids: set[int],
        reason: str,
    ) -> None:
        for group_id in list(self.groups):
            if group_id in kept_group_ids:
                continue
            group = self.groups.get(group_id)
            if group is None:
                continue
            # Do not put these prompts on hard cooldown: this is scheduler cleanup, not a task failure.
            await self.drop_group(group_id, reason=reason)

    async def _cancel_tasks_for_batch_cleanup(
        self,
        tasks_to_cancel: list[tuple[asyncio.Task, InflightRolloutInfo]],
        *,
        step: int,
        status: str,
        reason: str,
        kept_worker_keys: set[tuple[int, int]],
        requeue_slots_for_kept_groups: set[int] | None = None,
    ) -> None:
        cancelled_tasks: list[asyncio.Task] = []
        restart_infos_by_worker: dict[tuple[int, int], InflightRolloutInfo] = {}
        for task, info in tasks_to_cancel:
            if task not in self.inflight_requests:
                continue
            self.inflight_requests.pop(task, None)
            if hasattr(self, "task_done_perf"):
                self.task_done_perf.pop(task, None)
            await self._release_worker_reservation(info)
            cancelled_tasks.append(task)
            task.cancel()
            self._log_attempt_finished(info, step=step, status=status, error=reason, task=task)
            self.scheduler_cancelled_batch_complete = getattr(self, "scheduler_cancelled_batch_complete", 0) + int(
                status == "cancelled_batch_complete"
            )
            if status == "cancelled_batch_complete":
                worker_name = info.env_worker_name or "unknown"
                if not hasattr(self, "scheduler_cancelled_batch_complete_by_worker"):
                    self.scheduler_cancelled_batch_complete_by_worker = defaultdict(int)
                self.scheduler_cancelled_batch_complete_by_worker[worker_name] += 1
            if requeue_slots_for_kept_groups and info.group_id in requeue_slots_for_kept_groups:
                group = self.groups.get(info.group_id)
                if (
                    group is not None
                    and info.slot_index is not None
                    and info.slot_index not in group.pending_slots
                    and info.slot_index not in group.completed_rollouts
                ):
                    group.pending_slots.appendleft(info.slot_index)
            worker_key = self._worker_generation_key(info)
            if worker_key is not None and worker_key not in kept_worker_keys:
                restart_infos_by_worker.setdefault(worker_key, info)

        if cancelled_tasks:
            await safe_cancel_all(cancelled_tasks)
        for info in restart_infos_by_worker.values():
            await self._restart_worker_for_batch_cancel(info, step=step, reason=reason)

    async def _cancel_stale_carryover(self, *, step: int) -> None:
        if not self.config.async_scheduling.cancel_stale_carryover:
            return
        stale_items = [
            (task, info)
            for task, info in list(self.inflight_requests.items())
            if self._is_stale_carryover(info, step=step)
        ]
        if not stale_items:
            return
        self.scheduler_stale_after_batch_complete = getattr(self, "scheduler_stale_after_batch_complete", 0) + len(
            stale_items
        )
        affected_group_ids = {info.group_id for _, info in stale_items if info.group_id is not None}
        await self._cancel_tasks_for_batch_cleanup(
            stale_items,
            step=step,
            status="stale_after_batch_complete",
            reason="stale_carryover",
            kept_worker_keys=set(),
        )
        for group_id in affected_group_ids:
            if group_id in self.groups:
                await self.drop_group(group_id, reason="stale_carryover")

    async def _trim_carryover_at_batch_completion(self, *, step: int) -> None:
        max_carryover = self.config.async_scheduling.max_cross_step_carryover
        if max_carryover is None:
            return
        if len(self.inflight_requests) <= max_carryover:
            return

        keep_tasks = self._select_carryover_tasks(max_count=max_carryover)
        kept_infos = [info for task, info in self.inflight_requests.items() if task in keep_tasks]
        kept_group_ids = {info.group_id for info in kept_infos if info.group_id is not None}
        kept_worker_keys = {
            key for info in kept_infos if (key := self._worker_generation_key(info)) is not None
        }
        cancel_items = [
            (task, info)
            for task, info in list(self.inflight_requests.items())
            if task not in keep_tasks
        ]
        await self._cancel_tasks_for_batch_cleanup(
            cancel_items,
            step=step,
            status="cancelled_batch_complete",
            reason="batch_complete_excess_carryover",
            kept_worker_keys=kept_worker_keys,
            requeue_slots_for_kept_groups=kept_group_ids,
        )
        await self._drop_groups_without_kept_carryover(
            kept_group_ids=kept_group_ids,
            reason="batch_complete_excess_carryover",
        )

    async def _rollout_deadline_watchdog_loop(self, *, step: int) -> None:
        while True:
            await asyncio.sleep(1.0)
            await self._enforce_rollout_deadlines(step=step)

    async def _reschedule_or_drop_slot(
        self,
        info: InflightRolloutInfo,
        *,
        step: int,
        reason: str,
        count_as_reschedule: bool,
    ) -> None:
        if info.group_id is None or info.slot_index is None:
            return
        group = self.groups.get(info.group_id)
        if group is None or group.dropped:
            return
        max_attempts = self.config.env_worker_recovery.max_rollout_attempts_per_slot
        if info.attempt_number >= max_attempts:
            self.attempt_group_drops += 1
            self.attempt_drops_by_task[info.task] += 1
            await self.drop_group(info.group_id, step=step, reason=f"{reason}:max_attempts_exceeded")
            self.logger.warning(
                f"Dropped group {info.group_id} ({info.task}) after slot {info.slot_index} reached "
                f"{info.attempt_number}/{max_attempts} attempts; example is cooling down for "
                f"{self.config.env_worker_recovery.max_attempts_cooldown_steps} step(s)."
            )
            return
        if info.slot_index not in group.pending_slots and info.slot_index not in group.completed_rollouts:
            group.pending_slots.appendleft(info.slot_index)
        if count_as_reschedule:
            self.attempt_reschedules += 1

    def _log_attempt_finished(
        self,
        info: InflightRolloutInfo,
        *,
        step: int,
        status: str,
        rollout: vf.RolloutOutput | None = None,
        error: str | None = None,
        task: asyncio.Task | None = None,
    ) -> None:
        timings = self._attempt_timings(task, info)
        worker_runtime_s = timings["worker_runtime_s"]
        scheduler_consume_lag_s = timings["scheduler_consume_lag_s"]
        wall_s = timings["wall_s"]
        self.scheduler_attempts_finished = getattr(self, "scheduler_attempts_finished", 0) + 1
        if not hasattr(self, "attempts_by_task"):
            self.attempts_by_task = defaultdict(int)
        if not hasattr(self, "attempt_duration_seconds"):
            self.attempt_duration_seconds = []
        if not hasattr(self, "attempt_wall_seconds"):
            self.attempt_wall_seconds = []
        if not hasattr(self, "attempt_scheduler_consume_lag_seconds"):
            self.attempt_scheduler_consume_lag_seconds = []
        self.attempt_duration_seconds.append(worker_runtime_s)
        self.attempt_wall_seconds.append(wall_s)
        self.attempt_scheduler_consume_lag_seconds.append(scheduler_consume_lag_s)
        if status == "timeout":
            if not hasattr(self, "attempt_timeouts"):
                self.attempt_timeouts = 0
            if not hasattr(self, "attempt_timeouts_by_task"):
                self.attempt_timeouts_by_task = defaultdict(int)
            self.attempt_timeouts += 1
            self.attempt_timeouts_by_task[info.task] += 1
        if status == "late_success_after_timeout":
            if not hasattr(self, "attempt_late_success_after_timeout"):
                self.attempt_late_success_after_timeout = 0
            if not hasattr(self, "attempt_late_success_after_timeout_by_task"):
                self.attempt_late_success_after_timeout_by_task = defaultdict(int)
            self.attempt_late_success_after_timeout += 1
            self.attempt_late_success_after_timeout_by_task[info.task] += 1
        self.attempts_by_task[info.task] += 1
        payload = {
            "attempt_id": info.attempt_id,
            "phase": self.current_phase,
            "step": info.scheduled_step,
            "scheduled_step": info.scheduled_step,
            "finished_step": step,
            "group_id": info.group_id,
            "slot_index": info.slot_index,
            "attempt_number": info.attempt_number,
            "source_id": info.source_id,
            "example_id": info.example_id,
            "dataset_name": info.dataset_name,
            "task": info.task,
            "worker_id": info.worker_id,
            "worker_name": info.env_worker_name,
            "worker_generation": info.worker_generation,
            "env_address": info.env_address,
            "inference_client": self._client_identity(info.client_config),
            "policy_ckpt_step": info.policy_ckpt_step,
            "off_policy_steps_at_start": 0,
            "off_policy_steps_at_end": info.off_policy_steps,
            "start_time_iso": info.start_time_iso,
            "end_time_iso": datetime.now(UTC).isoformat(),
            "duration_ms": worker_runtime_s * 1000.0,
            "worker_runtime_ms": worker_runtime_s * 1000.0,
            "scheduler_consume_lag_ms": scheduler_consume_lag_s * 1000.0,
            "wall_ms": wall_s * 1000.0,
            "timeout_seconds": self.config.rollout_timeout_seconds,
            "status": status,
            "stop_condition": rollout.get("stop_condition") if rollout is not None else None,
            "error": error,
            "reschedule_count": info.reschedule_count,
            "group_completed_count": (
                len(self.groups[info.group_id].completed_rollouts)
                if info.group_id is not None and info.group_id in self.groups
                else None
            ),
        }
        if rollout is not None:
            payload.update(self._rollout_protocol_stats(rollout))
        self._log_attempt_event("attempt_finished", payload)

    def _consume_rollout_buffer_into_batch(
        self,
        *,
        batch_rollouts: list[vf.RolloutOutput],
        batch_progress: int,
        pbar: ProgressTracker,
    ) -> int:
        while batch_progress < self.batch_target and len(self.buffer.rollout_buffer) >= self.rollouts_per_example:
            accepted_rollouts = self.buffer.sample_rollouts(n=self.rollouts_per_example)
            if not accepted_rollouts:
                break
            batch_rollouts.extend(accepted_rollouts)
            progress_increment = self.get_batch_progress_increment(accepted_rollouts)
            batch_progress += progress_increment
            pbar.update(progress_increment)
        return batch_progress

    async def _process_finished_task(
        self,
        finished_task: asyncio.Task,
        *,
        step: int,
        batch_rollouts: list[vf.RolloutOutput],
        batch_progress: int,
        pbar: ProgressTracker,
    ) -> int:
        if finished_task.done() and finished_task not in self.task_done_perf:
            self.task_done_perf[finished_task] = time.perf_counter()

        rollout_info = self.inflight_requests.pop(finished_task, None)
        if rollout_info is None:
            return batch_progress
        await self._release_worker_reservation(rollout_info)
        if self._is_stale_carryover(rollout_info, step=step):
            self.scheduler_stale_after_batch_complete = getattr(
                self, "scheduler_stale_after_batch_complete", 0
            ) + 1
            self._log_attempt_finished(
                rollout_info,
                step=step,
                status="stale_after_batch_complete",
                error="stale_finished_after_batch_complete",
                task=finished_task,
            )
            if rollout_info.group_id is not None:
                await self.drop_group(rollout_info.group_id, reason="stale_finished_after_batch_complete")
            return batch_progress

        group_id = rollout_info.group_id

        try:
            group = self.groups.get(group_id)
            if group is None:
                return batch_progress
            rollout = finished_task.result()

            task = rollout_info.task
            if rollout.get("stop_condition") == "rollout_timeout":
                await self._handle_rollout_timeout(
                    finished_task,
                    rollout_info,
                    step=step,
                    reason="rollout_timeout",
                    rollout=rollout,
                )
                return batch_progress
            if self._worker_runtime_exceeded_timeout(finished_task, rollout_info):
                self._log_attempt_finished(
                    rollout_info,
                    step=step,
                    status="late_success_after_timeout",
                    rollout=rollout,
                    error="worker_runtime_exceeded_timeout",
                    task=finished_task,
                )
                await self._restart_worker_for_timeout(rollout_info, step=step)
                if self._should_drop_group_on_first_timeout():
                    await self._drop_group_on_timeout(
                        rollout_info,
                        step=step,
                        reason="late_success_after_timeout",
                    )
                else:
                    await self._reschedule_or_drop_slot(
                        rollout_info,
                        step=step,
                        reason="late_success_after_timeout",
                        count_as_reschedule=True,
                    )
                return batch_progress
            self.total_rollouts_by_task[task] += 1
            should_reschedule = False
            if len(rollout["trajectory"]) == 0:
                self.empty_rollouts_by_task[task] += 1
                should_reschedule = True
                self.logger.warning(
                    f"Empty trajectory in group {group_id} ({task}), re-scheduling "
                    f"({len(group.completed_rollouts)}/{self.rollouts_per_example} complete)"
                )
            if rollout["error"] is not None:
                self.errored_rollouts_by_task[task] += 1
                should_reschedule = True
                self.logger.warning(
                    f"Rollout error in group {group_id} ({task}), re-scheduling "
                    f"({len(group.completed_rollouts)}/{self.rollouts_per_example} complete): "
                    f"{rollout['error']['error_chain_repr']}"
                )
            if should_reschedule:
                self._log_attempt_finished(
                    rollout_info,
                    step=step,
                    status="rescheduled",
                    rollout=rollout,
                    task=finished_task,
                )
                await self._reschedule_or_drop_slot(
                    rollout_info,
                    step=step,
                    reason="empty_or_error",
                    count_as_reschedule=True,
                )
                return batch_progress

            self._log_attempt_finished(rollout_info, step=step, status="success", rollout=rollout, task=finished_task)
            if rollout_info.scheduled_step < step:
                self.scheduler_carryover_accepted = getattr(self, "scheduler_carryover_accepted", 0) + 1
            assert rollout_info.slot_index is not None
            group.completed_rollouts[rollout_info.slot_index] = rollout
            if len(group.completed_rollouts) < self.rollouts_per_example:
                return batch_progress
            completed_group = self.groups.pop(group_id)
            completed_by_slot = completed_group.completed_rollouts
            completed_rollouts = [completed_by_slot[idx] for idx in sorted(completed_by_slot)]
            if self._should_score_group_in_background(task):
                self._enqueue_group_scoring(
                    group_id=group_id,
                    example=completed_group.example,
                    completed_rollouts=completed_rollouts,
                    scheduled_step=rollout_info.scheduled_step,
                )
                return batch_progress
            completed_rollouts = await self._score_group_if_deferred(completed_rollouts)
        except asyncio.CancelledError:
            if group_id is not None:
                await self.drop_group(group_id)
            return batch_progress
        except Exception as e:
            self.logger.warning(f"Rollout failed: {e}")
            self._log_attempt_finished(rollout_info, step=step, status="error", error=repr(e), task=finished_task)
            if group_id is not None:
                await self.drop_group(group_id)
            return batch_progress

        self.buffer.update(completed_rollouts, step=step)
        if batch_progress < self.batch_target:
            batch_progress = self._consume_rollout_buffer_into_batch(
                batch_rollouts=batch_rollouts,
                batch_progress=batch_progress,
                pbar=pbar,
            )
        return batch_progress

    async def _drain_done_tasks(
        self,
        *,
        step: int,
        batch_rollouts: list[vf.RolloutOutput],
        batch_progress: int,
        pbar: ProgressTracker,
    ) -> int:
        while True:
            done_tasks = [task for task in list(self.inflight_requests) if task.done()]
            if not done_tasks:
                return batch_progress
            for task in done_tasks:
                batch_progress = await self._process_finished_task(
                    task,
                    step=step,
                    batch_rollouts=batch_rollouts,
                    batch_progress=batch_progress,
                    pbar=pbar,
                )

    async def generate_batch(self, step: int) -> list[vf.RolloutOutput]:
        """Continuously generates a batch of rollouts."""
        self.step = step
        self.buffer.current_step = step

        # Cancel the previous update policy task to avoid concurrent updates
        if self.update_policy_task is not None:
            await safe_cancel(self.update_policy_task)

        # Check the async barrier before starting, then re-create the update policy loop.
        # This ensures we respect max_async_level while still listening for policy updates mid-step.
        await self.maybe_update_policy()
        self.update_policy_task = asyncio.create_task(self.update_policy_loop())
        deadline_watchdog_task = asyncio.create_task(self._rollout_deadline_watchdog_loop(step=step))

        batch_start_time = time.perf_counter()

        try:
            self.logger.debug("Starting to generate batch rollouts")
            self.buffer.release_due_hard_examples(step)
            self.scheduler_carryover_count += sum(
                1 for info in self.inflight_requests.values() if info.scheduled_step < step
            )
            await self._cancel_stale_carryover(step=step)

            batch_rollouts: list[vf.RolloutOutput] = []
            batch_progress = 0
            pbar = ProgressTracker(
                total=self.batch_target, desc="Generating rollouts (train)", json_logging=self.json_logging, step=step
            )

            while batch_progress < self.batch_target:
                batch_progress = await self._drain_done_group_scoring_tasks(
                    step=step,
                    batch_rollouts=batch_rollouts,
                    batch_progress=batch_progress,
                    pbar=pbar,
                )
                batch_progress = self._consume_rollout_buffer_into_batch(
                    batch_rollouts=batch_rollouts,
                    batch_progress=batch_progress,
                    pbar=pbar,
                )
                if batch_progress >= self.batch_target:
                    break
                allowed_inflight = self._allowed_inflight_for_progress(batch_progress)
                await self._fill_inflight_requests(allowed_inflight=allowed_inflight)
                inflight_tasks = list(self.inflight_requests.keys())
                scoring_tasks = list(getattr(self, "group_scoring_tasks", {}).keys())
                wait_tasks = inflight_tasks + scoring_tasks
                if not wait_tasks:
                    await asyncio.sleep(1.0)
                    continue

                finished_tasks, _ = await asyncio.wait(
                    wait_tasks,
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                await self.checkpoint_ready.wait()
                await self._enforce_rollout_deadlines(step=step)

                for finished_task in finished_tasks:
                    if finished_task in self.inflight_requests:
                        batch_progress = await self._process_finished_task(
                            finished_task,
                            step=step,
                            batch_rollouts=batch_rollouts,
                            batch_progress=batch_progress,
                            pbar=pbar,
                        )
                    elif finished_task in getattr(self, "group_scoring_tasks", {}):
                        batch_progress = await self._process_finished_scoring_task(
                            finished_task,
                            step=step,
                            batch_rollouts=batch_rollouts,
                            batch_progress=batch_progress,
                            pbar=pbar,
                        )

            if self.config.async_scheduling.prefetch_next_batch:
                await self._fill_inflight_requests()
            batch_progress = await self._drain_done_tasks(
                step=step,
                batch_rollouts=batch_rollouts,
                batch_progress=batch_progress,
                pbar=pbar,
            )
            batch_progress = await self._drain_done_group_scoring_tasks(
                step=step,
                batch_rollouts=batch_rollouts,
                batch_progress=batch_progress,
                pbar=pbar,
            )
            await self._trim_carryover_at_batch_completion(step=step)

            batch_rollouts = self.finalize_batch_rollouts(batch_rollouts)
            pbar.close()
            self.last_batch_generation_time = time.perf_counter() - batch_start_time
            return batch_rollouts
        finally:
            await safe_cancel(deadline_watchdog_task)

    async def stop(self) -> None:
        await self.cancel_inflight_rollouts()
        if self.update_policy_task is not None:
            await safe_cancel(self.update_policy_task)
            self.update_policy_task = None

    @property
    def max_off_policy_level(self) -> int:
        steps = [info.off_policy_steps for info in self.inflight_requests.values()]
        if not steps:
            return 0
        return max(steps)

    @property
    def min_off_policy_level(self) -> int:
        steps = [info.off_policy_steps for info in self.inflight_requests.values()]
        if not steps:
            return 0
        return min(steps)

    @property
    def mean_off_policy_level(self) -> float:
        steps = [info.off_policy_steps for info in self.inflight_requests.values()]
        if not steps:
            return 0
        return sum(steps) / len(steps)

    @property
    def async_level(self) -> int:
        return self.step - self.ckpt_step

    def get_metrics(self) -> dict[str, float]:
        total_rollouts = sum(self.total_rollouts_by_task.values())
        total_attempts = max(sum(self.attempts_by_task.values()), 1)
        batch_target = max(self.batch_target, 1)
        worker_loads = [
            load
            for pool in self._env_worker_pools()
            for load in pool.worker_loads().values()
        ]
        max_requests_per_worker = self.config.async_scheduling.max_requests_per_env_worker
        at_capacity_count = sum(
            pool.at_capacity_count(max_requests_per_worker)
            for pool in self._env_worker_pools()
        )
        metrics = {
            "time/wait_for_ckpt": self.wait_for_ckpt_time,
            "time/update_weights": self.update_weights_time,
            "scheduler/async_level": self.async_level,
            "scheduler/inflight_rollouts": self.inflight_rollout_count,
            "scheduler/inflight_samples": self.inflight_sample_count,
            "scheduler/cancelled_rollouts": self.cancelled_rollouts_count,
            "scheduler/attempts_started": self.scheduler_attempts_started,
            "scheduler/attempts_finished": self.scheduler_attempts_finished,
            "scheduler/replacement_factor": self.scheduler_attempts_started / batch_target,
            "scheduler/carryover_count": self.scheduler_carryover_count,
            "scheduler/carryover_accepted": self.scheduler_carryover_accepted,
            "scheduler/stale_after_batch_complete": self.scheduler_stale_after_batch_complete,
            "scheduler/cancelled_batch_complete": self.scheduler_cancelled_batch_complete,
            "scheduler/allowed_inflight": self.scheduler_allowed_inflight,
            "empty_rollouts/all": sum(self.empty_rollouts_by_task.values()) / max(total_rollouts, 1),
            "errored_rollouts/all": sum(self.errored_rollouts_by_task.values()) / max(total_rollouts, 1),
            "timeout_rollouts/all": sum(self.timeout_rollouts_by_task.values()) / max(total_rollouts, 1),
            "off_policy_level/all/max": self.max_off_policy_level,
            "off_policy_level/all/mean": self.mean_off_policy_level,
            "off_policy_level/all/min": self.min_off_policy_level,
            "attempt/duration_s/mean": (
                sum(self.attempt_duration_seconds) / len(self.attempt_duration_seconds)
                if self.attempt_duration_seconds
                else 0.0
            ),
            "attempt/duration_s/p95": self._percentile(self.attempt_duration_seconds, 95),
            "attempt/duration_s/p99": self._percentile(self.attempt_duration_seconds, 99),
            "attempt/duration_s/p998": self._percentile(self.attempt_duration_seconds, 99.8),
            "attempt/wall_s/mean": (
                sum(self.attempt_wall_seconds) / len(self.attempt_wall_seconds)
                if self.attempt_wall_seconds
                else 0.0
            ),
            "attempt/wall_s/p95": self._percentile(self.attempt_wall_seconds, 95),
            "attempt/wall_s/p99": self._percentile(self.attempt_wall_seconds, 99),
            "attempt/wall_s/p998": self._percentile(self.attempt_wall_seconds, 99.8),
            "attempt/scheduler_consume_lag_s/mean": (
                sum(self.attempt_scheduler_consume_lag_seconds) / len(self.attempt_scheduler_consume_lag_seconds)
                if self.attempt_scheduler_consume_lag_seconds
                else 0.0
            ),
            "attempt/scheduler_consume_lag_s/p95": self._percentile(self.attempt_scheduler_consume_lag_seconds, 95),
            "attempt/scheduler_consume_lag_s/p99": self._percentile(self.attempt_scheduler_consume_lag_seconds, 99),
            "attempt/scheduler_consume_lag_s/p998": self._percentile(
                self.attempt_scheduler_consume_lag_seconds,
                99.8,
            ),
            "attempt/timeout_rate": self.attempt_timeouts / total_attempts,
            "attempt/late_success_after_timeout_rate": self.attempt_late_success_after_timeout / total_attempts,
            "attempt/reschedule_rate": self.attempt_reschedules / total_attempts,
            "attempt/group_drop_rate": self.attempt_group_drops / total_attempts,
            "attempt/group_drop_first_timeout_rate": self.attempt_group_drops_first_timeout / total_attempts,
            "buffer/timeout_cooldown_groups": self.timeout_cooldown_groups,
            "worker/restart_count": self.worker_restart_count,
            "worker/max_effective_load": max(worker_loads, default=0),
            "worker/at_capacity_count": at_capacity_count,
            "group_scoring/active_tasks": len(getattr(self, "group_scoring_tasks", {})),
            "group_scoring/pending_tasks": len(getattr(self, "group_scoring_tasks", {})),
            "group_scoring/started_groups": getattr(self, "group_scoring_started", 0),
            "group_scoring/finished_groups": getattr(self, "group_scoring_finished", 0),
            "group_scoring/failed_groups": getattr(self, "group_scoring_failed", 0),
            "group_scoring/stale_groups": getattr(self, "group_scoring_stale", 0),
            "group_scoring/max_pending_reached": getattr(self, "group_scoring_max_pending_reached", 0),
            "group_scoring/runtime_s/mean": (
                sum(self.group_scoring_runtime_seconds) / len(self.group_scoring_runtime_seconds)
                if getattr(self, "group_scoring_runtime_seconds", [])
                else 0.0
            ),
            "group_scoring/runtime_s/p95": self._percentile(
                getattr(self, "group_scoring_runtime_seconds", []),
                95,
            ),
            "group_scoring/runtime_s/p99": self._percentile(
                getattr(self, "group_scoring_runtime_seconds", []),
                99,
            ),
            "group_scoring/queue_wait_s/mean": (
                sum(self.group_scoring_queue_wait_seconds) / len(self.group_scoring_queue_wait_seconds)
                if getattr(self, "group_scoring_queue_wait_seconds", [])
                else 0.0
            ),
            "group_scoring/queue_wait_s/p95": self._percentile(
                getattr(self, "group_scoring_queue_wait_seconds", []),
                95,
            ),
            "group_scoring/queue_wait_s/p99": self._percentile(
                getattr(self, "group_scoring_queue_wait_seconds", []),
                99,
            ),
            "group_scoring/lag_s/mean": (
                sum(self.group_scoring_lag_seconds) / len(self.group_scoring_lag_seconds)
                if getattr(self, "group_scoring_lag_seconds", [])
                else 0.0
            ),
            "group_scoring/lag_s/p95": self._percentile(
                getattr(self, "group_scoring_lag_seconds", []),
                95,
            ),
            "group_scoring/lag_s/p99": self._percentile(
                getattr(self, "group_scoring_lag_seconds", []),
                99,
            ),
        }
        for task, count in self.empty_rollouts_by_task.items():
            task_total = max(self.total_rollouts_by_task[task], 1)
            metrics[f"empty_rollouts/{task}"] = count / task_total
        for task, count in self.errored_rollouts_by_task.items():
            task_total = max(self.total_rollouts_by_task[task], 1)
            metrics[f"errored_rollouts/{task}"] = count / task_total
        for task, count in self.timeout_rollouts_by_task.items():
            task_total = max(self.total_rollouts_by_task[task], 1)
            metrics[f"timeout_rollouts/{task}"] = count / task_total
        for task, count in self.attempt_timeouts_by_task.items():
            task_total = max(self.attempts_by_task[task], 1)
            metrics[f"attempt/timeout_rate/{task}"] = count / task_total
        for task, count in self.attempt_late_success_after_timeout_by_task.items():
            task_total = max(self.attempts_by_task[task], 1)
            metrics[f"attempt/late_success_after_timeout_rate/{task}"] = count / task_total
        for task, count in self.attempt_drops_by_task.items():
            task_total = max(self.attempts_by_task[task], 1)
            metrics[f"attempt/drop_rate/{task}"] = count / task_total
        for task, count in self.attempt_first_timeout_drops_by_task.items():
            task_total = max(self.attempts_by_task[task], 1)
            metrics[f"attempt/first_timeout_drop_rate/{task}"] = count / task_total
        for worker_name, count in self.worker_restart_count_by_name.items():
            metrics[f"worker/restart_count/{worker_name}"] = count
        for worker_name, count in self.scheduler_cancelled_batch_complete_by_worker.items():
            metrics[f"scheduler/cancelled_batch_complete_by_worker/{worker_name}"] = count
        by_task: dict[str, list[int]] = {}
        for info in self.inflight_requests.values():
            by_task.setdefault(info.task, []).append(info.off_policy_steps)
        for task, steps in by_task.items():
            metrics[f"off_policy_level/{task}/max"] = max(steps)
            metrics[f"off_policy_level/{task}/mean"] = sum(steps) / len(steps)
            metrics[f"off_policy_level/{task}/min"] = min(steps)
        self.cancelled_rollouts_count = 0
        self.empty_rollouts_by_task.clear()
        self.errored_rollouts_by_task.clear()
        self.timeout_rollouts_by_task.clear()
        self.total_rollouts_by_task.clear()
        self.attempt_duration_seconds.clear()
        self.attempt_wall_seconds.clear()
        self.attempt_scheduler_consume_lag_seconds.clear()
        self.attempt_timeouts = 0
        self.attempt_late_success_after_timeout = 0
        self.attempt_reschedules = 0
        self.attempt_group_drops = 0
        self.attempt_group_drops_first_timeout = 0
        self.timeout_cooldown_groups = 0
        self.attempts_by_task.clear()
        self.attempt_timeouts_by_task.clear()
        self.attempt_late_success_after_timeout_by_task.clear()
        self.attempt_drops_by_task.clear()
        self.attempt_first_timeout_drops_by_task.clear()
        self.worker_restart_count = 0
        self.worker_restart_count_by_name.clear()
        self.scheduler_attempts_started = 0
        self.scheduler_attempts_finished = 0
        self.scheduler_carryover_count = 0
        self.scheduler_carryover_accepted = 0
        self.scheduler_stale_after_batch_complete = 0
        self.scheduler_cancelled_batch_complete = 0
        self.scheduler_cancelled_batch_complete_by_worker.clear()
        self.group_scoring_started = 0
        self.group_scoring_finished = 0
        self.group_scoring_failed = 0
        self.group_scoring_stale = 0
        self.group_scoring_max_pending_reached = 0
        self.group_scoring_runtime_seconds.clear()
        self.group_scoring_queue_wait_seconds.clear()
        self.group_scoring_lag_seconds.clear()

        # Add inference pool metrics (e.g. elastic pool server counts)
        metrics.update(self.inference_pool.get_metrics())

        return metrics
