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


@dataclass
class GroupState:
    """Tracks the state of a rollout group (one example × N rollouts)."""

    example: dict
    pending_slots: deque[int]
    completed_rollouts: dict[int, vf.RolloutOutput] = field(default_factory=dict)
    attempts_by_slot: dict[int, int] = field(default_factory=dict)
    dropped: bool = False
    pinned_client: vf.ClientConfig | None = None


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

        # Track in-progress groups while rollouts are generated independently.
        self.next_group_id = 0
        self.groups: dict[int, GroupState] = {}

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
        self.attempt_timeouts = 0
        self.attempt_reschedules = 0
        self.attempt_group_drops = 0
        self.attempts_by_task: dict[str, int] = defaultdict(int)
        self.attempt_timeouts_by_task: dict[str, int] = defaultdict(int)
        self.attempt_drops_by_task: dict[str, int] = defaultdict(int)
        self.worker_restart_count = 0
        self.worker_restart_count_by_name: dict[str, int] = defaultdict(int)

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
        self.groups.clear()
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

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))))
        return ordered[idx]

    async def drop_group(self, group_id: int, *, step: int | None = None, reason: str | None = None) -> int:
        """Drop a group and cancel any remaining in-flight rollouts for it."""
        tasks_to_cancel = []
        group = self.groups.pop(group_id, None)
        if group is not None:
            group.dropped = True
            if step is not None and self.config.env_worker_recovery.enabled:
                self.buffer.put_example_on_hard_cooldown(
                    group.example,
                    step=step,
                    cooldown_steps=self.config.env_worker_recovery.max_attempts_cooldown_steps,
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
                    },
                )
        for task, info in list(self.inflight_requests.items()):
            if info.group_id != group_id:
                continue
            self.inflight_requests.pop(task, None)
            await self._release_worker_reservation(info)
            tasks_to_cancel.append(task)
        await safe_cancel_all(tasks_to_cancel)
        return len(tasks_to_cancel)

    async def schedule_rollout(self, group_id: int):
        """Asynchronously schedules a rollout request."""
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        group = self.groups.get(group_id)
        if group is None or group.dropped or not group.pending_slots:
            return
        slot_index = group.pending_slots.popleft()
        attempt_number = group.attempts_by_slot.get(slot_index, 0) + 1
        group.attempts_by_slot[slot_index] = attempt_number
        if group.pinned_client is not None:
            client_config = group.pinned_client
        else:
            client_config = await self._select_least_loaded_client()
            if group_id not in self.groups:
                return
            group.pinned_client = client_config
        worker_pool = self._get_env_worker_pool(group.example["task"])
        worker_reservation = None
        worker_token = None
        if self.config.env_worker_recovery.enabled and worker_pool is not None:
            worker_reservation = await worker_pool.reserve_worker()
            worker_token = worker_pool.activate_reservation(worker_reservation)
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
        )
        self._log_attempt_event(
            "attempt_started",
            {
                "attempt_id": attempt_id,
                "phase": self.current_phase,
                "step": self.step,
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
        run_rollout_task = asyncio.create_task(
            run_rollout(
                env=self.env,
                client=client_config,
                example=group.example,
                model_name=self.model_name,
                sampling_args=self.sampling_args,
                max_retries=self.max_retries_by_task.get(group.example["task"], 0),
                rollout_timeout_seconds=self.config.rollout_timeout_seconds,
            )
        )
        if worker_pool is not None and worker_token is not None:
            worker_pool.reset_active_reservation(worker_token)
        self.inflight_requests[run_rollout_task] = info

    @property
    def inflight_rollout_count(self) -> int:
        return len(self.inflight_requests)

    @property
    def inflight_sample_count(self) -> int:
        return self.inflight_rollout_count + sum(len(g.pending_slots) for g in self.groups.values())

    async def _schedule_next_request(self) -> bool:
        remaining_capacity = self.max_inflight_rollouts - self.inflight_rollout_count

        if remaining_capacity <= 0:
            return False

        for group_id, group in self.groups.items():
            if group.pending_slots and not group.dropped:
                await self.schedule_rollout(group_id=group_id)
                return True

        example = self.buffer.sample_examples(n=1)[0]
        group_id = self.next_group_id
        self.next_group_id += 1
        self.groups[group_id] = GroupState(example=example, pending_slots=deque(range(self.rollouts_per_example)))
        await self.schedule_rollout(group_id=group_id)
        return True

    async def _fill_inflight_requests(self) -> None:
        while await self._schedule_next_request():
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

    async def _score_group_if_deferred(self, completed_rollouts: list[vf.RolloutOutput]) -> list[vf.RolloutOutput]:
        if not completed_rollouts:
            return completed_rollouts
        task = completed_rollouts[0]["task"]
        if not self._should_defer_group_scoring(task):
            return completed_rollouts
        env_for_task = self.env.get_env_for_task(task)
        await env_for_task.rubric.score_group(cast(list[vf.State], completed_rollouts))
        return completed_rollouts

    async def _cancel_inflight_for_worker_generation(
        self,
        *,
        worker_id: int,
        worker_generation: int,
        step: int,
        reason: str,
    ) -> int:
        tasks_to_cancel = []
        affected_infos: list[InflightRolloutInfo] = []
        for task, info in list(self.inflight_requests.items()):
            if info.worker_id != worker_id or info.worker_generation != worker_generation:
                continue
            self.inflight_requests.pop(task, None)
            tasks_to_cancel.append(task)
            affected_infos.append(info)
            await self._release_worker_reservation(info)

        for info in affected_infos:
            self._log_attempt_finished(info, step=step, status="cancelled_for_worker_restart")
            await self._reschedule_or_drop_slot(info, step=step, reason=reason, count_as_reschedule=True)

        await safe_cancel_all(tasks_to_cancel)
        return len(tasks_to_cancel)

    async def _restart_worker_for_timeout(self, info: InflightRolloutInfo, step: int) -> None:
        if (
            not self.config.env_worker_recovery.enabled
            or not self.config.env_worker_recovery.restart_on_rollout_timeout
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
                "reason": "rollout_timeout",
            },
        )
        await self._cancel_inflight_for_worker_generation(
            worker_id=info.worker_id,
            worker_generation=info.worker_generation,
            step=step,
            reason="worker_restart",
        )
        try:
            handle = await worker_pool.restart_worker(
                worker_id=info.worker_id,
                expected_generation=info.worker_generation,
                reason="rollout_timeout",
                cancel_grace_seconds=self.config.env_worker_recovery.cancel_grace_seconds,
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
                    "reason": "rollout_timeout",
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
                    "reason": "rollout_timeout",
                },
            )
            self.logger.warning(f"Failed to restart env worker {info.env_worker_name}: {exc!r}")

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
    ) -> None:
        end_time_perf = time.perf_counter()
        duration_s = max(0.0, end_time_perf - info.start_time_perf)
        if status == "success":
            self.attempt_duration_seconds.append(duration_s)
        if status == "timeout":
            self.attempt_timeouts += 1
            self.attempt_timeouts_by_task[info.task] += 1
        self.attempts_by_task[info.task] += 1
        payload = {
            "attempt_id": info.attempt_id,
            "phase": self.current_phase,
            "step": step,
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
            "duration_ms": duration_s * 1000.0,
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

    async def generate_batch(self, step: int) -> list[vf.RolloutOutput]:
        """Continuously generates a batch of rollouts."""
        self.step = step

        # Cancel the previous update policy task to avoid concurrent updates
        if self.update_policy_task is not None:
            await safe_cancel(self.update_policy_task)

        # Check the async barrier before starting, then re-create the update policy loop.
        # This ensures we respect max_async_level while still listening for policy updates mid-step.
        await self.maybe_update_policy()
        self.update_policy_task = asyncio.create_task(self.update_policy_loop())

        batch_start_time = time.perf_counter()

        self.logger.debug("Starting to generate batch rollouts")
        self.buffer.release_due_hard_examples(step)

        batch_rollouts: list[vf.RolloutOutput] = []
        batch_progress = 0
        pbar = ProgressTracker(
            total=self.batch_target, desc="Generating rollouts (train)", json_logging=self.json_logging, step=step
        )

        while batch_progress < self.batch_target:
            await self._fill_inflight_requests()
            inflight_tasks = list(self.inflight_requests.keys())

            finished_tasks, _ = await asyncio.wait(
                inflight_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            await self.checkpoint_ready.wait()

            for finished_task in finished_tasks:
                if batch_progress >= self.batch_target:
                    break

                rollout_info = self.inflight_requests.pop(finished_task, None)
                if rollout_info is None:
                    continue
                await self._release_worker_reservation(rollout_info)

                group_id = rollout_info.group_id

                try:
                    group = self.groups.get(group_id)
                    if group is None:
                        continue
                    rollout = finished_task.result()

                    task = rollout_info.task
                    self.total_rollouts_by_task[task] += 1
                    should_reschedule = False
                    if rollout.get("stop_condition") == "rollout_timeout":
                        self.timeout_rollouts_by_task[task] += 1
                        self.empty_rollouts_by_task[task] += int(len(rollout["trajectory"]) == 0)
                        self.errored_rollouts_by_task[task] += int(rollout["error"] is not None)
                        self._log_attempt_finished(rollout_info, step=step, status="timeout", rollout=rollout)
                        await self._restart_worker_for_timeout(rollout_info, step=step)
                        await self._reschedule_or_drop_slot(
                            rollout_info,
                            step=step,
                            reason="rollout_timeout",
                            count_as_reschedule=True,
                        )
                        self.logger.warning(
                            f"Rollout timeout in group {group_id} ({task}) after "
                            f"{self.config.rollout_timeout_seconds}s, re-scheduling "
                            f"({len(group.completed_rollouts)}/{self.rollouts_per_example} complete, "
                            f"attempt {rollout_info.attempt_number}/"
                            f"{self.config.env_worker_recovery.max_rollout_attempts_per_slot})"
                        )
                        continue
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
                        self._log_attempt_finished(rollout_info, step=step, status="rescheduled", rollout=rollout)
                        await self._reschedule_or_drop_slot(
                            rollout_info,
                            step=step,
                            reason="empty_or_error",
                            count_as_reschedule=True,
                        )
                        continue

                    self._log_attempt_finished(rollout_info, step=step, status="success", rollout=rollout)
                    assert rollout_info.slot_index is not None
                    group.completed_rollouts[rollout_info.slot_index] = rollout
                    if len(group.completed_rollouts) < self.rollouts_per_example:
                        continue
                    completed_by_slot = self.groups.pop(group_id).completed_rollouts
                    completed_rollouts = [completed_by_slot[idx] for idx in sorted(completed_by_slot)]
                    completed_rollouts = await self._score_group_if_deferred(completed_rollouts)
                except asyncio.CancelledError:
                    if group_id is not None:
                        await self.drop_group(group_id)
                    continue
                except Exception as e:
                    self.logger.warning(f"Rollout failed: {e}")
                    self._log_attempt_finished(rollout_info, step=step, status="error", error=repr(e))
                    if group_id is not None:
                        await self.drop_group(group_id)
                    continue

                self.buffer.update(completed_rollouts, step=step)
                accepted_rollouts = self.buffer.sample_rollouts(n=self.rollouts_per_example)

                batch_rollouts.extend(accepted_rollouts)
                progress_increment = self.get_batch_progress_increment(accepted_rollouts)
                batch_progress += progress_increment
                pbar.update(progress_increment)

        await self._fill_inflight_requests()

        batch_rollouts = self.finalize_batch_rollouts(batch_rollouts)
        pbar.close()
        self.last_batch_generation_time = time.perf_counter() - batch_start_time
        return batch_rollouts

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
        metrics = {
            "time/wait_for_ckpt": self.wait_for_ckpt_time,
            "time/update_weights": self.update_weights_time,
            "scheduler/async_level": self.async_level,
            "scheduler/inflight_rollouts": self.inflight_rollout_count,
            "scheduler/inflight_samples": self.inflight_sample_count,
            "scheduler/cancelled_rollouts": self.cancelled_rollouts_count,
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
            "attempt/timeout_rate": self.attempt_timeouts / total_attempts,
            "attempt/reschedule_rate": self.attempt_reschedules / total_attempts,
            "attempt/group_drop_rate": self.attempt_group_drops / total_attempts,
            "worker/restart_count": self.worker_restart_count,
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
        for task, count in self.attempt_drops_by_task.items():
            task_total = max(self.attempts_by_task[task], 1)
            metrics[f"attempt/drop_rate/{task}"] = count / task_total
        for worker_name, count in self.worker_restart_count_by_name.items():
            metrics[f"worker/restart_count/{worker_name}"] = count
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
        self.attempt_timeouts = 0
        self.attempt_reschedules = 0
        self.attempt_group_drops = 0
        self.attempts_by_task.clear()
        self.attempt_timeouts_by_task.clear()
        self.attempt_drops_by_task.clear()
        self.worker_restart_count = 0
        self.worker_restart_count_by_name.clear()

        # Add inference pool metrics (e.g. elastic pool server counts)
        metrics.update(self.inference_pool.get_metrics())

        return metrics
