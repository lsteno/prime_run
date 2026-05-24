import asyncio
import contextvars
import logging
import multiprocessing as mp
import random
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any

import verifiers as vf
from verifiers.envs.environment import EnvClient
from verifiers.workers.types import (
    HealthRequest,
    HealthResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)
from verifiers.workers import ZMQEnvClient, ZMQEnvServer

from prime_rl.utils.logger import InterceptHandler, ProgressTracker

DEFAULT_RETRIES = 0
REQUIRED_STATE_COLUMNS = [
    "trajectory",
    "sampling_args",
    "rlm_segments",
    "rlm_trace",
    "final_answer",
    "used_forced_finalize_prompt",
    "hit_max_turn_without_final",
    "missing_final",
    "finalized_before_forced_prompt",
    "finalized_on_forced_prompt",
    "used_repl",
    "used_recursion",
    "used_llm_subcalls",
    "used_rlm_subcalls",
    "num_subcalls",
    "num_llm_subcalls",
    "num_rlm_subcalls",
    "max_depth_reached",
    "subcall_budget_enabled",
    "subcall_budget_total",
    "subcall_budget_remaining",
    "subcall_budget_exhausted",
]
DEFAULT_STATE_COLUMNS = []
_RESERVED_ENV_SERVER_PORTS: set[int] = set()


def get_stable_free_port_pair() -> int:
    """Find a free TCP port pair outside Linux's default ephemeral range."""
    # Linux defaults to 32768..60999 for ephemeral client ports. Env servers
    # also spawn a health responder on port+1, so choosing from 61000..65535
    # avoids races where outbound HTTP connections grab the checked successor
    # port before the health responder binds it.
    candidates = list(range(61000, 65534, 2))
    random.shuffle(candidates)
    for port in candidates:
        if port in _RESERVED_ENV_SERVER_PORTS or port + 1 in _RESERVED_ENV_SERVER_PORTS:
            continue
        with (
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1,
            socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2,
        ):
            try:
                s1.bind(("127.0.0.1", port))
                s2.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
        _RESERVED_ENV_SERVER_PORTS.update({port, port + 1})
        return port
    raise RuntimeError("Could not find a stable free port pair in 61000..65535")


def spawn_env_server(
    env_id: str,
    env_args: dict[str, Any],
    extra_env_kwargs: dict[str, Any],
    address: str | None = None,
    # logging configs
    log_level: str | None = None,
    log_file: str | None = None,
    log_file_level: str | None = None,
    json_logging: bool = False,
) -> tuple[str, mp.Process]:
    """
    Starts a ZMQEnvServer process in a subprocess.

    Mirrors vf.Environment.start_server().
    """
    address = address or f"tcp://127.0.0.1:{get_stable_free_port_pair()}"
    # Use spawn to avoid inheriting file descriptors (e.g. sockets) from
    # the parent process, which has caused hangs when multiple env server
    # subprocesses share the same fds.
    process = mp.get_context("spawn").Process(
        target=ZMQEnvServer.run_server,
        args=(
            env_id,
            env_args,
            extra_env_kwargs,
            log_level,
            log_file,
            log_file_level,
        ),
        kwargs=dict(address=address, json_logging=json_logging),
        daemon=False,  # cannot run daemon because env server uses subprocesses
    )
    process.start()

    return address, process


def setup_env_client(
    address: str,
    name: str | None = None,
    # health check configs
    health_check_interval: float = 5.0,  # 5s (we detect an env server as unhealth after 3 * 5s = 15s of unsuccessful health checks)
    startup_timeout: float = 600.0,  # 10m
    recovery_timeout: float = 600.0,  # 10m
) -> EnvClient:
    """Sets up a ZMQEnvClient for a given address."""
    return ZMQEnvClient(
        address=address,
        name=name,
        health_check_interval=health_check_interval,
        startup_timeout=startup_timeout,
        recovery_timeout=recovery_timeout,
    )


class EnvClientPool(EnvClient):
    """Routes requests for one logical environment across several env servers."""

    def __init__(self, clients: list[EnvClient], name: str | None = None):
        if not clients:
            raise ValueError("EnvClientPool requires at least one client")
        self.clients = list(clients)
        self._lock = asyncio.Lock()
        self._inflight = [0 for _ in self.clients]
        self._next_index = 0
        address = ",".join(client.address for client in self.clients)
        super().__init__(address=f"pool://{address}", name=name)

    async def _reserve(self, weight: int = 1) -> tuple[int, EnvClient]:
        weight = max(1, weight)
        async with self._lock:
            loads = [self._inflight[i] + self._client_pending_count(client) for i, client in enumerate(self.clients)]
            min_load = min(loads)
            candidates = [i for i, load in enumerate(loads) if load == min_load]
            # Round-robin among equally loaded workers so a burst of new requests
            # does not collapse onto worker 0 before child pending counts update.
            chosen = candidates[self._next_index % len(candidates)]
            self._next_index += 1
            self._inflight[chosen] += weight
            return chosen, self.clients[chosen]

    async def _release(self, index: int, weight: int = 1) -> None:
        weight = max(1, weight)
        async with self._lock:
            self._inflight[index] = max(0, self._inflight[index] - weight)

    @staticmethod
    def _client_pending_count(client: EnvClient) -> int:
        pending_requests = getattr(client, "pending_requests", None)
        if pending_requests is None:
            return 0
        try:
            return len(pending_requests)
        except TypeError:
            return 0

    async def wait_for_server_startup(self, timeout: float | None = None) -> None:
        await asyncio.gather(*(client.wait_for_server_startup(timeout=timeout) for client in self.clients))

    async def handle_health_request(self, request: HealthRequest, timeout: float | None) -> HealthResponse:
        responses = await asyncio.gather(
            *(client.handle_health_request(request, timeout=timeout) for client in self.clients),
            return_exceptions=True,
        )
        failures = [response for response in responses if isinstance(response, BaseException)]
        unsuccessful = [
            response
            for response in responses
            if not isinstance(response, BaseException) and not response.success
        ]
        if failures or unsuccessful:
            errors = [repr(error) for error in failures]
            errors.extend(response.error or "unhealthy" for response in unsuccessful)
            return HealthResponse(success=False, error="; ".join(errors))
        return HealthResponse(success=True)

    async def handle_run_rollout_request(
        self, request: RunRolloutRequest, timeout: float | None
    ) -> RunRolloutResponse:
        index, client = await self._reserve()
        try:
            return await client.handle_run_rollout_request(request, timeout=timeout)
        finally:
            await self._release(index)

    async def handle_run_group_request(self, request: RunGroupRequest, timeout: float | None) -> RunGroupResponse:
        weight = len(request.group_inputs)
        index, client = await self._reserve(weight)
        try:
            return await client.handle_run_group_request(request, timeout=timeout)
        finally:
            await self._release(index, weight)

    async def run_rollout(
        self,
        input: vf.RolloutInput,
        client_config: vf.ClientConfig,
        model: str,
        sampling_args: vf.SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
    ) -> vf.RolloutOutput:
        index, client = await self._reserve()
        try:
            return await client.run_rollout(
                input=input,
                client_config=client_config,
                model=model,
                sampling_args=sampling_args,
                max_retries=max_retries,
                state_columns=state_columns,
            )
        finally:
            await self._release(index)

    async def run_group(
        self,
        group_inputs: list[vf.RolloutInput],
        client_config: vf.ClientConfig,
        model: str,
        sampling_args: vf.SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
    ) -> list[vf.RolloutOutput]:
        weight = len(group_inputs)
        index, client = await self._reserve(weight)
        try:
            return await client.run_group(
                group_inputs=group_inputs,
                client_config=client_config,
                model=model,
                sampling_args=sampling_args,
                max_retries=max_retries,
                state_columns=state_columns,
            )
        finally:
            await self._release(index, weight)

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self.clients), return_exceptions=True)


@dataclass
class EnvWorkerSpec:
    """Inputs needed to respawn one logical env worker."""

    env_id: str
    env_args: dict[str, Any]
    extra_env_kwargs: dict[str, Any]
    log_level: str | None
    log_file: str | None
    log_file_level: str | None
    json_logging: bool


@dataclass
class EnvWorkerHandle:
    """Mutable state for one managed environment worker."""

    worker_id: int
    worker_name: str
    address: str
    process: mp.Process | None
    client: EnvClient
    spec: EnvWorkerSpec | None = None
    generation: int = 0
    restart_count: int = 0
    quarantined: bool = False
    inflight_count: int = 0
    restart_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class EnvWorkerReservation:
    """A scheduler-visible reservation for one managed env worker."""

    def __init__(self, pool: "ManagedEnvClientPool", handle: EnvWorkerHandle, weight: int = 1):
        self.pool = pool
        self.handle = handle
        self.weight = max(1, weight)
        self.released = False

    @property
    def worker_id(self) -> int:
        return self.handle.worker_id

    @property
    def worker_name(self) -> str:
        return self.handle.worker_name

    @property
    def worker_generation(self) -> int:
        return self.handle.generation

    @property
    def address(self) -> str:
        return self.handle.address

    @property
    def client(self) -> EnvClient:
        return self.handle.client

    async def release(self) -> None:
        if self.released:
            return
        self.released = True
        await self.pool.release_worker(self)

    async def __aenter__(self) -> "EnvWorkerReservation":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()


class ManagedEnvClientPool(EnvClient):
    """Routes requests across env workers with scheduler-visible reservations.

    The pool remains usable as a normal ``EnvClient``. If a scheduler activates a
    reservation in the current task context, requests are routed to that reserved
    worker; otherwise the pool falls back to least-loaded routing and releases the
    reservation internally.
    """

    _active_reservation: contextvars.ContextVar[EnvWorkerReservation | None] = contextvars.ContextVar(
        "prime_rl_active_env_worker_reservation", default=None
    )

    def __init__(self, handles: list[EnvWorkerHandle], name: str | None = None):
        if not handles:
            raise ValueError("ManagedEnvClientPool requires at least one worker handle")
        self.handles = handles
        self._condition = asyncio.Condition()
        self._next_index = 0
        address = ",".join(handle.address for handle in self.handles)
        super().__init__(address=f"managed-pool://{address}", name=name)

    @property
    def clients(self) -> list[EnvClient]:
        return [handle.client for handle in self.handles]

    @staticmethod
    def _client_pending_count(client: EnvClient) -> int:
        pending_requests = getattr(client, "pending_requests", None)
        if pending_requests is None:
            return 0
        try:
            return len(pending_requests)
        except TypeError:
            return 0

    def _worker_load(self, handle: EnvWorkerHandle) -> int:
        # Scheduler-visible reservations and EnvClient pending requests normally
        # describe the same work. Use the larger value so orphaned internal
        # requests still count without double-counting healthy reservations.
        return max(handle.inflight_count, self._client_pending_count(handle.client))

    def worker_loads(self) -> dict[str, int]:
        return {handle.worker_name: self._worker_load(handle) for handle in self.handles}

    def at_capacity_count(self, max_requests_per_worker: int | None) -> int:
        if max_requests_per_worker is None:
            return 0
        return sum(
            1
            for handle in self.handles
            if not handle.quarantined and self._worker_load(handle) >= max_requests_per_worker
        )

    async def reserve_worker(self, weight: int = 1) -> EnvWorkerReservation:
        weight = max(1, weight)
        async with self._condition:
            while True:
                candidates = [handle for handle in self.handles if not handle.quarantined]
                if candidates:
                    min_load = min(self._worker_load(handle) for handle in candidates)
                    least_loaded = [handle for handle in candidates if self._worker_load(handle) == min_load]
                    chosen = least_loaded[self._next_index % len(least_loaded)]
                    self._next_index += 1
                    chosen.inflight_count += weight
                    return EnvWorkerReservation(self, chosen, weight=weight)
                await self._condition.wait()

    async def try_reserve_worker(
        self,
        *,
        weight: int = 1,
        max_requests_per_worker: int | None = None,
    ) -> EnvWorkerReservation | None:
        weight = max(1, weight)
        async with self._condition:
            candidates = [handle for handle in self.handles if not handle.quarantined]
            if max_requests_per_worker is not None:
                candidates = [
                    handle
                    for handle in candidates
                    if self._worker_load(handle) + weight <= max_requests_per_worker
                ]
            if not candidates:
                return None
            min_load = min(self._worker_load(handle) for handle in candidates)
            least_loaded = [handle for handle in candidates if self._worker_load(handle) == min_load]
            chosen = least_loaded[self._next_index % len(least_loaded)]
            self._next_index += 1
            chosen.inflight_count += weight
            return EnvWorkerReservation(self, chosen, weight=weight)

    async def release_worker(self, reservation: EnvWorkerReservation) -> None:
        async with self._condition:
            handle = self.handles[reservation.worker_id]
            handle.inflight_count = max(0, handle.inflight_count - reservation.weight)
            self._condition.notify_all()

    def activate_reservation(self, reservation: EnvWorkerReservation) -> contextvars.Token:
        return self._active_reservation.set(reservation)

    def reset_active_reservation(self, token: contextvars.Token) -> None:
        self._active_reservation.reset(token)

    def _get_active_reservation(self) -> EnvWorkerReservation | None:
        reservation = self._active_reservation.get()
        if reservation is None or reservation.pool is not self or reservation.released:
            return None
        handle = self.handles[reservation.worker_id]
        if handle.generation != reservation.worker_generation or handle.quarantined:
            return None
        return reservation

    async def wait_for_server_startup(self, timeout: float | None = None) -> None:
        await asyncio.gather(*(handle.client.wait_for_server_startup(timeout=timeout) for handle in self.handles))

    async def handle_health_request(self, request: HealthRequest, timeout: float | None) -> HealthResponse:
        responses = await asyncio.gather(
            *(handle.client.handle_health_request(request, timeout=timeout) for handle in self.handles),
            return_exceptions=True,
        )
        failures = [response for response in responses if isinstance(response, BaseException)]
        unsuccessful = [
            response
            for response in responses
            if not isinstance(response, BaseException) and not response.success
        ]
        if failures or unsuccessful:
            errors = [repr(error) for error in failures]
            errors.extend(response.error or "unhealthy" for response in unsuccessful)
            return HealthResponse(success=False, error="; ".join(errors))
        return HealthResponse(success=True)

    async def handle_run_rollout_request(
        self, request: RunRolloutRequest, timeout: float | None
    ) -> RunRolloutResponse:
        reservation = self._get_active_reservation()
        if reservation is not None:
            return await reservation.client.handle_run_rollout_request(request, timeout=timeout)
        async with await self.reserve_worker() as reservation:
            return await reservation.client.handle_run_rollout_request(request, timeout=timeout)

    async def handle_run_group_request(self, request: RunGroupRequest, timeout: float | None) -> RunGroupResponse:
        reservation = self._get_active_reservation()
        if reservation is not None:
            return await reservation.client.handle_run_group_request(request, timeout=timeout)
        async with await self.reserve_worker(weight=len(request.group_inputs)) as reservation:
            return await reservation.client.handle_run_group_request(request, timeout=timeout)

    async def run_rollout(
        self,
        input: vf.RolloutInput,
        client_config: vf.ClientConfig,
        model: str,
        sampling_args: vf.SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
    ) -> vf.RolloutOutput:
        reservation = self._get_active_reservation()
        if reservation is not None:
            return await reservation.client.run_rollout(
                input=input,
                client_config=client_config,
                model=model,
                sampling_args=sampling_args,
                max_retries=max_retries,
                state_columns=state_columns,
            )
        async with await self.reserve_worker() as reservation:
            return await reservation.client.run_rollout(
                input=input,
                client_config=client_config,
                model=model,
                sampling_args=sampling_args,
                max_retries=max_retries,
                state_columns=state_columns,
            )

    async def run_group(
        self,
        group_inputs: list[vf.RolloutInput],
        client_config: vf.ClientConfig,
        model: str,
        sampling_args: vf.SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
    ) -> list[vf.RolloutOutput]:
        reservation = self._get_active_reservation()
        if reservation is not None:
            return await reservation.client.run_group(
                group_inputs=group_inputs,
                client_config=client_config,
                model=model,
                sampling_args=sampling_args,
                max_retries=max_retries,
                state_columns=state_columns,
            )
        async with await self.reserve_worker(weight=len(group_inputs)) as reservation:
            return await reservation.client.run_group(
                group_inputs=group_inputs,
                client_config=client_config,
                model=model,
                sampling_args=sampling_args,
                max_retries=max_retries,
                state_columns=state_columns,
            )

    async def restart_worker(
        self,
        worker_id: int,
        expected_generation: int,
        reason: str,
        cancel_grace_seconds: float = 5.0,
    ) -> EnvWorkerHandle | None:
        handle = self.handles[worker_id]
        async with handle.restart_lock:
            if handle.generation != expected_generation:
                return None
            if handle.spec is None or handle.process is None:
                return None

            async with self._condition:
                handle.quarantined = True
                self._condition.notify_all()

            await asyncio.sleep(cancel_grace_seconds)
            await handle.client.close()

            handle.process.terminate()
            handle.process.join(timeout=5)
            if handle.process.is_alive():
                handle.process.kill()
                handle.process.join(timeout=5)

            address, process = spawn_env_server(
                env_id=handle.spec.env_id,
                env_args=handle.spec.env_args,
                extra_env_kwargs=handle.spec.extra_env_kwargs,
                address=handle.address,
                log_level=handle.spec.log_level,
                log_file=handle.spec.log_file,
                log_file_level=handle.spec.log_file_level,
                json_logging=handle.spec.json_logging,
            )
            client = setup_env_client(address=address, name=handle.worker_name)
            await client.wait_for_server_startup()

            async with self._condition:
                handle.address = address
                handle.process = process
                handle.client = client
                handle.generation += 1
                handle.restart_count += 1
                handle.inflight_count = 0
                handle.quarantined = False
                self._condition.notify_all()
            logging.getLogger(__name__).warning(
                "Restarted env worker %s (generation=%s, reason=%s)",
                handle.worker_name,
                handle.generation,
                reason,
            )
            return handle

    async def close(self) -> None:
        await asyncio.gather(*(handle.client.close() for handle in self.handles), return_exceptions=True)
        for handle in self.handles:
            if handle.process is None:
                continue
            handle.process.terminate()
            handle.process.join(timeout=5)
            if handle.process.is_alive():
                handle.process.kill()
                handle.process.join(timeout=5)


async def wait_for_env_servers(env_clients: list[EnvClient]) -> None:
    await asyncio.gather(*[env_client.wait_for_server_startup() for env_client in env_clients])


async def run_rollout(
    env: vf.Environment,
    client: vf.ClientConfig,
    model_name: str,
    example: dict,
    sampling_args: dict,
    max_retries: int = DEFAULT_RETRIES,
    state_columns: list[str] = DEFAULT_STATE_COLUMNS,
    rollout_timeout_seconds: float | None = None,
) -> vf.RolloutOutput:
    """
    Wrapper for vf.Environment.run_rollout().

    Asynchronously generates and scores one rollout.
    """
    state_columns = state_columns + REQUIRED_STATE_COLUMNS
    rollout_input = vf.RolloutInput(**example)
    rollout = env.run_rollout(
        rollout_input,
        client=client,
        model=model_name,
        sampling_args=sampling_args,
        max_retries=max_retries,
        state_columns=state_columns,
    )
    try:
        if rollout_timeout_seconds is None:
            return await rollout
        return await asyncio.wait_for(rollout, timeout=rollout_timeout_seconds)
    except TimeoutError:
        return make_timeout_rollout(example, sampling_args, rollout_timeout_seconds)


async def run_group(
    env: vf.Environment,
    client: vf.ClientConfig,
    model_name: str,
    example: dict,
    rollouts_per_example: int,
    sampling_args: dict,
    max_retries: int = DEFAULT_RETRIES,
    state_columns: list[str] = DEFAULT_STATE_COLUMNS,
    rollout_timeout_seconds: float | None = None,
) -> list[vf.RolloutOutput]:
    """
    Wrapper for vf.Environment.run_group().

    Asynchronously generates and scores a group.
    """
    state_columns = state_columns + REQUIRED_STATE_COLUMNS
    group_inputs = [vf.RolloutInput(**example) for _ in range(rollouts_per_example)]
    group = env.run_group(
        group_inputs,
        client=client,
        model=model_name,
        sampling_args=sampling_args,
        max_retries=max_retries,
        state_columns=state_columns,
    )
    try:
        if rollout_timeout_seconds is None:
            return await group
        return await asyncio.wait_for(group, timeout=rollout_timeout_seconds)
    except TimeoutError:
        return [
            make_timeout_rollout(example, sampling_args, rollout_timeout_seconds)
            for _ in range(rollouts_per_example)
        ]


def make_timeout_rollout(
    example: dict,
    sampling_args: dict,
    rollout_timeout_seconds: float,
) -> vf.RolloutOutput:
    """Create a non-trainable rollout output for a wall-clock timeout."""
    message = f"rollout exceeded rollout_timeout_seconds={rollout_timeout_seconds}"
    return vf.RolloutOutput(
        example_id=example.get("example_id", -1),
        task=example.get("task", ""),
        prompt=example.get("prompt"),
        completion=None,
        reward=0.0,
        timing={"total_ms": rollout_timeout_seconds * 1000.0},
        is_completed=False,
        is_truncated=False,
        metrics={"rollout/timeout": 1.0},
        answer="",
        info=example.get("info", {}),
        error={
            "error": message,
            "error_chain_repr": message,
            "error_chain_str": message,
        },
        stop_condition="rollout_timeout",
        trajectory=[],
        tool_defs=[],
        token_usage={"input_tokens": 0.0, "output_tokens": 0.0},
        sampling_args=sampling_args,
    )


# TODO: migrate this to vf.Environment.generate() once it supports multiple clients
async def generate(
    env: vf.Environment,
    model_name: str,
    examples: list,
    rollouts_per_example: int,
    sampling_args: dict,
    clients: list[vf.ClientConfig] | None = None,
    get_client: Callable[[], Awaitable[vf.ClientConfig]] | None = None,
    max_retries: int = DEFAULT_RETRIES,
    state_columns: list[str] = DEFAULT_STATE_COLUMNS,
    pbar_description: str = "Generating rollouts",
    rollout_timeout_seconds: float | None = None,
) -> list[vf.RolloutOutput]:
    """
    Wrapper for vf.Environment.generate().

    NOTE: Currently we cannot use vf.Environment.generate() directly because it does not support multiple clients.

    Asynchronously generates and scores a list of groups.
    """

    if not clients and get_client is None:
        raise ValueError("generate requires at least one client or a get_client callback")

    if get_client is None:
        client_cycle = cycle(clients)

        async def get_client() -> vf.ClientConfig:
            return next(client_cycle)

    total_rollouts = len(examples) * rollouts_per_example
    pbar = ProgressTracker(total=total_rollouts, desc=pbar_description)

    async def run_group_with_progress(example):
        client = await get_client()
        result = await run_group(
            env=env,
            client=client,
            model_name=model_name,
            example=example,
            rollouts_per_example=rollouts_per_example,
            max_retries=max_retries,
            state_columns=state_columns,
            sampling_args=sampling_args,
            rollout_timeout_seconds=rollout_timeout_seconds,
        )
        pbar.update(rollouts_per_example)
        return result

    try:
        group_outputs_list: list[list[vf.RolloutOutput]] = await asyncio.gather(
            *[run_group_with_progress(example) for example in examples]
        )
    finally:
        pbar.close()

    return [output for group_outputs in group_outputs_list for output in group_outputs]


async def evaluate(
    env: vf.Environment,
    model_name: str,
    sampling_args: dict,
    num_examples: int,
    rollouts_per_example: int,
    clients: list[vf.ClientConfig] | None = None,
    get_client: Callable[[], Awaitable[vf.ClientConfig]] | None = None,
    max_retries: int = DEFAULT_RETRIES,
    state_columns: list[str] = DEFAULT_STATE_COLUMNS,
    rollout_timeout_seconds: float | None = None,
) -> list[vf.RolloutOutput]:
    """
    Wrapper for vf.Environment.evaluate().

    NOTE: Currently we cannot use vf.Environment.evaluate() directly because it does not support multiple clients.
          Instead, we use our generate() wrapper which round-robins clients.

    """
    inputs = env._get_eval_inputs(num_examples, rollouts_per_example)
    outputs = await generate(
        env=env,
        clients=clients,
        get_client=get_client,
        model_name=model_name,
        examples=inputs,
        # _get_eval_inputs() already repeats the examples, this currently means
        # we do not support eval envs with group scoring well -- this should be
        # resolved once we can use vf.Environment.generate() and
        # vf.Environment.evaluate() directly though
        rollouts_per_example=1,
        sampling_args=sampling_args,
        max_retries=max_retries,
        state_columns=state_columns,
        rollout_timeout_seconds=rollout_timeout_seconds,
    )
    return outputs


# TODO: remove once usage is tracked by verifiers
def get_prompt_len(output: vf.RolloutOutput) -> int:
    """
    Computes the number of prompt tokens from vf.RolloutOutput. Defined as the
    number of prompt ids from the first trajectory step. If raw tokens are not
    available, falls back to checking the usage of the first response.
    """
    if not output["trajectory"]:
        return 0
    first_step = output["trajectory"][0]
    if first_step["tokens"] is not None:
        return len(first_step["tokens"]["prompt_ids"])
    first_step_response = first_step["response"]
    return (first_step_response.get("usage") or {}).get("prompt_tokens", 0)


# TODO: remove once usage is tracked by verifiers
def get_seq_len(output: vf.RolloutOutput) -> int:
    """
    Computes the number of tokens from vf.RolloutOutput. Defined as the sum of prompt
    and completion tokens from the last trajectory step. If raw tokens are not
    available, falls back to checking the usage of the last response.
    """
    if not output["trajectory"]:
        return 0
    last_step = output["trajectory"][-1]
    if last_step["tokens"] is not None:
        return len(last_step["tokens"]["prompt_ids"]) + len(last_step["tokens"]["completion_ids"])
    last_step_response = last_step["response"]
    return (last_step_response.get("usage") or {}).get("total_tokens", 0)


# TODO: remove once usage is tracked by verifiers
def get_completion_len(output: vf.RolloutOutput) -> int:
    """
    Computes the number of completion tokens from vf.RolloutOutput. Defined as
    the difference between the total number of tokens and the number of prompt
    tokens.
    """
    return get_seq_len(output) - get_prompt_len(output)


def task_uses_group_scoring(env: vf.Environment, task_name: str) -> bool:
    """Check if a task's rubric contains any group-level reward functions."""
    rubric = env.get_env_for_task(task_name).rubric
    return any(rubric._is_group_func(func) for func in rubric._get_reward_funcs())


def intercept_vf_logging(logger: str = "verifiers", level: str = "DEBUG", prefix: str | None = None):
    """Intercepts verifiers logging and routes through prime-rl logger with optional prefix."""
    vf_logger = logging.getLogger(logger)
    vf_logger.handlers.clear()
    vf_logger.addHandler(InterceptHandler(prefix=prefix))
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False
