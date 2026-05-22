import json
import os
import copy
import subprocess
import sys
import time
import uuid
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread

import pynvml
import tomli_w

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import validate_output_dir
from prime_rl.utils.process import cleanup_processes, cleanup_threads, monitor_process
from prime_rl.utils.utils import (
    get_free_port,
    get_log_dir,
)

RL_TOML = "rl.toml"
RL_SBATCH = "rl.sbatch"

TRAINER_TOML = "trainer.toml"
ORCHESTRATOR_TOML = "orchestrator.toml"
INFERENCE_TOML = "inference.toml"
TEACHER_INFERENCE_TOML = "teacher_inference.toml"


def _configure_local_nccl_independent_inference_servers(config: RLConfig) -> list[Path]:
    """Split local NCCL inference DP into independent vLLM servers.

    Prime-RL's NCCL broadcaster assigns inference ranks by enumerating
    orchestrator admin clients. A single vLLM process with internal DP exposes
    one admin endpoint, so all DP workers receive the same server rank. For
    local full-FT NCCL runs, launch one vLLM server per inference TP group
    instead, giving the orchestrator one admin client per receiver group.
    """
    if config.deployment.type != "single_node":
        return []
    if config.inference is None or config.weight_broadcast is None or config.weight_broadcast.type != "nccl":
        return []

    num_infer_gpus = config.deployment.num_infer_gpus
    tp = config.inference.parallel.tp
    if num_infer_gpus <= tp:
        return []
    if num_infer_gpus % tp != 0:
        raise ValueError(
            f"num_infer_gpus ({num_infer_gpus}) must be divisible by inference.parallel.tp ({tp}) "
            "for local independent NCCL inference servers."
        )
    if config.inference.data_parallel_size_local not in (None, 1):
        raise ValueError(
            "Local independent NCCL inference servers require inference.data_parallel_size_local to be unset or 1."
        )

    server_count = num_infer_gpus // tp
    base_port = config.inference.server.port
    host = config.inference.server.host or "localhost"
    client_host = "localhost" if host in {"0.0.0.0", "::"} else host

    config.orchestrator.client.base_url = [f"http://{client_host}:{base_port + i}/v1" for i in range(server_count)]
    config.orchestrator.client.dp_rank_count = 1

    # The base inference config written to inference.toml is server 0. Extra
    # server configs are emitted by rl_local after write_subconfigs().
    config.inference.server.port = base_port
    config.inference.parallel.dp = 1
    config.inference.data_parallel_size_local = 1
    config.inference.api_server_count = 1

    return [Path(f"inference_{i}.toml") for i in range(1, server_count)]


def _write_local_nccl_extra_inference_configs(config: RLConfig, config_dir: Path, extra_paths: list[Path]) -> None:
    if config.inference is None or not extra_paths:
        return
    import tomli_w

    base_port = config.inference.server.port
    for index, relative_path in enumerate(extra_paths, start=1):
        server_config = copy.deepcopy(config.inference)
        server_config.server.port = base_port + index
        server_config.parallel.dp = 1
        server_config.data_parallel_size_local = 1
        server_config.api_server_count = 1
        with open(config_dir / relative_path, "wb") as f:
            tomli_w.dump(server_config.model_dump(exclude_none=True, mode="json"), f)


def get_physical_gpu_ids() -> list[int]:
    """Return physical GPU IDs visible to the launcher."""
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_visible is None:
        pynvml.nvmlInit()
        return list(range(pynvml.nvmlDeviceGetCount()))
    return [int(token.strip()) for token in raw_visible.split(",") if token.strip()]


def write_config(config: RLConfig, output_dir: Path, exclude: set[str] | None = None) -> None:
    """Write resolved config to disk, excluding launcher-only fields."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = config.model_dump(exclude=exclude, exclude_none=True, mode="json")
    with open(output_dir / RL_TOML, "wb") as f:
        tomli_w.dump(config_dict, f)


def write_subconfigs(config: RLConfig, output_dir: Path) -> None:
    """Write resolved subconfigs to disk as TOML files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / TRAINER_TOML, "wb") as f:
        tomli_w.dump(config.trainer.model_dump(exclude_none=True, mode="json"), f)

    with open(output_dir / ORCHESTRATOR_TOML, "wb") as f:
        tomli_w.dump(config.orchestrator.model_dump(exclude_none=True, mode="json"), f)

    if config.inference is not None:
        with open(output_dir / INFERENCE_TOML, "wb") as f:
            tomli_w.dump(config.inference.model_dump(exclude_none=True, mode="json"), f)

    teacher_inference = getattr(config, "teacher_inference", None)
    if teacher_inference is not None:
        with open(output_dir / TEACHER_INFERENCE_TOML, "wb") as f:
            tomli_w.dump(teacher_inference.model_dump(exclude_none=True, mode="json"), f)


def check_gpus_available(gpu_ids: list[int]) -> None:
    """Raise error if there are existing processes on the specified GPUs."""
    pynvml.nvmlInit()

    occupied = []
    for gpu_id in gpu_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if processes:
            pids = [p.pid for p in processes]
            occupied.append((gpu_id, pids))

    if occupied:
        msg = "Existing processes found on GPUs:\n"
        for gpu_id, pids in occupied:
            msg += f"  GPU {gpu_id}: PIDs {pids}\n"
        msg += "Kill these processes or use different GPUs."
        raise RuntimeError(msg)


def rl_local(config: RLConfig):
    assert config.deployment.type == "single_node"

    logger = setup_logger(
        config.log.level or "info",
        log_file=config.output_dir / "logs" / "rl.log" if config.log.file else None,
        json_logging=config.log.json_logging,
    )

    if config.wandb is not None and not config.wandb.offline and os.environ.get("WANDB_API_KEY") is None:
        logger.warning(
            "W&B online mode is enabled but WANDB_API_KEY is not set in the current environment. "
            "If you rely on persisted wandb login state instead, verify it is available to child processes before restarting the run."
        )

    extra_inference_config_paths = _configure_local_nccl_independent_inference_servers(config)

    config_dir = config.output_dir / "configs"
    write_subconfigs(config, config_dir)
    _write_local_nccl_extra_inference_configs(config, config_dir, extra_inference_config_paths)
    logger.info(f"Wrote subconfigs to {config_dir}")

    if config.dry_run:
        logger.success("Dry run complete. To start an RL run locally, remove --dry-run from your command.")
        return

    # Derive launcher-local GPU IDs from deployment config
    gpu_offset = 0
    num_infer_gpus = config.deployment.num_infer_gpus if config.inference is not None else 0
    infer_local_gpu_ids = list(range(gpu_offset, gpu_offset + num_infer_gpus))
    gpu_offset += num_infer_gpus
    trainer_local_gpu_ids = list(range(gpu_offset, gpu_offset + config.deployment.num_train_gpus))
    gpu_offset += config.deployment.num_train_gpus
    num_teacher_gpus = config.deployment.num_teacher_gpus or 0
    teacher_local_gpu_ids = list(range(gpu_offset, gpu_offset + num_teacher_gpus)) if num_teacher_gpus > 0 else []

    total_requested_gpus = num_infer_gpus + config.deployment.num_train_gpus + num_teacher_gpus
    physical_gpu_ids = get_physical_gpu_ids()
    if total_requested_gpus > len(physical_gpu_ids):
        raise ValueError(
            f"Requested {total_requested_gpus} GPUs via deployment settings, but only "
            f"{len(physical_gpu_ids)} physical GPU(s) are available: {physical_gpu_ids}"
        )
    physical_gpu_mapping = {local_id: physical_gpu_ids[local_id] for local_id in range(total_requested_gpus)}
    logger.info(f"Using local->physical GPU mapping: {physical_gpu_mapping}")

    infer_gpu_ids = [physical_gpu_mapping[local_gpu_id] for local_gpu_id in infer_local_gpu_ids]
    trainer_gpu_ids = [physical_gpu_mapping[local_gpu_id] for local_gpu_id in trainer_local_gpu_ids]
    teacher_gpu_ids = [physical_gpu_mapping[local_gpu_id] for local_gpu_id in teacher_local_gpu_ids]

    start_command = sys.argv
    logger.info("Starting RL run")
    logger.debug(f"RL start command: {' '.join(start_command)}")

    # Check for existing processes on GPUs
    all_gpu_ids = list(set(infer_gpu_ids + trainer_gpu_ids + teacher_gpu_ids))
    check_gpus_available(all_gpu_ids)

    # Validate client port matches inference server port
    if config.inference is not None and not config.orchestrator.client.is_elastic:
        from urllib.parse import urlparse

        base_url = config.orchestrator.client.base_url[0]
        parsed = urlparse(base_url)
        client_port = parsed.port
        expected_port = config.inference.server.port
        if client_port != expected_port:
            raise ValueError(
                f"orchestrator.client.base_url port ({client_port}) does not match "
                f"inference.server.port ({expected_port}). "
                f"Update the base_url to use port {expected_port} to match the inference server."
            )

    # Prepare paths to communicate with the trainer
    log_dir = get_log_dir(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Start processes
    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}

    try:
        # Optionally, start inference process(es)
        if config.inference:
            inference_config_paths = [Path(INFERENCE_TOML), *extra_inference_config_paths]
            tp = config.inference.parallel.tp
            for server_idx, inference_config_path in enumerate(inference_config_paths):
                server_gpu_ids = infer_gpu_ids[server_idx * tp : (server_idx + 1) * tp]
                if not server_gpu_ids:
                    raise RuntimeError(
                        f"No GPU IDs available for inference server {server_idx}; "
                        f"inference_config_paths={inference_config_paths}, infer_gpu_ids={infer_gpu_ids}"
                    )
                inference_cmd = ["uv", "run", "--no-sync", "inference", "@", (config_dir / inference_config_path).as_posix()]
                process_name = "inference" if server_idx == 0 else f"inference_{server_idx}"
                log_name = "inference.stdout" if server_idx == 0 else f"inference_{server_idx}.stdout"
                logger.info(f"Starting {process_name} on GPU(s) {' '.join(map(str, server_gpu_ids))}")
                logger.debug(f"{process_name} start command: {' '.join(inference_cmd)}")
                # If we don't log stdout, the server hangs
                with open(log_dir / log_name, "w") as log_file:
                    inference_process = Popen(
                        inference_cmd,
                        env={
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": ",".join(map(str, server_gpu_ids)),
                        },
                        stdout=log_file,
                        stderr=log_file,
                    )
                processes.append(inference_process)

                # Start monitoring thread
                stop_event = Event()
                stop_events[process_name] = stop_event
                monitor_thread = Thread(
                    target=monitor_process,
                    args=(inference_process, stop_event, error_queue, process_name),
                    daemon=True,
                )
                monitor_thread.start()
                monitor_threads.append(monitor_thread)
        else:
            logger.warning(
                "No inference config specified, skipping starting inference server. Make sure your inference server is running."
            )

        # Optionally, start teacher inference process
        if config.teacher_inference:
            if not teacher_gpu_ids:
                raise ValueError(
                    "teacher_inference is configured but deployment.num_teacher_gpus is not set. "
                    "Either set deployment.num_teacher_gpus to start a teacher inference server, "
                    "or omit teacher_inference and configure orchestrator.teacher_model to use an existing server."
                )

            teacher_inference_cmd = [
                "uv",
                "run",
                "--no-sync",
                "inference",
                "@",
                (config_dir / TEACHER_INFERENCE_TOML).as_posix(),
            ]
            logger.info(f"Starting teacher inference process on GPU(s) {' '.join(map(str, teacher_gpu_ids))}")
            logger.debug(f"Teacher inference start command: {' '.join(teacher_inference_cmd)}")
            with open(log_dir / "teacher_inference.stdout", "w") as log_file:
                teacher_inference_process = Popen(
                    teacher_inference_cmd,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": ",".join(map(str, teacher_gpu_ids)),
                    },
                    stdout=log_file,
                    stderr=log_file,
                )
            processes.append(teacher_inference_process)

            # Start monitoring thread
            stop_event = Event()
            stop_events["teacher_inference"] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(teacher_inference_process, stop_event, error_queue, "teacher_inference"),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)
        elif (
            config.trainer.loss.type == "default" and config.trainer.loss.teacher_tau > 0
        ) or config.orchestrator.teacher_model:
            logger.warning(
                "No teacher_inference config specified, skipping starting teacher inference server. "
                "Is your teacher inference server running? Make sure orchestrator.teacher_model is configured."
            )

        # Start orchestrator process
        orchestrator_cmd = [
            "uv",
            "run",
            "--no-sync",
            "orchestrator",
            "@",
            (config_dir / ORCHESTRATOR_TOML).as_posix(),
        ]
        logger.info("Starting orchestrator process")
        logger.debug(f"Orchestrator start command: {' '.join(orchestrator_cmd)}")
        with open(log_dir / "orchestrator.stdout", "w") as log_file:
            orchestrator_process = Popen(
                orchestrator_cmd,
                stdout=log_file,
                stderr=log_file,
                env={
                    **os.environ,
                    "LOGURU_FORCE_COLORS": "1",
                    "WANDB_PROGRAM": "uv run rl",
                    "WANDB_ARGS": json.dumps(start_command),
                },
            )
        processes.append(orchestrator_process)

        # Start monitoring thread
        stop_event = Event()
        stop_events["orchestrator"] = stop_event
        monitor_thread = Thread(
            target=monitor_process,
            args=(orchestrator_process, stop_event, error_queue, "orchestrator"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        # Start training process
        trainer_cmd = [
            "uv",
            "run",
            "--no-sync",
            "env",
            "PYTHONUNBUFFERED=1",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "torchrun",
            f"--rdzv-endpoint=localhost:{get_free_port()}",
            f"--rdzv-id={uuid.uuid4().hex}",
            # Pipe all logs to file, and only master rank logs to stdout
            f"--log-dir={config.output_dir / 'torchrun'}",
            "--local-ranks-filter=0",
            "--redirect=3",
            "--tee=3",
            f"--nproc-per-node={len(trainer_gpu_ids)}",
            "-m",
            "prime_rl.trainer.rl.train",
            "@",
            (config_dir / TRAINER_TOML).as_posix(),
        ]
        logger.info(f"Starting trainer on GPU(s) {' '.join(map(str, trainer_gpu_ids))}")
        logger.debug(f"Training start command: {' '.join(trainer_cmd)}")
        with open(log_dir / "trainer.stdout", "w") as log_file:
            trainer_process = Popen(
                trainer_cmd,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, trainer_gpu_ids)),
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "LOGURU_FORCE_COLORS": "1",
                    "WANDB_PROGRAM": "uv run rl",
                    "WANDB_ARGS": json.dumps(start_command),
                },
                stdout=log_file,
                stderr=log_file,
            )
        processes.append(trainer_process)

        # Start monitoring thread
        stop_event = Event()
        stop_events["trainer"] = stop_event
        monitor_thread = Thread(
            target=monitor_process, args=(trainer_process, stop_event, error_queue, "trainer"), daemon=True
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        # Monitor all processes for failures
        logger.success("Startup complete. Showing trainer logs...")

        tail_process = Popen(["tail", "-F", log_dir / "trainer.stdout"])
        processes.append(tail_process)

        # Check for errors from monitor threads
        while not (stop_events["orchestrator"].is_set() and stop_events["trainer"].is_set()):
            if error_queue:
                error = error_queue[0]
                logger.error(f"Error: {error}")
                logger.error("Terminating all processes...")
                cleanup_threads(monitor_threads)
                cleanup_processes(processes)
                sys.exit(1)

            # Small delay to avoid busy waiting
            time.sleep(1)

        # Check if any critical process failed
        if orchestrator_process.returncode != 0:
            logger.error(f"Orchestrator failed with exit code {orchestrator_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        if trainer_process.returncode != 0:
            logger.error(f"Trainer failed with exit code {trainer_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        logger.success("RL training finished!")

        # Cleanup threads and processes
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)

    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, terminating all processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        raise


def write_slurm_script(config: RLConfig, config_dir: Path, script_path: Path) -> None:
    """Write the SLURM script to disk."""
    from jinja2 import Environment, FileSystemLoader

    assert config.slurm is not None
    assert config.slurm.template_path is not None

    env = Environment(loader=FileSystemLoader(config.slurm.template_path.parent), keep_trailing_newline=True)
    template = env.get_template(config.slurm.template_path.name)

    if config.deployment.type == "single_node":
        script = template.render(
            **config.slurm.template_vars,
            config_path=config_dir / RL_TOML,
            output_dir=config.output_dir,
            gpus_per_node=config.deployment.gpus_per_node,
        )
    else:
        script = template.render(
            **config.slurm.template_vars,
            config_dir=config_dir,  # TODO: should prob have each subconfig path separately
            output_dir=config.output_dir,
            orchestrator_output_dir=config.orchestrator.output_dir,
            num_train_nodes=config.deployment.num_train_nodes,
            num_infer_nodes=config.deployment.num_infer_nodes,
            num_teacher_nodes=config.deployment.num_teacher_nodes,
            gpus_per_node=config.deployment.gpus_per_node,
            inference_tp=config.inference.parallel.tp if config.inference else 1,
            inference_enable_expert_parallel=config.inference.enable_expert_parallel if config.inference else False,
            inference_data_parallel_rpc_port=config.inference.data_parallel_rpc_port if config.inference else 29600,
            use_nccl_broadcast=config.weight_broadcast is not None and config.weight_broadcast.type == "nccl",
        )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)


def rl_slurm(config: RLConfig):
    assert config.slurm is not None

    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)

    config_dir = config.output_dir / "configs"
    if config.deployment.type == "single_node":
        write_config(config, config_dir, exclude={"slurm", "dry_run", "clean_output_dir"})
        logger.info(f"Wrote config to {config_dir / RL_TOML}")

        log_dir = get_log_dir(config.output_dir)
        log_message = (
            f"Logs:\n"
            f"  Trainer:       tail -F {log_dir}/trainer.stdout\n"
            f"  Orchestrator:  tail -F {log_dir}/orchestrator.stdout\n"
            f"  Inference:     tail -F {log_dir}/inference.stdout"
        )
    else:
        write_subconfigs(config, config_dir)
        logger.info(f"Wrote subconfigs to {config_dir}")

        slurm_log_dir = config.output_dir / "slurm"
        log_lines = [f"  Trainer:       tail -F {slurm_log_dir}/latest_train_node_rank_0.log"]
        if config.deployment.num_infer_nodes > 0:
            log_lines.append(f"  Orchestrator:  tail -F {slurm_log_dir}/latest_orchestrator.log")
            log_lines.append(f"  Inference:     tail -F {slurm_log_dir}/latest_infer_node_rank_0.log")
        log_message = "Logs:\n" + "\n".join(log_lines)

    script_path = config.output_dir / RL_SBATCH
    write_slurm_script(config, config_dir, script_path)
    logger.info(f"Wrote SLURM script to {script_path}")

    if config.dry_run:
        logger.success(f"Dry run complete. To submit manually:\n\n  sbatch {script_path}\n\n{log_message}")
        return

    logger.info(f"Submitting: sbatch {script_path}")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed: {result.stderr.strip()}")
        sys.exit(1)

    logger.success(f"{result.stdout.strip()}\n\n{log_message}")


def rl(config: RLConfig):
    resuming = config.ckpt is not None and config.ckpt.resume_step is not None
    clean = config.clean_output_dir and not os.environ.get("NEVER_CLEAN_OUTPUT_DIR")
    ckpt_output_dir = config.ckpt.output_dir if config.ckpt else None
    validate_output_dir(config.output_dir, resuming=resuming, clean=clean, ckpt_output_dir=ckpt_output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_output_dir is not None:
        ckpt_output_dir.mkdir(parents=True, exist_ok=True)

    if config.slurm is not None:
        rl_slurm(config)
    else:
        rl_local(config)


def main():
    rl(cli(RLConfig))


if __name__ == "__main__":
    main()
