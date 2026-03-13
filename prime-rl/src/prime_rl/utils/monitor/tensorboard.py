from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from prime_rl.configs.shared import TensorBoardConfig
from prime_rl.utils.monitor.base import Monitor


class TensorBoardMonitor(Monitor):
    """Logs scalar metrics to TensorBoard event files."""

    def __init__(self, config: TensorBoardConfig, output_dir: Path | None = None):
        self.config = config
        self.output_dir = output_dir
        log_dir = config.log_dir or ((output_dir / "tensorboard") if output_dir is not None else Path("tensorboard"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir), flush_secs=config.flush_secs)
        self.history: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.history.append(metrics)
        for key, value in metrics.items():
            if isinstance(value, bool):
                self.writer.add_scalar(key, int(value), global_step=step)
            elif isinstance(value, int | float):
                self.writer.add_scalar(key, value, global_step=step)

    def log_samples(self, rollouts: list[Any], step: int) -> None:
        del rollouts, step

    def log_final_samples(self) -> None:
        pass

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        del filename

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        del distributions, step

    def flush(self, step: int) -> None:
        del step
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()