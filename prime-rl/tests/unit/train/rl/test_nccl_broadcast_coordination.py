from pathlib import Path
from types import SimpleNamespace

from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast
from prime_rl.trainer.runs import Progress


def _broadcast(tmp_path: Path, *, is_master: bool) -> NCCLWeightBroadcast:
    broadcast = NCCLWeightBroadcast.__new__(NCCLWeightBroadcast)
    broadcast.logger = SimpleNamespace(debug=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
    broadcast.world = SimpleNamespace(is_master=is_master)
    run_dir = tmp_path / "run_default"
    broadcast.multi_run_manager = SimpleNamespace(
        used_idxs=[0],
        ready_to_update=[True],
        progress={0: Progress(step=30)},
        get_run_dir=lambda idx: run_dir,
    )
    return broadcast


def test_nccl_broadcast_non_master_waits_for_same_ready_marker(tmp_path):
    broadcast = _broadcast(tmp_path, is_master=False)

    notified_runs = broadcast._notify_orchestrator()

    save_dir = tmp_path / "run_default" / "broadcasts" / "step_30"
    assert notified_runs == [(0, save_dir)]
    assert not (save_dir / "STABLE").exists()
    assert broadcast.multi_run_manager.ready_to_update[0] is True


def test_nccl_broadcast_master_notifies_and_all_ranks_wait_before_materializing(monkeypatch, tmp_path):
    broadcast = _broadcast(tmp_path, is_master=True)
    calls = []

    def notify():
        calls.append("notify")
        return [(0, tmp_path / "run_default" / "broadcasts" / "step_30")]

    def wait(notified_runs):
        calls.append(("wait", notified_runs))

    class Sender:
        def broadcast_weights(self, model, step):
            calls.append(("broadcast", step))

    monkeypatch.setattr(broadcast, "_notify_orchestrator", notify)
    monkeypatch.setattr(broadcast, "_wait_for_nccl_ready", wait)
    broadcast.nccl_broadcast_sender = Sender()

    broadcast.broadcast_weights(model=object(), step=30)

    assert calls == [
        "notify",
        ("wait", [(0, tmp_path / "run_default" / "broadcasts" / "step_30")]),
        ("broadcast", 30),
    ]
