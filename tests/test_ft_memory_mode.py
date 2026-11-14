import ast
from pathlib import Path

TRAIN = Path(__file__).parent.parent / "train.py"


def guarded_calls():
    """Names of calls that sit under an `if not ft_memory_restart(cfg):` in train.py."""
    tree = ast.parse(TRAIN.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "ft_memory_restart" in ast.unparse(node.test):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    out.add(ast.unparse(inner.func))
    return out


def test_memory_restart_skips_every_startup_shared_fs_touch():
    calls = guarded_calls()
    for name in ("mkdir", "log_restart_infos", "get_last_restart_step", "open", "check_cluster"):
        assert name in calls, f"{name} runs on a memory restart"


def test_memory_restart_only_with_ft_and_memory_mode(monkeypatch):
    from types import SimpleNamespace

    from xllm.ft.trainer import memory_restart

    cfg = SimpleNamespace(ft=SimpleNamespace(enabled=True))
    monkeypatch.setenv("FT_RESTORE", "memory")
    assert memory_restart(cfg)
    monkeypatch.setenv("FT_RESTORE", "disk")
    assert not memory_restart(cfg)
    cfg.ft.enabled = False
    monkeypatch.setenv("FT_RESTORE", "memory")
    assert not memory_restart(cfg)


def test_ft_trainer_ignores_sigusr1(monkeypatch):
    import signal
    import types

    import xllm.ft.trainer as trainer

    monkeypatch.setattr(trainer, "startup_gate", lambda *a: None)
    monkeypatch.setattr(trainer, "Replica", lambda *a: None)
    monkeypatch.setattr(trainer, "gpus_per_node", lambda: 8)
    saved = signal.getsignal(signal.SIGUSR1)
    try:
        signal.signal(signal.SIGUSR1, lambda *a: None)  # as xLLM's init_signal_handler() leaves it
        cfg = types.SimpleNamespace(ft=types.SimpleNamespace(slot_dir="/x", k=0, mtbf_sys=1.0), dcp_for_optimizer=False)
        trainer.FtTrainer(cfg, None, None)
        assert signal.getsignal(signal.SIGUSR1) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGUSR1, saved)
