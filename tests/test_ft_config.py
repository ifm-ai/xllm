import pytest

from xllm.config import DataLoaderConfig, ModelConf, OptimConf, SlurmConf, TokenizerConf, TrainerConf, ValidConf
from xllm.ft.config import FtConf
from xllm.ft.node import discover_node, existing_ancestor

RENAME = {"log_dir": "local_log_dir"}


def make_cfg(**kwargs) -> TrainerConf:
    ft = FtConf(**{RENAME.get(k[3:], k[3:]): kwargs.pop(k) for k in [k for k in kwargs if k.startswith("ft_")]})
    return TrainerConf(
        ft=ft,
        tokenizer=TokenizerConf(),
        slurm=SlurmConf(),
        model=ModelConf(),
        optim=OptimConf(),
        valid=ValidConf(),
        dataloader=DataLoaderConfig(),
        **kwargs,
    )


def test_ft_is_off_by_default():
    cfg = make_cfg()
    assert cfg.ft.enabled is False
    assert cfg.ft.mtbf_sys is None


def test_ft_on_requires_mtbf():
    with pytest.raises(AssertionError):
        make_cfg(ft_enabled=True)
    assert make_cfg(ft_enabled=True, ft_mtbf_sys=3600.0).ft.enabled


def test_xid_list_must_parse():
    make_cfg(ft_enabled=True, ft_mtbf_sys=3600.0, ft_xid_replace="48,79")
    with pytest.raises(AssertionError):
        make_cfg(ft_enabled=True, ft_mtbf_sys=3600.0, ft_xid_replace="48,abc")


def test_discover_node(tmp_path):
    missing = tmp_path / "a" / "b"
    assert existing_ancestor(missing) == tmp_path
    node = discover_node(str(missing))
    assert node.gpus >= 0
    assert 0 < node.shm_free_bytes <= node.shm_bytes
    assert node.ram_bytes > 0


def test_log_dir_is_dump_dir_when_off_and_local_when_on(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    assert make_cfg(dump_dir="/shared/run").log_dir == "/shared/run"
    cfg = make_cfg(dump_dir="/shared/run", ft_enabled=True, ft_mtbf_sys=3600.0, ft_log_dir="/local/logs")
    assert cfg.log_dir == "/local/logs/job_77"


def test_every_dump_dir_logger_uses_log_dir():
    """Under ft every logger must write to the per-host log_dir; with ft off log_dir is dump_dir."""
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).parent.parent / "train.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    for name in ("add_logger_file_handler", "build_mlogger", "log_restart_infos", "check_cluster"):
        found = [c for c in calls if ast.unparse(c.func) == name]
        assert found, name
        assert all("cfg.log_dir" in ast.unparse(c) for c in found), name
