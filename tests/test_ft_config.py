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
