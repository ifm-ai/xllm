from xllm.checkpointing import CHECKPOINT_FOLDER_NAME, get_checkpoint_dir
from xllm.ft.save_rule import FT_SAVE, FT_SKIP, ft_save_decision


def make(dump, step, complete):
    d = get_checkpoint_dir(dump, step)
    d.mkdir(parents=True)
    if complete:
        (d / "config.json").write_text("{}")
    return d


def test_complete_checkpoint_is_kept_and_the_save_skipped(tmp_path):
    d = make(tmp_path, 20, complete=True)
    assert ft_save_decision(tmp_path, d, 20, global_rank=0) == (FT_SKIP, False)
    assert (d / "config.json").exists()


def test_incomplete_checkpoint_is_moved_aside(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    d = make(tmp_path, 20, complete=False)
    assert ft_save_decision(tmp_path, d, 20, global_rank=0) == (FT_SAVE, False)
    assert not d.exists()
    assert (tmp_path / CHECKPOINT_FOLDER_NAME / f".stale-7.0-0-{d.name}").exists()
    make(tmp_path, 20, complete=False)  # found incomplete again in the same run
    assert ft_save_decision(tmp_path, d, 20, global_rank=0) == (FT_SAVE, False)
    assert (tmp_path / CHECKPOINT_FOLDER_NAME / f".stale-7.0-1-{d.name}").exists()


def test_rewind_save_is_flagged_so_retention_is_skipped(tmp_path):
    make(tmp_path, 30, complete=True)
    d = get_checkpoint_dir(tmp_path, 20)  # 20 was deleted by the retention, 30 exists
    assert ft_save_decision(tmp_path, d, 20, global_rank=0) == (FT_SAVE, True)
    assert ft_save_decision(tmp_path, get_checkpoint_dir(tmp_path, 40), 40, global_rank=0) == (FT_SAVE, False)


def test_other_ranks_decide_nothing_alone(tmp_path):
    d = make(tmp_path, 20, complete=True)
    assert ft_save_decision(tmp_path, d, 20, global_rank=1) == (FT_SAVE, False)  # rank 0's decision is broadcast
