from xllm.ft.agent import Agent, AgentConf, agent_conf
from xllm.ft.progress import Progress, progress_path


def test_agent_reads_settings_from_the_trainer_arguments():
    conf = agent_conf(["train.py", "--ft.enabled", "true", "--ft.slot_dir", "/dev/shm/x", "--ft.retries", "3",
                       "--model", "llama3-7B", "--nccl_timeout", "1800"])
    assert conf.slot_dir == "/dev/shm/x" and conf.retries == 3 and conf.nccl_timeout == 1800
    assert conf.node_deadline == 10.0 and conf.hb_deadline == 0.0  # TrainerConf defaults


def test_progress_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "5")
    path = progress_path(str(tmp_path), 3)
    assert path.name == "ft_5_progress_r000003"
    agent_side = Progress(path, create=True)
    monkeypatch.setenv("FT_PROGRESS_FILE", str(path))
    trainer_side = Progress.for_trainer()
    trainer_side.step(42)
    trainer_side.pause(True)
    assert agent_side.read() == (42, True)
    Progress(path, create=True)  # a new generation resets it
    assert agent_side.read() == (0, False)


def test_hang_deadline_auto_and_configured():
    agent = Agent.__new__(Agent)
    agent.gpus = 2
    agent.conf = AgentConf(slot_dir="", node_deadline=10, hb_interval=0.2, hb_deadline=0, hb_margin=0.5,
                           retries=2, release_timeout=0, nccl_timeout=3600)
    assert agent.hang_deadline([1.0] * 19) == 0.9 * 3600  # not measured yet: just below the NCCL timeout
    assert agent.hang_deadline([1.0] * 19 + [2.0]) == 30.0  # floor
    assert agent.hang_deadline([1.0] * 19 + [40.0]) == 60.0  # (1 + m) * longest gap
    assert agent.hang_deadline([3000.0] * 20) == 0.9 * 3600  # always below the NCCL timeout
    agent.conf.hb_deadline = 30.0
    assert agent.hang_deadline([]) == 30.0


def test_startup_deadline_uses_the_last_measured_startup():
    agent = Agent.__new__(Agent)
    agent.conf = AgentConf(slot_dir="", node_deadline=10, hb_interval=0.2, hb_deadline=20, hb_margin=0.5,
                           retries=2, release_timeout=0, nccl_timeout=3600)
    agent.startup_s = None
    assert agent.startup_deadline() == 0.9 * 3600  # first generation: not measured yet
    agent.startup_s = 30.0
    assert agent.startup_deadline() == 60.0  # floor
    agent.startup_s = 100.0
    assert agent.startup_deadline() == 150.0
