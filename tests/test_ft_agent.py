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
                           retries=2, release_timeout=0, nccl_timeout=3600, xid_replace="", spares=0,
                           spare_health=True, refill_wait=0, dump_dir="")
    assert agent.hang_deadline([1.0] * 19) == 0.9 * 3600  # not measured yet: just below the NCCL timeout
    assert agent.hang_deadline([1.0] * 19 + [2.0]) == 30.0  # floor
    assert agent.hang_deadline([1.0] * 19 + [40.0]) == 60.0  # (1 + m) * longest gap
    assert agent.hang_deadline([3000.0] * 20) == 0.9 * 3600  # always below the NCCL timeout
    agent.conf.hb_deadline = 30.0
    assert agent.hang_deadline([]) == 30.0


def test_startup_deadline_uses_the_last_measured_startup():
    agent = Agent.__new__(Agent)
    agent.conf = AgentConf(slot_dir="", node_deadline=10, hb_interval=0.2, hb_deadline=20, hb_margin=0.5,
                           retries=2, release_timeout=0, nccl_timeout=3600, xid_replace="", spares=0,
                           spare_health=True, refill_wait=0, dump_dir="")
    agent.startup_s = None
    assert agent.startup_deadline() == 0.9 * 3600  # first generation: not measured yet
    agent.startup_s = 30.0
    assert agent.startup_deadline() == 60.0  # floor
    agent.startup_s = 100.0
    assert agent.startup_deadline() == 150.0


class DictStore:
    def __init__(self):
        self.data = {}

    def set(self, k, v):
        self.data[k] = v

    def check(self, keys):
        return all(k in self.data for k in keys)

    def get(self, k):
        v = self.data[k]
        return v.encode() if isinstance(v, str) else v


def test_gpu_faults_restart_or_replace():
    import threading

    agent = Agent.__new__(Agent)
    agent.phys, agent.logical, agent.store = 5, 1, DictStore()
    agent.faults, agent.faults_lock = [], threading.Lock()
    agent.xids_to_replace = {79, 95}
    agent._on_gpu_fault(3, 13)  # a critical Xid that does not need a reset: restart the trainers
    agent.check_gpu_faults(gen=4)
    assert "req/4" in agent.store.data and "bad/5" not in agent.store.data
    agent._on_gpu_fault(5, 79)  # needs a reset: the node is marked bad, the leader replaces it with a spare
    agent.check_gpu_faults(gen=5)
    assert "Xid 79" in agent.store.data["bad/5"] and "req/5" in agent.store.data
    assert agent.faults == []


def test_fault_lane_without_gpus_disables_itself_but_injects():
    from xllm.ft.gpu_faults import GpuFaultLane

    seen = []
    lane = GpuFaultLane(lambda gpu, xid: seen.append((gpu, xid)))
    lane.inject(2, 48)
    assert seen == [(2, 48)]


def test_get_or_never_waits_on_a_missing_key():
    import time

    agent = Agent.__new__(Agent)
    agent.store = DictStore()
    t0 = time.time()
    assert agent.get_or("host/7", "fallback") == "fallback"
    agent.store.set("host/7", "dgx-1")
    assert agent.get_or("host/7", "fallback") == "dgx-1"
    assert time.time() - t0 < 1.0


def test_bar_list_outlives_the_control_store(tmp_path):
    import types

    import pytest

    agent = Agent.__new__(Agent)
    agent.conf, agent.store, agent.main_job = types.SimpleNamespace(dump_dir=str(tmp_path)), DictStore(), 5
    agent.store.set("host/4", "dgx-7")
    agent.bar(4, "Xid 79 on GPU 2")
    assert agent.barred() == ["dgx-7"]
    # a requeue: a new control store, seeded from the file by the new leader
    again = Agent.__new__(Agent)
    again.conf, again.store, again.main_job = agent.conf, DictStore(), 5
    assert again.barred_file_hosts() == ["dgx-7"]
    again.store.set("barred", ",".join(again.barred_file_hosts()))
    again.phys = 2
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("socket.gethostname", lambda: "dgx-7")
        assert again.park() is None  # never admitted as a spare again, health check or not: it leaves the run
    assert again.store.data["spare/2"] == "barred"


def test_a_node_that_never_sent_a_heartbeat_gets_the_startup_deadline():
    import types

    class Counters(DictStore):
        def add(self, k, v):
            self.data[k] = self.data.get(k, 0) + v
            return self.data[k]

    agent = Agent.__new__(Agent)
    agent.store, agent.hb_seen, agent.startup_s = Counters(), {}, None
    agent.conf = types.SimpleNamespace(node_deadline=10.0, hb_margin=0.5, nccl_timeout=3600)
    assert agent.lost_node([1]) is None
    agent.hb_seen[1] = (0, agent.hb_seen[1][1] - 11.0)  # 11 s silent, but it never started: still starting
    assert agent.lost_node([1]) is None
    agent.store.add("hb/1", 1)
    assert agent.lost_node([1]) is None
    agent.hb_seen[1] = (1, agent.hb_seen[1][1] - 11.0)  # it ran, then went silent past the node deadline
    assert agent.lost_node([1]) == 1



def test_holds_copies_sees_own_and_held_copies_of_this_run(tmp_path, monkeypatch):
    import pickle
    import types

    import torch

    from xllm.ft.replica import run_epoch, slot_path, write_slot

    monkeypatch.setenv("FT_EPOCH", "11:0")
    agent = Agent.__new__(Agent)
    agent.conf, agent.ntrain, agent.gpus = types.SimpleNamespace(slot_dir=str(tmp_path)), 2, 2
    assert not agent.holds_copies(1)  # a failure before the first copy: memory cannot restore anything

    def copy(kind, rank, epoch):
        meta = {"step": 5, "epoch": epoch, "rank": rank, "nbytes": 4, "training_state": {}}
        write_slot(slot_path(str(tmp_path), kind, rank, 0), torch.zeros(4, dtype=torch.uint8), pickle.dumps(meta))

    copy("own", 2, ("11", "1"))  # an earlier run's copy does not count
    assert not agent.holds_copies(1)
    copy("peer", 1, run_epoch())  # node 1 holds node 0's ranks 0-1
    assert agent.holds_copies(1)


def test_a_follower_keeps_trying_to_join_a_late_leader(monkeypatch):
    import types

    agent = Agent.__new__(Agent)
    agent.conf = types.SimpleNamespace(nccl_timeout=3600, node_deadline=10.0)
    agent.leader_addr, agent.store_port = "node0", 20000
    attempts = []

    def connect(host):
        attempts.append(host)
        if len(attempts) < 3:  # the leader's container is still starting
            raise RuntimeError("The client socket has timed out")
        return "store"

    agent._connect = connect
    assert agent._join() == "store" and attempts == [False, False, False]
