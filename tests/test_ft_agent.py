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
                           spare_health=True, refill_wait=0, dump_dir="", dump_enabled=False,
                           dump_keep=2, ft_dump_dir=None)
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
                           spare_health=True, refill_wait=0, dump_dir="", dump_enabled=False,
                           dump_keep=2, ft_dump_dir=None)
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


class FakeStore(DictStore):
    def add(self, k, v):
        self.data[k] = int(self.data.get(k, 0)) + v
        return self.data[k]


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
    agent.store, agent.hb_seen, agent.startup_s, agent.known_lost = Counters(), {}, None, set()
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


def test_child_launcher_keeps_unbuffered_output_and_runs_the_script(tmp_path):
    import os
    import subprocess
    import sys

    script = tmp_path / "train.py"
    script.write_text("import sys\nprint(sys.stdout.write_through, sys.argv[1:])\n")
    from pathlib import Path

    repo = str(Path(__file__).resolve().parent.parent)
    path = repo + os.pathsep + os.environ.get("PYTHONPATH", "")
    env = dict(os.environ, FT_AGENT_PID=str(os.getpid()), PYTHONPATH=path)
    out = subprocess.run([sys.executable, "-u", "-m", "xllm.ft.child", str(script), "--a", "1"], env=env,
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == "True ['--a', '1']"  # -u kept across the exec, arguments passed as given
    env.pop("FT_AGENT_PID")
    assert subprocess.run([sys.executable, "-m", "xllm.ft.child", str(script)], env=env,
                          capture_output=True, text=True).returncode == 0  # by hand: works, with a warning


def test_hang_deadline_must_fire_before_nccl_timeout():
    import pytest

    with pytest.raises(AssertionError):
        agent_conf(["train.py", "--ft.hb_deadline", "4000", "--nccl_timeout", "3600"])


def test_a_follower_keeps_trying_to_join_a_late_leader(tmp_path, monkeypatch):
    import types

    agent = Agent.__new__(Agent)
    agent.conf = types.SimpleNamespace(nccl_timeout=3600, node_deadline=10.0, dump_dir=str(tmp_path))
    agent.leader_addr, agent.store_port, agent.port = "node0", 20000, 20000
    agent.main_job, agent.term, agent.newer_term = 9, 0, 0
    attempts = []

    def connect(host):
        attempts.append(host)
        if len(attempts) < 3:  # the leader's container is still starting
            raise RuntimeError("The client socket has timed out")
        return "store"

    agent._connect = connect
    monkeypatch.setattr("xllm.ft.agent.time.sleep", lambda s: None)
    assert agent._join() == "store" and attempts == [False, False, False]


def test_takeover_order_standby_first_then_the_others():
    from xllm.ft.agent import takeover_wait

    leader, state = {"term": 0, "phys": 0, "standby": 2}, {"map": {"0": 0, "1": 1, "2": 2}}
    assert takeover_wait(leader, state, 2, 60.0) == 90.0  # the designated standby
    assert takeover_wait(leader, state, 1, 60.0) == 150.0  # next in line, a lease_ttl later
    assert takeover_wait(leader, state, 0, 60.0) == float("inf")  # the old leader itself
    assert takeover_wait(leader, state, 5, 60.0) == float("inf")  # a spare (not a training node)
    stepdown = dict(leader, stepdown=True)  # the leader handed over on purpose: the standby goes at once
    assert takeover_wait(stepdown, state, 2, 60.0) == 0.0 and takeover_wait(stepdown, state, 1, 60.0) == 150.0


def leader_agent(tmp_path, monkeypatch, phys=1):
    """A follower (physical node phys) of a 2-node run whose leader (node 0) published term 0."""
    import threading
    import types

    from xllm.ft.leader import Lease

    monkeypatch.setenv("FT_EPOCH", "9:0")
    agent = Agent.__new__(Agent)
    agent.conf = types.SimpleNamespace(lease_ttl=60.0, lease_renew=6.0, hb_interval=0.01, node_deadline=10.0,
                                       hb_margin=0.5, nccl_timeout=3600, refill_wait=0.0, dump_dir=str(tmp_path),
                                       slot_dir=str(tmp_path))
    agent.phys, agent.ntrain, agent.gpus, agent.is_leader, agent.term, agent.newer_term = phys, 2, 8, False, 0, 0
    agent.store, agent.store_epoch, agent.known_lost, agent.takeover = FakeStore(), 0, set(), None
    agent.teardown_s = None
    agent.procs, agent.spare_ready, agent.hb_seen, agent.store_port = [], False, {}, 20000
    agent.renewed_at = __import__("time").monotonic()
    agent.lease, agent.stopping, agent.finished_locally = Lease(tmp_path / "lease"), threading.Event(), False
    agent.request_refill = lambda: None
    agent.bad_reason, agent.lease_lock = "", threading.Lock()
    agent.new_failures, agent.warm_gen = 0, -1
    return agent


def test_gen_is_fenced_by_a_newer_term_or_a_moved_store(tmp_path, monkeypatch):
    import pytest

    from xllm.ft.agent import LeaderLost

    agent = leader_agent(tmp_path, monkeypatch)
    agent.store.set("gen", "3")
    assert agent.gen() == 3
    agent.newer_term = 1  # the standby thread read term 1 on the shared FS
    with pytest.raises(LeaderLost):
        agent.gen()
    agent.newer_term = 0
    agent.store.set("moved", "1")  # the old leader stepped down and says so in its own store
    with pytest.raises(LeaderLost):
        agent.gen()
    agent.store.data.pop("moved")
    agent.takeover = {"term": 1}  # this agent won term 1 while its leader is alive but cut off
    with pytest.raises(LeaderLost):
        agent.gen()


def test_a_follower_rejoins_the_leader_of_the_next_term(tmp_path, monkeypatch):
    agent = leader_agent(tmp_path, monkeypatch)
    new_store = FakeStore()
    agent._connect = lambda host: new_store
    agent.mapped = lambda: {0: 2, 1: 1}
    agent.lease.publish({"term": 1, "host": "node1", "port": 22001, "phys": 1, "standby": 2})
    assert agent.rejoin() is True
    assert agent.store is new_store and agent.term == 1 and (agent.leader_addr, agent.port) == ("node1", 22001)
    assert new_store.data["reg/1/1"] == "1" and agent.store_epoch == 1  # registered; the side threads reconnect


def test_the_new_leader_replaces_the_old_leaders_node(tmp_path, monkeypatch):
    import xllm.ft.agent as agent_mod

    monkeypatch.setattr(agent_mod, "REFILL_WAIT_FLOOR_S", 0.0)
    agent = leader_agent(tmp_path, monkeypatch)
    agent._start_lease = lambda term: None
    store = FakeStore()
    for k, v in {"map/0": "0", "map/1": "1", "gen": "5", "next_phys": 3, "spare/2": "ready", "host/2": "dgx-c",
                 "reg/1/2": "1"}.items():
        store.set(k, v)
    agent.takeover = {"term": 1, "store": store, "port": 22001, "old": 0, "state": {"map": {"0": 0, "1": 1}}}
    assert agent._take_over() is True
    data = store.data
    assert agent.is_leader and agent.term == 1 and 0 in agent.known_lost
    assert data["map/0"] == "2" and data["spare/2"] == "used" and data["gen"] == "6" and data["mode/6"] == "memory"
    assert data["spare/0"] == "lost" and data["reg/1/1"] == "1"


def test_a_leader_lost_during_the_finish_ends_the_run_without_training(tmp_path, monkeypatch):
    agent = leader_agent(tmp_path, monkeypatch)
    agent.finished_locally = True
    (tmp_path / "ft_finished.9.0").write_text("finished\n")  # the old leader wrote it before it died
    agent.wait_finished()  # returns at once, though the store never says "finished"
    agent._start_lease = lambda term: None
    store = FakeStore()
    for k, v in {"map/0": "0", "map/1": "1", "gen": "5", "reg/1/1": "1"}.items():
        store.set(k, v)
    agent.takeover = {"term": 1, "store": store, "port": 22001, "old": 0, "state": {"map": {"0": 0, "1": 1}}}
    assert agent._take_over() is True and store.data["gen"] == "5"  # no restart, no spare taken


def test_a_finished_agent_tells_the_new_leader_the_run_is_over(tmp_path, monkeypatch):
    agent = leader_agent(tmp_path, monkeypatch, phys=2)  # this one was a poll behind: not finished itself
    agent.ntrain = 3
    agent._start_lease = lambda term: None
    store = FakeStore()
    for k, v in {"map/0": "0", "map/1": "1", "map/2": "2", "gen": "5", "reg/1/1": "1", "fin/1/1": "1"}.items():
        store.set(k, v)
    state = {"map": {"0": 0, "1": 1, "2": 2}}
    agent.takeover = {"term": 1, "store": store, "port": 22001, "old": 0, "state": state}
    assert agent._take_over() is True and agent.finished_locally and store.data["gen"] == "5"


def test_the_marker_is_written_past_the_fence(tmp_path, monkeypatch):
    agent = leader_agent(tmp_path, monkeypatch)
    agent.store.set("gen", "7")
    agent.newer_term = 2  # a third leader appeared during the finish: gen() would raise
    agent.write_finished_marker()
    assert (tmp_path / "ft_finished.9.0").read_text() == "finished at generation 7\n"


def test_a_node_that_registers_late_is_not_lost_for_ever(tmp_path, monkeypatch):
    agent = leader_agent(tmp_path, monkeypatch)
    agent.term, agent.known_lost = 1, {2}
    assert agent.lost_node([2]) == 2
    agent.store.set("reg/1/2", "1")  # it was in a long GPU release when the barrier ended
    assert agent.lost_node([2]) is None


def test_a_leader_whose_node_must_go_hands_over_to_the_standby(tmp_path, monkeypatch):
    import pytest

    from xllm.ft.agent import LeaderLost

    agent = leader_agent(tmp_path, monkeypatch, phys=0)
    agent.is_leader, agent.retries_used, agent.failed_at = True, 0, 0
    for k, v in {"map/0": "0", "map/1": "1", "gen": "2", "host/0": "n0", "host/1": "n1", "barred": "n0"}.items():
        agent.store.set(k, v)
    agent.store.add("next_phys", 2)
    agent.lease.publish({"term": 0, "host": "n0", "port": 20000, "phys": 0, "standby": 1})
    with pytest.raises(LeaderLost):
        agent.replace(2, 0, 0, "Xid 79 on GPU 3", "memory")
    assert agent.lease.leader()["stepdown"] and agent.store.get("moved") == b"stepdown"
    assert agent.lease.state()["barred"] == ["n0"] and agent.lease.state()["gen"] == 2
    assert not agent.store.check(["end"])


def test_a_leader_without_a_standby_ends_the_step(tmp_path, monkeypatch):
    agent = leader_agent(tmp_path, monkeypatch, phys=0)
    agent.is_leader = True
    agent.lease.publish({"term": 0, "host": "n0", "port": 20000, "phys": 0, "standby": -1})
    assert agent.replace(2, 0, 0, "Xid 79 on GPU 3", "memory") == "end" and agent.store.check(["end"])


def test_a_node_that_must_be_replaced_tells_the_new_leader_again(tmp_path, monkeypatch):
    agent = leader_agent(tmp_path, monkeypatch)
    agent.term, agent.bad_reason, agent.spare_ready = 1, "Xid 79 on GPU 3", False
    agent.store.set("map/0", "0")
    agent.store.set("map/1", "1")
    agent.reannounce()  # in the new leader's store
    assert agent.store.get("bad/1") == b"Xid 79 on GPU 3" and agent.store.check(["reg/1/1"])


def test_an_agent_that_starts_late_joins_the_leader_of_the_newest_term(tmp_path, monkeypatch):
    from xllm.ft.leader import Lease, lease_dir

    agent = leader_agent(tmp_path, monkeypatch)
    agent.leader_addr, agent.port, agent.main_job = "n0", 20000, 9
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    Lease(lease_dir(str(tmp_path), "9", "0")).publish({"term": 1, "host": "n1", "port": 22001, "phys": 1, "standby": 2})
    tried = []

    def connect(host):
        tried.append((agent.leader_addr, agent.port))
        if agent.port == 20000:  # the first leader stepped down and its store is closed
            raise ConnectionRefusedError("closed")
        return "store of term 1"

    agent._connect = connect
    monkeypatch.setattr("xllm.ft.agent.time.sleep", lambda s: None)
    assert agent._join() == "store of term 1" and agent.term == 1 and tried == [("n0", 20000), ("n1", 22001)]


def test_a_leader_cut_off_from_the_lease_freezes_and_leaves_once_its_agents_left(tmp_path, monkeypatch):
    import time

    import pytest

    from xllm.ft.agent import LeaderLost

    agent = leader_agent(tmp_path, monkeypatch, phys=0)
    agent.is_leader, agent.conf.node_deadline = True, 0.05
    agent.store.set("map/0", "0")
    agent.store.set("map/1", "1")
    agent.renewed_at = time.monotonic() - agent.conf.lease_ttl  # no renewal for a whole lease (or one blocking)
    assert agent.frozen and agent.lead(3) is None  # frozen: no decision
    for _ in range(3):  # node 1 still heartbeats here (a shared-FS outage for everyone): stay, frozen
        agent.store.add("hb/1", 1)
        time.sleep(0.03)
        assert agent.lead(3) is None
    time.sleep(0.1)  # node 1 went quiet: it follows a newer leader
    with pytest.raises(LeaderLost):
        agent.lead(3)
    assert agent.store.get("moved") == b"abandoned"


def test_a_leader_that_ends_the_step_is_never_taken_over(tmp_path, monkeypatch):
    from xllm.ft.agent import EXIT_REQUEUE, takeover_wait

    leader = leader_agent(tmp_path, monkeypatch, phys=0)
    leader.is_leader = True
    leader.lease.publish({"term": 0, "host": "n0", "port": 20000, "phys": 0, "standby": 1})
    leader.mark_ended(EXIT_REQUEUE)  # retries used up, no spare, ...: the job requeues from disk
    ended = leader.lease.leader()
    assert ended["ended"] and takeover_wait(ended, {"map": {"0": 0, "1": 1}}, 1, 60.0) == float("inf")
    follower = leader_agent(tmp_path, monkeypatch, phys=1)
    follower.lease, follower.kill = leader.lease, lambda: True
    monkeypatch.setattr("xllm.ft.agent.time.sleep", lambda s: None)
    assert follower.rejoin() is False  # it stops too, instead of waiting for a takeover


def test_a_refill_numbered_after_the_last_journal_save_keeps_its_number_unique(tmp_path, monkeypatch):
    agent = leader_agent(tmp_path, monkeypatch, phys=5)
    agent.store.add("next_phys", 4)  # the new leader's store, seeded from a journal that predates node 5
    agent.reannounce()
    assert int(agent.store.add("next_phys", 0)) == 6


def test_a_spare_checked_before_a_takeover_is_not_checked_again(tmp_path, monkeypatch):
    import pytest

    agent = leader_agent(tmp_path, monkeypatch, phys=2)
    agent.conf.spare_health, agent.spare_ready, agent.state = True, True, lambda *a: None
    agent.store.set("gen", "1")
    agent.store.set("finished", "1")  # ends park() at once
    agent.wait_finished = lambda: None
    monkeypatch.setattr("xllm.ft.agent.run_spare_check", lambda *a: pytest.fail("checked again"))
    agent.barred = lambda: []
    assert agent.park() is None and agent.store.get("spare/2") == b"ready"


def test_a_closed_store_port_fails_fast_instead_of_waiting_the_store_timeout(tmp_path, monkeypatch):
    import socket
    import time

    agent = leader_agent(tmp_path, monkeypatch)
    with socket.socket() as s:  # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        agent.leader_addr, agent.port = "127.0.0.1", s.getsockname()[1]
    start = time.monotonic()
    try:
        agent._connect(False)
        raise AssertionError("connected to a closed port")
    except OSError:
        assert time.monotonic() - start < 5


def test_a_term_claimed_but_never_led_is_skipped(tmp_path, monkeypatch):
    import threading
    import time

    agent = leader_agent(tmp_path, monkeypatch, phys=1)
    agent.conf.lease_ttl, agent.conf.lease_renew = 0.2, 0.05
    agent.lease.publish({"term": 0, "host": "n0", "port": 20000, "phys": 0, "standby": 1})
    assert agent.lease.claim(1)  # its winner died before it opened a store or renewed once
    agent._become_leader = lambda term, old, state: setattr(agent, "takeover", {"term": term})
    watcher = threading.Thread(target=agent._watch_leader, daemon=True)
    watcher.start()
    deadline = time.monotonic() + 20
    while agent.takeover is None and time.monotonic() < deadline:
        time.sleep(0.05)
    agent.stopping.set()
    assert agent.takeover == {"term": 2} and agent.lease.leader()["term"] == 1


def test_the_lease_thread_stops_at_a_step_down_and_never_publishes_over_it(tmp_path, monkeypatch):
    import threading
    import time

    import pytest

    from xllm.ft.agent import LeaderLost

    agent = leader_agent(tmp_path, monkeypatch, phys=0)
    agent.is_leader, agent.retries_used, agent.failed_at, agent.leader_addr, agent.port = True, 0, 0, "n0", 20000
    agent.conf.lease_renew = 0.01
    for k, v in {"map/0": "0", "map/1": "1", "gen": "2", "host/0": "n0", "host/1": "n1"}.items():
        agent.store.set(k, v)
    agent.store.add("next_phys", 2)
    agent._connect = lambda host: agent.store
    lease_thread = threading.Thread(target=agent._lead_lease, args=(0,), daemon=True)
    lease_thread.start()
    deadline = time.monotonic() + 10
    while (agent.lease.leader() or {}).get("standby") != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(LeaderLost):
        agent.step_down("Xid 79 on GPU 3")
    lease_thread.join(5)
    assert not lease_thread.is_alive() and agent.lease.leader()["stepdown"]


def test_a_spare_mapped_into_the_generation_being_started_starts_at_once(tmp_path, monkeypatch):
    import threading

    agent = leader_agent(tmp_path, monkeypatch, phys=4)
    agent.conf.spare_health, agent.state = False, lambda *a: None
    agent.ntrain, agent.barred = 3, lambda: []
    for k, v in {"gen": "1", "map/0": "0", "map/1": "3", "map/2": "2"}.items():
        agent.store.set(k, v)
    result = []
    parked = threading.Thread(target=lambda: result.append(agent.park()), daemon=True)
    parked.start()
    __import__("time").sleep(0.2)
    agent.store.set("map/2", "4")  # a second node lost while generation 1 is being started: no gen change
    parked.join(5)
    assert result == [1]


def test_new_nodes_failing_their_first_step_in_a_row_stop_being_barred(tmp_path, monkeypatch):
    from xllm.ft.agent import NEW_NODE_FAILURES

    agent = leader_agent(tmp_path, monkeypatch, phys=0)
    agent.is_leader, agent.retries_used, agent.failed_at, agent.new_failures = True, 0, 0, 0
    calls = []
    agent.lost_node = lambda nodes: None
    agent.spares = lambda: []
    agent.bar = lambda phys, reason: calls.append("bar")
    agent.replace = lambda gen, lg, phys, reason, mode: calls.append("replace") or "restart"
    agent.account = lambda gen, reason: calls.append("account") or None
    for k, v in {"map/0": "0", "map/1": "5", "req/3": "logical node 1: exit 1", "reqnode/3": "1", "new/3/1": "1"}.items():
        agent.store.set(k, v)
    for _ in range(NEW_NODE_FAILURES):
        assert agent.lead(3) == "restart"
    assert agent.lead(3) == "end"  # one more: a cause other than the nodes, so the retries (here used up) decide
    assert calls == ["bar", "replace"] * NEW_NODE_FAILURES + ["account"]
