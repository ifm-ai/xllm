import multiprocessing

from xllm.ft.leader import Lease, Liveness, lease_dir


def _claim(folder, term, out):
    out.put(Lease(folder).claim(term))


def test_a_term_is_claimed_exactly_once_even_by_racing_processes(tmp_path):
    folder = lease_dir(str(tmp_path), "77", "0")
    out = multiprocessing.Queue()
    procs = [multiprocessing.Process(target=_claim, args=(folder, 1, out)) for _ in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    wins = [out.get() for _ in procs]
    assert wins.count(True) == 1
    assert Lease(folder).claim(1) is False and Lease(folder).claim(2) is True  # the next term is free


def test_leader_state_and_counter_round_trip(tmp_path):
    lease = Lease(lease_dir(str(tmp_path), "77", "1"))
    assert lease.leader() is None and lease.state() is None and lease.counter(0) is None
    assert lease.publish({"term": 0, "host": "node0", "port": 20077, "standby": 1})
    assert lease.leader() == {"term": 0, "host": "node0", "port": 20077, "standby": 1}
    assert lease.renew(0, 5) and lease.counter(0) == 5
    assert lease.save_state({"gen": 3, "map": {"0": 2}}) and lease.state() == {"gen": 3, "map": {"0": 2}}
    assert not list(lease.folder.glob(".*tmp"))  # every write is a rename of a finished file


def test_filesystem_errors_never_raise(tmp_path):
    (tmp_path / "ft_leader").write_text("not a dir")  # the folder cannot be created
    lease = Lease(lease_dir(str(tmp_path), "77", "0"))
    assert lease.claim(0) is False and lease.renew(0, 1) is False and lease.publish({}) is False
    assert lease.save_state({}) is False and lease.leader() is None and lease.counter(0) is None


def test_liveness_is_judged_by_a_changing_counter(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("xllm.ft.leader.time.monotonic", lambda: clock[0])
    alive = Liveness()
    assert alive.silent_for(1) == 0.0
    clock[0] = 130.0
    assert alive.silent_for(1) == 30.0  # the same counter: silent for 30 s
    assert alive.silent_for(2) == 0.0  # it moved
    clock[0] = 140.0
    assert alive.silent_for(None) == 10.0  # unreadable counts as silent


def test_two_threads_writing_the_leader_file_never_tear_it(tmp_path):
    import threading

    lease = Lease(tmp_path)
    stop, bad = threading.Event(), []

    def write(flag):
        while not stop.is_set():
            lease.publish({"term": 0, "host": "n0", "port": 1, "phys": 0, "standby": 1, "stepdown": flag})

    writers = [threading.Thread(target=write, args=(f,), daemon=True) for f in (False, True)]
    lease.publish({"term": 0})
    for w in writers:
        w.start()
    for _ in range(3000):
        if lease.leader() is None:
            bad.append(1)
    stop.set()
    for w in writers:
        w.join()
    assert not bad
