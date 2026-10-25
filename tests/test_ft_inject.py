import datetime
import json

import pytest
from torch.distributed import TCPStore

from tests.ft_inject import injection_from_args, parse_args
from xllm.ft.inject import COUNT_KEY, KINDS, REQUEST_KEY, Injection, InjectionReader, publish


class StubStore:
    """The subset of the TCPStore API the injector and the agent use."""

    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = value.encode() if isinstance(value, str) else value

    def get(self, key):
        return self.data[key]

    def add(self, key, amount):
        value = int(self.data.get(key, b"0")) + amount
        self.data[key] = str(value).encode()
        return value

    def check(self, keys):
        return all(k in self.data for k in keys)


def test_every_flag_maps_to_one_kind():
    flags = {
        "kill_rank": ["--kill-rank", "3"],
        "kill_process": ["--kill-process", "1"],
        "kill_node": ["--kill-node", "1"],
        "hang_rank": ["--hang-rank", "3"],
        "torn_write": ["--torn-write", "3"],
        "bitflip": ["--bitflip", "3"],
        "xid": ["--xid", "1", "79"],
        "gpu_lost": ["--gpu-lost", "1"],
        "fail_spare_health": ["--fail-spare-health", "1"],
        "dump_kill": ["--dump-kill", "1"],
        "lease_cut": ["--lease-cut", "0"],
    }
    assert set(flags) == set(KINDS)
    for kind, flag in flags.items():
        injection = injection_from_args(parse_args(["--store", "h:1", "--at-step", "7"] + flag))
        assert injection.kind == kind
        assert injection.step == 7
        assert getattr(injection, KINDS[kind]) is not None
    assert injection_from_args(parse_args(["--store", "h:1", "--xid", "1", "79"])).xid == 79


def test_one_flag_only():
    with pytest.raises(SystemExit):
        parse_args(["--store", "h:1", "--kill-rank", "1", "--kill-node", "1"])


def test_invalid_requests_are_refused():
    with pytest.raises(AssertionError):
        Injection("reboot", node=0)
    with pytest.raises(AssertionError):
        Injection("kill_rank", node=0)
    with pytest.raises(AssertionError):
        Injection("xid", node=0)


def test_reader_gets_only_its_targets_once_and_in_order():
    store = StubStore()
    publish(store, Injection("kill_rank", rank=9))
    publish(store, Injection("kill_node", node=0))
    publish(store, Injection("hang_rank", rank=2, step=5))
    publish(store, Injection("xid", node=1, xid=79))
    node1 = InjectionReader(node=1, ranks=list(range(8, 16)))
    got = node1.poll(store)
    assert [i.kind for i in got] == ["kill_rank", "xid"]
    assert node1.due(step=0) == got
    assert node1.poll(store) == []
    node0 = InjectionReader(node=0, ranks=list(range(8)))
    node0.poll(store)
    assert [i.kind for i in node0.due(step=0)] == ["kill_node"]
    assert node0.due(step=4) == []
    assert [i.kind for i in node0.due(step=5)] == ["hang_rank"]


def test_reader_waits_for_a_request_still_being_written():
    store = StubStore()
    store.add(COUNT_KEY, 1)
    reader = InjectionReader(node=0, ranks=[0])
    assert reader.poll(store) == []
    store.set(REQUEST_KEY.format(0), Injection("kill_rank", rank=0).to_json())
    assert [i.kind for i in reader.poll(store)] == ["kill_rank"]


def test_reader_skips_a_request_whose_publisher_died():
    store = StubStore()
    store.add(COUNT_KEY, 1)  # the publisher died after bumping the counter
    publish(store, Injection("kill_rank", rank=0))
    reader = InjectionReader(node=0, ranks=[0])
    assert reader.poll(store, now=100.0) == []
    assert reader.poll(store, now=129.0) == []
    assert [i.kind for i in reader.poll(store, now=131.0)] == ["kill_rank"]


def test_reader_starts_over_when_the_store_restarts():
    store = StubStore()
    publish(store, Injection("kill_rank", rank=0))
    reader = InjectionReader(node=0, ranks=[0])
    reader.poll(store)
    reader.due(step=0)
    store.data.clear()  # a restarted store
    publish(store, Injection("hang_rank", rank=0))
    assert [i.kind for i in reader.poll(store)] == ["hang_rank"]


def test_an_injection_is_carried_out_once_per_job():
    store = StubStore()
    publish(store, Injection("kill_node", node=1))
    first = InjectionReader(node=1, ranks=[8])
    first.poll(store)
    assert [i.kind for i in first.due(step=0, store=store)] == ["kill_node"]
    replacement = InjectionReader(node=1, ranks=[8])  # the spare that took over node 1
    replacement.poll(store)
    assert replacement.due(step=0, store=store) == []


def test_real_tcpstore_round_trip():
    store = TCPStore("127.0.0.1", 0, is_master=True, timeout=datetime.timedelta(seconds=10))
    client = TCPStore("127.0.0.1", store.port, is_master=False, timeout=datetime.timedelta(seconds=10))
    assert publish(client, Injection("torn_write", rank=1, step=3)) == 0
    reader = InjectionReader(node=0, ranks=[0, 1])
    (got,) = reader.poll(store)
    assert json.loads(got.to_json()) == json.loads(Injection("torn_write", rank=1, step=3).to_json())
