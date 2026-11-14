from xllm.ft.node import NodeResources
from xllm.ft.sizing import USABLE_FRACTION, check_gpu, check_host

GB = 10**9


def node(shm_gb, ram_gb, gpus=8):
    return NodeResources(gpus=gpus, shm_bytes=shm_gb * GB, shm_free_bytes=shm_gb * GB, ram_bytes=ram_gb * GB)


def test_8b_on_16_gpus_fits_both_clusters():
    # 8B over 16 GPUs: 0.5e9 local params; 48 B/param * 8 GPUs = 192 GB of /dev/shm (plan section 6)
    for shm in (504, 1008):
        shm_check, ram_check = check_host(int(0.5e9), node(shm, 2 * shm), dcp_for_optimizer=False)
        assert shm_check.needed == 192 * GB and shm_check.ok
        assert ram_check.needed == 76 * int(0.5e9) * 8 and ram_check.ok


def test_dcp_optimizer_staging_adds_8_bytes():
    plain = check_host(10**9, node(1000, 2000), False)[1].needed
    dcp = check_host(10**9, node(1000, 2000), True)[1].needed
    assert dcp - plain == 8 * 10**9 * 8


def test_shortfall_is_reported():
    shm_check, _ = check_host(int(1.953e9), node(504, 1000), False)  # 1T on 512 GPUs on the first cluster
    assert not shm_check.ok
    assert shm_check.available == int(USABLE_FRACTION * 504 * GB)
    assert "SHORT by" in str(shm_check)


def test_gpu_check_adds_snapshot_buffers():
    assert check_gpu(50 * GB, 10**9, 80 * GB).needed == 74 * GB
    assert not check_gpu(60 * GB, 10**9, 80 * GB).ok


class FakeCuda:
    def __init__(self, monkeypatch, peaks, total):
        self.peaks = iter(peaks)
        monkeypatch.setattr("torch.cuda.max_memory_reserved", lambda: next(self.peaks))
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr(
            "torch.cuda.get_device_properties", lambda d: type("P", (), {"total_memory": total})()
        )


def test_gpu_peak_counts_steps_not_iterations(monkeypatch):
    from xllm.ft.sizing import GpuPeakRecorder

    FakeCuda(monkeypatch, peaks=[10 * GB] * 20, total=80 * GB)
    rec = GpuPeakRecorder(local_params=10**9, steps=3)
    rec.record(0)  # model build: not a step sample
    assert rec.peaks == []
    rec.record(1)
    rec.record(1)  # an iteration that skipped the step (overflow)
    rec.record(2)
    assert len(rec.peaks) == 2 and not rec.copies_allowed
    rec.record(3)
    assert rec.copies_allowed and rec.verdict.needed == 34 * GB
    rec.record(4)  # no-op after the verdict
    assert len(rec.peaks) == 3


def test_gpu_shortfall_keeps_copies_off(monkeypatch):
    from xllm.ft.sizing import GpuPeakRecorder

    FakeCuda(monkeypatch, peaks=[70 * GB] * 5, total=80 * GB)
    rec = GpuPeakRecorder(local_params=10**9, steps=2)
    for step in range(4):
        rec.record(step)
    assert rec.verdict is not None and not rec.verdict.ok and not rec.copies_allowed
