import sys
import types

from xllm.ft.spare_check import nvml_healthy


def fake_nvml(monkeypatch, errors=0, pending=0, failed=0, ecc_supported=True):
    nvml = types.ModuleType("pynvml")
    nvml.NVMLError = type("NVMLError", (Exception,), {})
    nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, nvml.NVML_VOLATILE_ECC = 1, 0
    nvml.nvmlInit = lambda: None
    nvml.nvmlDeviceGetCount = lambda: 2
    nvml.nvmlDeviceGetHandleByIndex = lambda i: i

    def ecc(h, kind, counter):
        if not ecc_supported:
            raise nvml.NVMLError("not supported")
        return errors if h == 1 else 0

    nvml.nvmlDeviceGetTotalEccErrors = ecc
    nvml.nvmlDeviceGetRemappedRows = lambda h: (0, 0, pending if h == 1 else 0, failed if h == 1 else 0)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)


def test_spare_gpus_with_bad_health_counters_are_refused(monkeypatch):
    fake_nvml(monkeypatch)
    assert nvml_healthy()
    fake_nvml(monkeypatch, errors=3)
    assert not nvml_healthy()
    fake_nvml(monkeypatch, pending=1)
    assert not nvml_healthy()
    fake_nvml(monkeypatch, failed=1)
    assert not nvml_healthy()
    fake_nvml(monkeypatch, ecc_supported=False)  # ECC off: not a reason to refuse
    assert nvml_healthy()


def test_no_nvml_skips_the_counters(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import fails
    assert nvml_healthy()
