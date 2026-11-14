"""
GPU-fault lane of the ft agent (NVIDIA only). A thread waits on NVML events for
the node's GPUs; a critical Xid or a GPU that became unavailable is reported to
the agent, which treats it as a crash of the node's trainers. Without NVML, or
when NVML fails, the lane disables itself and the agent's other lanes (exit
codes, heartbeats, hang deadline) keep working.
"""

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger()

WAIT_MS = 1000


class GpuFaultLane:
    def __init__(self, on_fault: Callable[[int, Optional[int]], None]):
        """on_fault(gpu, xid) is called from the lane's thread; xid is None when the GPU is lost."""
        self.on_fault = on_fault
        self.enabled = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            self.event_set = pynvml.nvmlEventSetCreate()
            wanted = pynvml.nvmlEventTypeXidCriticalError | pynvml.nvmlEventTypeGpuUnavailableError
            for handle in self.handles:
                types = pynvml.nvmlDeviceGetSupportedEventTypes(handle) & wanted
                if types:
                    pynvml.nvmlDeviceRegisterEvents(handle, types, self.event_set)
            self.enabled = True
        except Exception as e:  # not an NVIDIA node, or NVML unusable: the other lanes still work
            logger.warning(f"ft: GPU-fault lane disabled ({type(e).__name__}: {e})")
            return
        threading.Thread(target=self._watch, daemon=True).start()

    def _gpu_index(self, handle) -> int:
        for i, h in enumerate(self.handles):
            if self.nvml.nvmlDeviceGetIndex(h) == self.nvml.nvmlDeviceGetIndex(handle):
                return i
        return -1

    def _watch(self):
        nvml = self.nvml
        while True:
            try:
                data = nvml.nvmlEventSetWait(self.event_set, WAIT_MS)
            except nvml.NVMLError as e:
                if e.value == nvml.NVML_ERROR_TIMEOUT:
                    continue
                if e.value == nvml.NVML_ERROR_GPU_IS_LOST:
                    self.on_fault(-1, None)  # after GPU lost every NVML call fails: report once and stop
                else:
                    logger.warning(f"ft: GPU-fault lane stopped ({e})")
                return
            gpu = self._gpu_index(data.device)
            if data.eventType & nvml.nvmlEventTypeGpuUnavailableError:
                self.on_fault(gpu, None)
            elif data.eventType & nvml.nvmlEventTypeXidCriticalError:
                self.on_fault(gpu, int(data.eventData))

    def inject(self, gpu: int, xid: Optional[int]):
        """Test injection: delivered through the same callback as a real event."""
        self.on_fault(gpu, xid)
