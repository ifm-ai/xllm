from typing import Dict, Tuple
from dataclasses import dataclass, asdict
import logging
import pynvml
import traceback

import torch
from xllm.utils import aggregate_dict

logger = logging.getLogger()


@dataclass
class Timings:
    starting: float = 0.0
    data_loading: float = 0.0
    forward_backward: float = 0.0
    grad_clip: float = 0.0
    optimize: float = 0.0
    logging: float = 0.0
    checkpointing: float = 0.0
    eval: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def load_state_dict(self, state_dict):
        missing_keys = []
        new_state = asdict(self)
        keys = list(new_state.keys())
        for key in keys:
            if key in state_dict:
                setattr(self, key, state_dict.pop(key))
            else:
                missing_keys.append(key)

        unknown_keys = list(state_dict.keys())

        if len(missing_keys) > 0:
            logger.warning(f"Missing keys when reloading Timings: {missing_keys}")
        if len(unknown_keys) > 0:
            logger.warning(f"Unknown keys when reloading Timings: {unknown_keys}")

        logger.info(f"Reloaded timings: {asdict(self)}")

    def get_stats(self) -> Dict[str, float]:
        dict_ = self.to_dict()
        total = sum(dict_.values())
        stats = {}
        for key, value in dict_.items():
            stats[f"cum/{key}"] = float(value)
            stats[f"ratio/{key}"] = float(value) / total
        return stats


class GPUMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.device = torch.cuda.current_device()
        assert self.device >= 0
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)

    def _collect_nvml_stats(self) -> Tuple[float, float, float, float]:
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            memory_used = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1024**3  # GB
            temperature = pynvml.nvmlDeviceGetTemperature(
                self.handle, sensor=pynvml.NVML_TEMPERATURE_GPU
            )
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
            return utilization, memory_used, temperature, power
        except Exception as e:
            logger.error(
                "NVML stats collection failed with the exception and will not be retried later. "
            )
            logger.error(traceback.format_exc())
            raise e

    def get_stats(self) -> Dict[str, float]:
        utilization, memory_used, temperature, power = self._collect_nvml_stats()
        mem_stats = torch.cuda.memory_stats()
        gpu_stats = {
            "gpu_usage": utilization,
            "used_memory_gb": memory_used,
            "power": power,
            "temperature": temperature,
            "active_gb": mem_stats["active_bytes.all.peak"] / 1024**3,
            "allocated_gb": mem_stats["allocated_bytes.all.peak"] / 1024**3,
            "reserved_gb": mem_stats["reserved_bytes.all.peak"] / 1024**3,
            "num_alloc_retries": mem_stats["num_alloc_retries"],
        }
        return gpu_stats

    @staticmethod
    def aggregate_gpu_info(gpu_info):
        agg = aggregate_dict(gpu_info, ops=['MAX'])
        return {f"gpus/{k}": v for k, v in agg.items()}
