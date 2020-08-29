from pytorch_lightning.callbacks.gpu_stats_monitor import GPUStatsMonitor
from pytorch_lightning.utilities import rank_zero_warn


class GpuUsageLogger(GPUStatsMonitor):
    def __init__(self, *args, **kwargs):
        rank_zero_warn("`GpuUsageLogger is now `GPUStatsMonitor`"
                       " and it will be removed in v0.10.0", DeprecationWarning)
        super().__init__(*args, **kwargs)
