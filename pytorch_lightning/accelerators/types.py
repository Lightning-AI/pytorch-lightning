from enum import Enum


class BackendType(Enum):
    DP = 'dp'
    DDP = 'ddp'
    DDP2 = 'ddp2'
    DDP_SPAWN = 'ddp_spawn'
    HOROVOD = 'horovod'
    # TODO: decouple DDP & CPU
    DDP_CPU = 'ddp_cpu'


class DeviceType(Enum):
    GPU = 'gpu'
    CPU = 'cpu'
    TPU = 'tpu'
