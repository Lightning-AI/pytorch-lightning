from lightning_lite.plugins.collectives.collective import Collective
from lightning_lite.plugins.collectives.deepspeed_collective import DeepSpeedCollective
from lightning_lite.plugins.collectives.single_device_collective import SingleDeviceCollective
from lightning_lite.plugins.collectives.torch_collective import TorchCollective
from lightning_lite.plugins.collectives.xla_collective import XLACollective

__all__ = [
    "Collective",
    "DeepSpeedCollective",
    "TorchCollective",
    "SingleDeviceCollective",
    "XLACollective",
]
