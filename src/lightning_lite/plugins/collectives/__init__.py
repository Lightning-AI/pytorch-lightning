from lightning_lite.plugins.collectives.collective import Collective
from lightning_lite.plugins.collectives.deepspeed import DeepSpeedCollective
from lightning_lite.plugins.collectives.single_device import SingleDeviceCollective
from lightning_lite.plugins.collectives.torch_collective import TorchCollective
from lightning_lite.plugins.collectives.xla import XLACollective

__all__ = [
    "Collective",
    "DeepSpeedCollective",
    "TorchCollective",
    "SingleDeviceCollective",
    "XLACollective",
]
