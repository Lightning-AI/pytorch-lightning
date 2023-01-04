from lightning_fabric.plugins.collectives.collective import Collective
from lightning_fabric.plugins.collectives.single_device import SingleDeviceCollective
from lightning_fabric.plugins.collectives.torch_collective import TorchCollective

__all__ = [
    "Collective",
    "TorchCollective",
    "SingleDeviceCollective",
]
