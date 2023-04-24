import sys
from typing import Any


def _patch_sys_modules() -> None:
    self = sys.modules[__name__]
    sys.modules["lightning.pytorch.accelerators.ipu"] = self
    sys.modules["lightning.pytorch.strategies.ipu"] = self
    sys.modules["lightning.pytorch.plugins.precision.ipu"] = self


class IPUAccelerator:

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "The `IPUAccelerator` class has been moved to an external package."
            " Install the extension package as `pip install lightning-graphcore`"
            " and import with `from lightning_graphcore import IPUAccelerator`."
            " Please see: https://github.com/Lightning-AI/lightning-Graphcore for more details."
        )


class IPUStrategy:
    """Strategy for training on single HPU device."""

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "The `IPUStrategy` class has been moved to an external package."
            " Install the extension package as `pip install lightning-graphcore`"
            " and import with `from lightning_graphcore import IPUStrategy`."
            " Please see: https://github.com/Lightning-AI/lightning-Graphcore for more details."
        )


class IPUPrecisionPlugin:
    """Plugin that enables bfloat/half support on HPUs."""

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "The `IPUPrecisionPlugin` class has been moved to an external package."
            " Install the extension package as `pip install lightning-graphcore`"
            " and import with `from lightning_graphcore import IPUPrecisionPlugin`."
            " Please see: https://github.com/Lightning-AI/lightning-Graphcore for more details."
        )


def _patch_classes() -> None:
    setattr(pl.accelerators, "HPUAccelerator", HPUAccelerator)
    setattr(pl.strategies, "HPUParallelStrategy", HPUParallelStrategy)
    setattr(pl.strategies, "SingleHPUStrategy", SingleHPUStrategy)
    setattr(pl.plugins, "HPUCheckpointIO", HPUCheckpointIO)
    setattr(pl.plugins.io, "HPUCheckpointIO", HPUCheckpointIO)
    setattr(pl.plugins, "HPUPrecisionPlugin", HPUPrecisionPlugin)
    setattr(pl.plugins.precision, "HPUPrecisionPlugin", HPUPrecisionPlugin)


_patch_sys_modules()
_patch_classes()
