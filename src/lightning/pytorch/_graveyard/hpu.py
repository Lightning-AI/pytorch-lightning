import sys
from typing import Any

import lightning.pytorch as pl


def _patch_sys_modules() -> None:
    self = sys.modules[__name__]
    sys.modules["lightning.pytorch.accelerators.hpu"] = self
    sys.modules["lightning.pytorch.strategies.hpu_parallel"] = self
    sys.modules["lightning.pytorch.strategies.single_hpu"] = self
    sys.modules["lightning.pytorch.plugins.io.hpu_plugin"] = self
    sys.modules["lightning.pytorch.plugins.precision.hpu"] = self


class HPUAccelerator:
    auto_device_count = ...
    get_parallel_devices = ...
    is_available = ...
    parse_devices = ...
    setup_device = ...
    teardown = ...

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError("The `HPUAccelerator` class has been removed. Please contact developer@lightning.ai")


class HPUParallelStrategy:
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "The `HPUParallelStrategy` class has been removed. Please contact developer@lightning.ai"
        )

    def setup(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError

    def get_device_stats(self, *_: Any, **__: Any) -> dict:
        raise NotImplementedError


class SingleHPUStrategy:
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "The `SingleHPUStrategy` class has been removed. Please contact developer@lightning.ai"
        )


class HPUCheckpointIO:
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError("The `HPUCheckpointIO` class has been removed. Please contact developer@lightning.ai")


class HPUPrecisionPlugin:
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "The `HPUPrecisionPlugin` class has been removed. Please contact developer@lightning.ai"
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
