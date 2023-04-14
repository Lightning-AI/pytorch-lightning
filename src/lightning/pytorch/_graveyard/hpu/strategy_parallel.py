import sys
from typing import Any

import lightning.pytorch as pl


def _patch_sys_modules() -> None:
    self = sys.modules[__name__]
    sys.modules["lightning.pytorch.strategies.hpu_parallel"] = self


class HPUParallelStrategy(pl.strategies.DDPStrategy):
    """Strategy for distributed training on multiple HPU devices."""

    def __init__(self, *_: Any, **__: Any):
        raise NotImplementedError(
            "The `HPUParallelStrategy` class has been moved to an external package."
            " Install the extension package as `pip install lightning-habana`"
            " and import with `from lightning_habana import HPUParallelStrategy`."
            " Please see: https://github.com/Lightning-AI/lightning-Habana for more details."
        )


def _patch_classes() -> None:
    setattr(pl.accelerators, "HPUParallelStrategy", HPUParallelStrategy)


_patch_sys_modules()
_patch_classes()
