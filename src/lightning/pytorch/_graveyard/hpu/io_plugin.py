import sys
from typing import Any

import lightning.pytorch as pl


def _patch_sys_modules() -> None:
    self = sys.modules[__name__]
    sys.modules["lightning.plugins.io.hpu_plugin"] = self


class HPUCheckpointIO(pl.plugins.io.TorchCheckpointIO):
    """CheckpointIO to save checkpoints for HPU training strategies."""

    def __init__(self, *_: Any, **__: Any):
        raise NotImplementedError(
            "The `HPUCheckpointIO` class has been moved to an external package."
            " Install the extension package as `pip install lightning-habana`"
            " and import with `from lightning_habana import HPUCheckpointIO`."
            " Please see: https://github.com/Lightning-AI/lightning-Habana for more details."
        )


def _patch_classes() -> None:
    setattr(pl.accelerators, "HPUCheckpointIO", HPUCheckpointIO)


_patch_sys_modules()
_patch_classes()
