import sys
from typing import Any

import lightning.pytorch as pl


def _patch_sys_modules() -> None:
    self = sys.modules[__name__]
    sys.modules["lightning.plugins.precision.hpu"] = self


class HPUPrecisionPlugin(pl.plugins.precision.PrecisionPlugin):
    """Plugin that enables bfloat/half support on HPUs."""

    def __init__(self, *_: Any, **__: Any):
        raise NotImplementedError(
            "The `HPUPrecisionPlugin` class has been moved to an external package."
            " Install the extension package as `pip install lightning-habana`"
            " and import with `from lightning_habana import HPUPrecisionPlugin`."
            " Please see: https://github.com/Lightning-AI/lightning-Habana for more details."
        )


def _patch_classes() -> None:
    setattr(pl.accelerators, "HPUPrecisionPlugin", HPUPrecisionPlugin)


_patch_sys_modules()
_patch_classes()
