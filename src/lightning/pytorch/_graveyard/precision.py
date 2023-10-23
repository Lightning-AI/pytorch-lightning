import sys

import lightning.pytorch as pl
from lightning.pytorch.plugins.precision import PrecisionPlugin


def _patch_sys_modules() -> None:
    sys.modules["lightning.pytorch.plugins.precision.precision_plugin"] = sys.modules[
        "lightning.pytorch.plugins.precision.precision"
    ]


def _patch_classes() -> None:
    setattr(pl.plugins.precision, "PrecisionPlugin", PrecisionPlugin)
    setattr(pl.plugins, "PrecisionPlugin", PrecisionPlugin)


_patch_sys_modules()
_patch_classes()
