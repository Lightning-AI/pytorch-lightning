import sys
from typing import Any

import pytorch_lightning as pl


def _patch_sys_modules() -> None:
    # TODO: Remove in v2.0.0
    self = sys.modules[__name__]
    sys.modules["pytorch_lightning.accelerators.gpu"] = self


class GPUAccelerator:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "`pytorch_lightning.accelerators.gpu.GPUAccelerator` was deprecated in v1.7.0 and is no"
            " longer supported as of v1.9.0. Please use `pytorch_lightning.accelerators.CUDAAccelerator` instead"
        )


def _patch_classes() -> None:
    # TODO: Remove in v2.0.0
    setattr(pl.accelerators, "GPUAccelerator", GPUAccelerator)


_patch_sys_modules()
_patch_classes()
