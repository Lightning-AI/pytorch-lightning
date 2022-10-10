import sys
from typing import Any

from pytorch_lightning.callbacks import ModelCheckpoint

self = sys.modules[__name__]
sys.modules["pytorch_lightning.callbacks"] = self
sys.modules["pytorch_lightning.callbacks.model_checkpoint"] = self


def _save_checkpoint(_: ModelCheckpoint, __: Any) -> None:
    raise NotImplementedError(
        f"`{ModelCheckpoint.__name__}.save_checkpoint()` was deprecated in v1.6 and is no longer supported"
        f" as of 1.8. Please use `trainer.save_checkpoint()` to manually save a checkpoint. This method will be"
        f" removed completely in v2.0."
    )


# Methods
ModelCheckpoint.save_checkpoint = _save_checkpoint
