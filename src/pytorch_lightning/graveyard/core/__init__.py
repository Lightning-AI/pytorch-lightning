import sys

from pytorch_lightning import LightningModule

self = sys.modules[__name__]
sys.modules["pytorch_lightning.core"] = self
sys.modules["pytorch_lightning.core.module"] = self


def _use_amp(_: LightningModule) -> None:
    raise AttributeError(
        "`LightningModule.use_amp` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.amp_backend`.",
    )


def _use_amp_setter(_: LightningModule, __: bool) -> None:
    raise AttributeError(
        "`LightningModule.use_amp` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.amp_backend`.",
    )


LightningModule.use_amp = property(fget=_use_amp, fset=_use_amp_setter)
