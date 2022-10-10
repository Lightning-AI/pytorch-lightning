import sys
from typing import Any

from pytorch_lightning import Trainer

self = sys.modules[__name__]
sys.modules["pytorch_lightning.trainer"] = self
sys.modules["pytorch_lightning.trainer.trainer"] = self


def _run_stage(_: Trainer) -> None:
    raise NotImplementedError(
        "`Trainer.run_stage` was deprecated in v1.6 and is no longer supported as of v1.8."
        " Please use `Trainer.{fit,validate,test,predict}` instead."
    )


def _call_hook(_: Trainer, *__: Any, **___: Any) -> Any:
    raise NotImplementedError(
        "`Trainer.call_hook` was deprecated in v1.6 and is no longer supported as of v1.8."
    )


Trainer.run_stage = _run_stage
Trainer.call_hook = _call_hook
