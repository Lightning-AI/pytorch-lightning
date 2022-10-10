import sys

from pytorch_lightning import Trainer

self = sys.modules[__name__]
sys.modules["pytorch_lightning.plugins.training_type"] = self
sys.modules["pytorch_lightning.plugins.training_type.single_device"] = self


class SingleDevicePlugin:
    ...


def _training_type_plugin(_):
    raise RuntimeError(
        "`Trainer.training_type_plugin` is deprecated in v1.6 and was removed in v1.8. Use"
        " `Trainer.strategy` instead."
    )


Trainer.training_type_plugin = property(_training_type_plugin)
