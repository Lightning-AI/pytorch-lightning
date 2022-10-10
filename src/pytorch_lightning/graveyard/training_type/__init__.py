import sys
from typing import Any

from pytorch_lightning import Trainer

self = sys.modules[__name__]
# FIXME: this needs to add, not replace
# sys.modules["pytorch_lightning.plugins"] = self
sys.modules["pytorch_lightning.plugins.training_type"] = self
sys.modules["pytorch_lightning.plugins.training_type.ddp"] = self
sys.modules["pytorch_lightning.plugins.training_type.ddp2"] = self
sys.modules["pytorch_lightning.plugins.training_type.ddp_spawn"] = self
sys.modules["pytorch_lightning.plugins.training_type.deepspeed"] = self
sys.modules["pytorch_lightning.plugins.training_type.dp"] = self
sys.modules["pytorch_lightning.plugins.training_type.fully_sharded"] = self
sys.modules["pytorch_lightning.plugins.training_type.horovod"] = self
sys.modules["pytorch_lightning.plugins.training_type.ipu"] = self
sys.modules["pytorch_lightning.plugins.training_type.parallel"] = self
sys.modules["pytorch_lightning.plugins.training_type.sharded"] = self
sys.modules["pytorch_lightning.plugins.training_type.sharded_spawn"] = self
sys.modules["pytorch_lightning.plugins.training_type.single_device"] = self
sys.modules["pytorch_lightning.plugins.training_type.single_tpu"] = self
sys.modules["pytorch_lightning.plugins.training_type.tpu_spawn"] = self
sys.modules["pytorch_lightning.plugins.training_type.training_type_plugin"] = self


class DDPPlugin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "The `pl.plugins.training_type.ddp.DDPPlugin` was removed in v1.8. Use `pl.strategies.ddp.DDPStrategy`"
            " instead."
        )


class SingleDevicePlugin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "The `pl.plugins.training_type.single_device.SingleDevicePlugin` was  removed in v1.8. Use"
            " `pl.strategies.single_device.SingleDeviceStrategy` instead."
        )


def _training_type_plugin(_: Trainer):
    raise RuntimeError(
        "`Trainer.training_type_plugin` is deprecated in v1.6 and was removed in v1.8. Use"
        " `Trainer.strategy` instead."
    )


Trainer.training_type_plugin = property(_training_type_plugin)
