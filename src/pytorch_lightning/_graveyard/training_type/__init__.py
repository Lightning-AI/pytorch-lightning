import sys
from typing import Any

import pytorch_lightning as pl

self = sys.modules[__name__]
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
sys.modules["pytorch_lightning.plugins.training_type.utils"] = self


def _ttp_constructor(self: Any, *_: Any, **__: Any) -> None:
    raise RuntimeError(
        f"The `pl.plugins.{self._name}Plugin` class was removed in v1.8. Use `pl.strategies.{self._name}Strategy`"
        " instead."
    )


for _name in (
    "DDP",
    "DDP2",
    "DDPSpawn",
    "DeepSpeed",
    "DataParallel",
    "DDPFullySharded",
    "Horovod",
    "IPU",
    "Parallel",
    "DDPSharded",
    "DDPSpawnSharded",
    "SingleDevice",
    "SingleTPU",
    "TPUSpawn",
    "TrainingType",
):
    _plugin_name = _name + "Plugin"
    _plugin_cls = type(_plugin_name, (object,), {"__init__": _ttp_constructor, "_name": _name})
    setattr(self, _plugin_name, _plugin_cls)
    # do not overwrite sys.modules as `pl.plugins` still exists. manually patch instead
    setattr(pl.plugins, _plugin_name, _plugin_cls)


def on_colab_kaggle():
    raise RuntimeError(
        "`pl.plugins.training_type.utils.on_colab_kaggle` was removed in v1.8."
        " Use `pl.strategies.utils.on_colab_kaggle` instead."
    )


def _training_type_plugin(_: pl.Trainer) -> None:
    raise RuntimeError(
        "`Trainer.training_type_plugin` is deprecated in v1.6 and was removed in v1.8. Use"
        " `Trainer.strategy` instead."
    )


pl.Trainer.training_type_plugin = property(_training_type_plugin)
