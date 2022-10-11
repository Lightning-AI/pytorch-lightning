import sys
from typing import Any

import pytorch_lightning as pl


def _patch_sys_modules() -> None:
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


def _patch_plugin_classes() -> None:
    self = sys.modules[__name__]
    for name in (
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
        plugin_name = name + "Plugin"
        plugin_cls = type(plugin_name, (object,), {"__init__": _ttp_constructor, "_name": name})
        setattr(self, plugin_name, plugin_cls)
        # do not overwrite sys.modules as `pl.plugins` still exists. manually patch instead
        setattr(pl.plugins, plugin_name, plugin_cls)


def on_colab_kaggle() -> None:
    raise RuntimeError(
        "`pl.plugins.training_type.utils.on_colab_kaggle` was removed in v1.8."
        " Use `pl.strategies.utils.on_colab_kaggle` instead."
    )


def _training_type_plugin(_: pl.Trainer) -> None:
    raise RuntimeError("`Trainer.training_type_plugin` was removed in v1.8. Use `Trainer.strategy` instead.")


_patch_sys_modules()
_patch_plugin_classes()
pl.Trainer.training_type_plugin = property(_training_type_plugin)
