# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from typing import Any

import pytorch_lightning as pl


def _patch_sys_modules() -> None:
    # TODO: Remove in v2.0.0
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
    # TODO: Remove in v2.0.0
    raise NotImplementedError(
        f"The `pl.plugins.{self._name}Plugin` class was deprecated in v1.6 and is no longer supported as of v1.8."
        f" Use `pl.strategies.{self._name}Strategy` instead."
    )


def _patch_plugin_classes() -> None:
    # TODO: Remove in v2.0.0
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
    # TODO: Remove in v2.0.0
    raise NotImplementedError(
        "`pl.plugins.training_type.utils.on_colab_kaggle` was deprecated in v1.6 and is no longer supported as of v1.8."
        " Use `pl.strategies.utils.on_colab_kaggle` instead."
    )


def _training_type_plugin(_: pl.Trainer) -> None:
    # TODO: Remove in v2.0.0
    raise AttributeError(
        "`Trainer.training_type_plugin` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Use `Trainer.strategy` instead."
    )


_patch_sys_modules()
_patch_plugin_classes()

# Properties
pl.Trainer.training_type_plugin = property(_training_type_plugin)
