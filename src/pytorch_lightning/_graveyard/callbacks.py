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

from pytorch_lightning.callbacks import ModelCheckpoint


def _patch_sys_modules() -> None:
    # TODO: Remove in v2.0.0
    self = sys.modules[__name__]
    sys.modules["pytorch_lightning.callbacks.base"] = self


class Callback:
    # TODO: Remove in v2.0.0
    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "Importing `pytorch_lightning.callbacks.base.Callback` was deprecated in v1.7.0 and removed as of"
            " v1.9.0. Please use `from pytorch_lightning import Callback` instead"
        )


def _save_checkpoint(_: ModelCheckpoint, __: Any) -> None:
    # TODO: Remove in v2.0.0
    raise NotImplementedError(
        f"`{ModelCheckpoint.__name__}.save_checkpoint()` was deprecated in v1.6 and is no longer supported"
        f" as of 1.8. Please use `trainer.save_checkpoint()` to manually save a checkpoint."
    )


_patch_sys_modules()

# Methods
ModelCheckpoint.save_checkpoint = _save_checkpoint
