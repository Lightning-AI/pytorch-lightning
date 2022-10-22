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
from typing import Any

from pytorch_lightning import LightningDataModule, LightningModule


def _on_save_checkpoint(_: LightningDataModule, __: Any) -> None:
    # TODO: Remove in v2.0.0
    raise NotImplementedError(
        "`LightningDataModule.on_save_checkpoint` was deprecated in v1.6 and is no longer supported as of v1.8."
        " Use `state_dict` instead."
    )


def _on_load_checkpoint(_: LightningDataModule, __: Any) -> None:
    # TODO: Remove in v2.0.0
    raise NotImplementedError(
        "`LightningDataModule.on_load_checkpoint` was deprecated in v1.6 and is no longer supported as of v1.8."
        " Use `load_state_dict` instead."
    )


def _use_amp(_: LightningModule) -> None:
    # Remove in v2.0.0 and the skip in `__jit_unused_properties__`
    if not LightningModule._jit_is_scripting:
        # cannot use `AttributeError` as it messes up with `nn.Module.__getattr__`
        raise RuntimeError(
            "`LightningModule.use_amp` was deprecated in v1.6 and is no longer accessible as of v1.8."
            " Please use `Trainer.amp_backend`.",
        )


def _use_amp_setter(_: LightningModule, __: bool) -> None:
    # Remove in v2.0.0
    # cannot use `AttributeError` as it messes up with `nn.Module.__getattr__`
    raise RuntimeError(
        "`LightningModule.use_amp` was deprecated in v1.6 and is no longer accessible as of v1.8."
        " Please use `Trainer.amp_backend`.",
    )


# Properties
LightningModule.use_amp = property(fget=_use_amp, fset=_use_amp_setter)

# Methods
LightningDataModule.on_save_checkpoint = _on_save_checkpoint
LightningDataModule.on_load_checkpoint = _on_load_checkpoint
