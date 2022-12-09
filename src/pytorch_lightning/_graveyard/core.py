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

from pytorch_lightning import LightningDataModule


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


# Methods
LightningDataModule.on_save_checkpoint = _on_save_checkpoint
LightningDataModule.on_load_checkpoint = _on_load_checkpoint
