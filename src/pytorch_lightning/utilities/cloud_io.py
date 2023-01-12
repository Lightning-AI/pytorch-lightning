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
"""Utilities related to data saving/loading."""

from typing import Any

from lightning_fabric.utilities.cloud_io import _atomic_save as new_atomic_save
from lightning_fabric.utilities.cloud_io import _load as new_load
from lightning_fabric.utilities.cloud_io import get_filesystem as new_get_filesystem
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


def atomic_save(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.cloud_io.atomic_save` has been deprecated in v1.8.0 and will be"
        " removed in v2.0.0. This function is internal but you can copy over its implementation."
    )
    return new_atomic_save(*args, **kwargs)


def get_filesystem(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.cloud_io.get_filesystem` has been deprecated in v1.8.0 and will be"
        " removed in v2.0.0. Please use `lightning_fabric.utilities.cloud_io.get_filesystem` instead."
    )
    return new_get_filesystem(*args, **kwargs)


def load(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.cloud_io.load` has been deprecated in v1.8.0 and will be"
        " removed in v2.0.0. This function is internal but you can copy over its implementation."
    )
    return new_load(*args, **kwargs)
