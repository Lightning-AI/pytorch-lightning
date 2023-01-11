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
"""Utilities used for collections."""

from typing import Any

import torch
from lightning_utilities.core.apply_func import apply_to_collection as new_apply_to_collection
from lightning_utilities.core.apply_func import apply_to_collections as new_apply_to_collections

from lightning_fabric.utilities import move_data_to_device as new_move_data_to_device
from lightning_fabric.utilities.apply_func import _from_numpy
from lightning_fabric.utilities.apply_func import _TransferableDataType as NewTransferableDataType
from lightning_fabric.utilities.apply_func import convert_to_tensors as new_convert_to_tensors
from pytorch_lightning.utilities import rank_zero_deprecation
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def apply_to_collection(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.apply_func.apply_to_collection` has been deprecated in v1.8.0 and will be"
        " removed in v2.0.0. Please use `lightning_utilities.core.apply_func.apply_to_collection` instead."
    )
    try:
        return new_apply_to_collection(*args, **kwargs)
    except ValueError as e:
        # upstream had to change the exception type
        raise MisconfigurationException from e


def apply_to_collections(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.apply_func.apply_to_collections` has been deprecated in v1.8.0 and will be"
        " removed in v2.0.0. Please use `lightning_utilities.core.apply_func.apply_to_collections` instead."
    )
    try:
        return new_apply_to_collections(*args, **kwargs)
    except ValueError as e:
        # upstream had to change the exception type
        raise MisconfigurationException from e


def convert_to_tensors(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.apply_func.convert_to_tensors` has been deprecated in v1.8.0 and will be"
        " removed in v2.0.0. Please use `lightning_fabric.utilities.apply_func.convert_to_tensors` instead."
    )
    return new_convert_to_tensors(*args, **kwargs)


def from_numpy(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.apply_func.from_numpy` has been deprecated in v1.8.0 and will be"
        " removed in v2.0.0. Please use `torch.from_numpy().to()` instead."
    )
    return _from_numpy(*args, **kwargs)


def move_data_to_device(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.apply_func.move_data_to_device` has been deprecated in v1.8.0 and will be"
        " removed in v2.0.0. Please use `lightning_fabric.utilities.apply_func.move_data_to_device` instead."
    )
    return new_move_data_to_device(*args, **kwargs)


def to_dtype_tensor(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.apply_func.to_dtype_tensor` has been deprecated in v1.8.0 and will be"
        " removed in v2.0.0. Please use `torch.tensor` instead."
    )
    return torch.tensor(*args, **kwargs)


class TransferableDataType(NewTransferableDataType):
    def __init__(self) -> None:
        rank_zero_deprecation(
            "`pytorch_lightning.utilities.apply_func.TransferableDataType` has been deprecated in v1.8.0 and will be"
            " removed in v2.0.0. This function is internal but you can copy over its implementation."
        )
        super().__init__()
