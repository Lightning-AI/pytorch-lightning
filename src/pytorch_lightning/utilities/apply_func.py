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

from lightning_lite.utilities.apply_func import apply_to_collection as new_apply_to_collection
from lightning_lite.utilities.apply_func import apply_to_collections as new_apply_to_collections
from lightning_lite.utilities.apply_func import convert_to_tensors as new_convert_to_tensors
from lightning_lite.utilities.apply_func import from_numpy as new_from_numpy
from lightning_lite.utilities.apply_func import move_data_to_device as new_move_data_to_device
from lightning_lite.utilities.apply_func import to_dtype_tensor as new_to_dtype_tensor
from lightning_lite.utilities.apply_func import TransferableDataType as NewTransferableDataType
from pytorch_lightning.utilities import rank_zero_deprecation


def apply_to_collection(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.apply_func.apply_to_collection` has been deprecated in v1.8.0 and will be"
        " removed in v1.10.0. Please use `lightning_lite.utilities.apply_func.apply_to_collection` instead."
    )
    return new_apply_to_collection(*args, **kwargs)


def apply_to_collections(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.apply_func.apply_to_collections` has been deprecated in v1.8.0 and will be"
        " removed in v1.10.0. Please use `lightning_lite.utilities.apply_func.apply_to_collections` instead."
    )
    return new_apply_to_collections(*args, **kwargs)


def convert_to_tensors(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.apply_func.convert_to_tensors` has been deprecated in v1.8.0 and will be"
        " removed in v1.10.0. Please use `lightning_lite.utilities.apply_func.convert_to_tensors` instead."
    )
    return new_convert_to_tensors(*args, **kwargs)


def from_numpy(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.apply_func.from_numpy` has been deprecated in v1.8.0 and will be"
        " removed in v1.10.0. Please use `lightning_lite.utilities.apply_func.from_numpy` instead."
    )
    return new_from_numpy(*args, **kwargs)


def move_data_to_device(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.apply_func.move_data_to_device` has been deprecated in v1.8.0 and will be"
        " removed in v1.10.0. Please use `lightning_lite.utilities.apply_func.move_data_to_device` instead."
    )
    return new_move_data_to_device(*args, **kwargs)


def to_dtype_tensor(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.apply_func.to_dtype_tensor` has been deprecated in v1.8.0 and will be"
        " removed in v1.10.0. Please use `lightning_lite.utilities.apply_func.to_dtype_tensor` instead."
    )
    return new_to_dtype_tensor(*args, **kwargs)


class TransferableDataType(NewTransferableDataType):
    def __init__(self) -> None:
        rank_zero_deprecation(
            "`pytorch_lightning.utilities.apply_func.TransferableDataType` has been deprecated in v1.8.0 and will be"
            " removed in v1.10.0. Please use `lightning_lite.utilities.apply_func.TransferableDataType` instead."
        )
        super().__init__()
